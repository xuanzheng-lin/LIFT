import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224

import datasets
from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import *
from utils.templates import ZEROSHOT_TEMPLATES
from utils.visualizer import PGDVisualizer
from models.routing import Routing


def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        # if error, try state_dict format
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(backbone_name, prec):
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=True).eval()

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model


class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self.evaluator_spare = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None
        self.routing = Routing(cfg, self.model, 1)
        self.routing.to(self.device)

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if cfg.tte:
            if cfg.tte_mode == "fivecrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.FiveCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "tencrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.TenCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "randaug":
                _resize_and_flip = transforms.Compose([
                    transforms.RandomResizedCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.Lambda(lambda image: torch.stack([_resize_and_flip(image) for _ in range(cfg.randaug_times)])),
                    transforms.Normalize(mean, std),
                ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])

        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.train_loader = DataLoader(train_dataset,
            batch_size=cfg.micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=64, sampler=init_sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
            batch_size=32, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
        
        # 梯度累计方法训练大批量数据，计算梯度累计步数
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Total training points:", sum(self.cls_num_list))
        # print(self.cls_num_list)

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                print("No initialization with head")
            
            torch.cuda.empty_cache()
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                      {"params": self.head.parameters()}],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM": # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB": # https://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW": # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS": # https://arxiv.org/abs/2007.10740
            self.criterion == BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE": # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        
    def get_tokenized_prompts(self, classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        if cfg.prompt == "ensemble":
            all_text_features = []
            for template in tqdm(ZEROSHOT_TEMPLATES['imagenet']):
                prompts = self.get_tokenized_prompts(classnames, template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_text_features.append(text_features)
            all_text_features = torch.stack(all_text_features)
            text_features = all_text_features.mean(dim=0)
        elif cfg.prompt == "descriptor":
            with open("utils/descriptors_imagenet.json") as f:
                descriptors = json.load(f)
            template = "{}"
            all_class_features = []
            for cn in tqdm(classnames):
                prompts = self.get_tokenized_prompts(descriptors[cn], template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_class_features.append(text_features.mean(dim=0))
            text_features = torch.stack(all_class_features)
        elif cfg.prompt == "classname":
            template = "{}"
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        elif cfg.prompt == "default":
            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)

    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head.apply_weight(class_means)

    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    def train(self):
        cfg = self.cfg
        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)
        print(f"Initialize attack method {cfg.attack_method}")
        self.train_attack = self.create_attack(self.model, self.num_classes)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()
        
        best_rob = 0
        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)
                if not cfg.train_PGDAT:
                    if cfg.prec == "amp":
                        with autocast():
                            output = self.model(image)
                            loss = self.criterion(output, label)
                            loss_micro = loss / self.accum_step
                            self.scaler.scale(loss_micro).backward()
                        if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                            self.scaler.step(self.optim)
                            self.scaler.update()
                            self.optim.zero_grad()
                    else:
                        output = self.model(image)
                        loss = self.criterion(output, label)
                        loss_micro = loss / self.accum_step
                        loss_micro.backward()
                        if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                            self.optim.step()
                            self.optim.zero_grad()
                else:
                    # 划分 clean 和 adv 样本
                    batch_size = image.size(0)
                    attack_size = int(batch_size * cfg.attack_ratio)  # attack_ratio 为攻击比例
                    clean_size = batch_size - attack_size

                    # 随机选择攻击样本的索引
                    indices = torch.randperm(batch_size)
                    adv_indices = indices[:attack_size]
                    clean_indices = indices[attack_size:]

                    # 初始化 adv_image，默认和原始 image 相同
                    adv_image = image.clone()

                    adv_image[adv_indices] = self.train_attack(image[adv_indices], label[adv_indices])

                    # Mixup参数设置
                    mix_alpha = cfg.mix_alpha  # 例如1.0
                    mix_num = int(clean_size / 2)  # 选择Mixup的样本数量

                    if mix_alpha > 0 and mix_num > 0:
                        # 从clean和adv中各随机选择mix_num个样本
                        mix_clean_idx = torch.randperm(clean_size)[:mix_num]
                        mix_adv_idx = torch.randperm(attack_size)[:mix_num]
                        mix_clean = clean_indices[mix_clean_idx]
                        mix_adv = adv_indices[mix_adv_idx]

                        # 生成混合系数lambda
                        lam = torch.tensor(np.random.beta(mix_alpha, mix_alpha), device=adv_image.device).float()
                        mixed_images = lam * adv_image[mix_clean] + (1 - lam) * adv_image[mix_adv]

                        # 计算混合样本在两个分支的损失
                        if cfg.prec == "amp":
                            with autocast():
                                clean_mix_out = self.model(mixed_images, attack_supervise="clean")
                                adv_mix_out = self.model(mixed_images, attack_supervise="adv")
                        else:
                            clean_mix_out = self.model(mixed_images, attack_supervise="clean")
                            adv_mix_out = self.model(mixed_images, attack_supervise="adv")
                        
                        clean_mix_loss = lam * self.criterion(clean_mix_out, label[mix_clean]) + (1 - lam) * self.criterion(clean_mix_out, label[mix_adv])
                        adv_mix_loss = lam * self.criterion(adv_mix_out, label[mix_clean]) + (1 - lam) * self.criterion(adv_mix_out, label[mix_adv])

                        # 获取剩余样本索引
                        mask_clean = torch.ones(clean_size, dtype=torch.bool)
                        mask_clean[mix_clean_idx] = False
                        remaining_clean = clean_indices[mask_clean]
                        
                        mask_adv = torch.ones(attack_size, dtype=torch.bool)
                        mask_adv[mix_adv_idx] = False
                        remaining_adv = adv_indices[mask_adv]
                    else:
                        remaining_clean = clean_indices
                        remaining_adv = adv_indices
                        clean_mix_loss = 0.0
                        adv_mix_loss = 0.0

                    if cfg.prec == "amp":
                        with autocast():
                            # 分别计算 clean 和 adv 样本的输出和 loss
                            clean_output = self.model(adv_image[remaining_clean], attack_supervise="clean")
                            adv_output = self.model(adv_image[remaining_adv], attack_supervise="adv")
                            clean_loss = self.criterion(clean_output, label[remaining_clean]) + clean_mix_loss
                            adv_loss = self.criterion(adv_output, label[remaining_adv]) + adv_mix_loss
                            # 总 loss
                            loss = (clean_loss * clean_size + adv_loss * attack_size) / batch_size
                            loss_micro = loss / self.accum_step
                            self.scaler.scale(loss_micro).backward()
                        if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                            self.scaler.step(self.optim)
                            self.scaler.update()
                            self.optim.zero_grad()
                    else:
                        clean_output = self.model(adv_image[remaining_clean], attack_supervise="clean")
                        adv_output = self.model(adv_image[remaining_adv], attack_supervise="adv")        
                        clean_loss = self.criterion(clean_output, label[remaining_clean]) + clean_mix_loss
                        adv_loss = self.criterion(adv_output, label[remaining_adv]) + adv_mix_loss
                        loss = (clean_loss * clean_size + adv_loss * attack_size) / batch_size
                        loss_micro = loss / self.accum_step
                        loss_micro.backward()
                        if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                            self.optim.step()
                            self.optim.zero_grad()

                with torch.no_grad():
                    if not cfg.train_PGDAT:
                        pred = output.argmax(dim=1)
                        correct = pred.eq(label).float()
                    else:
                        # 初始化总预测结果和总标签
                        all_pred = []
                        all_label = []
                        # 收集未被混合的原始样本预测结果
                        if clean_output.numel() > 0:  # 确保存在剩余clean样本
                            all_pred.append(clean_output.argmax(dim=1))
                            all_label.append(label[remaining_clean])
                        if adv_output.numel() > 0:   # 确保存在剩余adv样本
                            all_pred.append(adv_output.argmax(dim=1))
                            all_label.append(label[remaining_adv])
                        # 如果有混合样本且需要包含（可选）
                        if mix_alpha > 0 and mix_num > 0 and cfg.include_mixup_in_acc:
                            # 使用混合样本中权重更大的类别作为伪标签
                            pseudo_label = torch.where(lam > 0.5, 
                                                    label[mix_clean], 
                                                    label[mix_adv])
                            all_pred.extend([
                                clean_mix_out.argmax(dim=1),
                                adv_mix_out.argmax(dim=1)
                            ])
                            all_label.extend([pseudo_label, pseudo_label])
                        # 合并所有结果
                        pred = torch.cat(all_pred, dim=0)
                        true_label = torch.cat(all_label, dim=0)
                        correct = pred.eq(true_label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_accs))
                many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"clean_loss {clean_loss.item():.4f} adv_loss {adv_loss.item():.4f}"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/loss.clean", clean_loss.item(), n_iter)
                self._writer.add_scalar("train/loss.adv", adv_loss.item(), n_iter)
                self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
                end = time.time()

            self.sched.step()
            torch.cuda.empty_cache()

            if cfg.evaluate_interval:
                if epoch_idx % cfg.interval == 0 :
                    clean, rob, _ = evaluate_interval(self.model, self.test_loader, cfg, self.num_classes, wandb=None, epoch=epoch_idx)
                if rob >= best_rob : 
                    best_epoch, best_clean, best_rob = epoch_idx, clean, rob
                    save_dir = os.path.join(cfg.output_dir, "bestcheckpoint") 
                    os.makedirs(save_dir, exist_ok=True)
                    best_save_fname = os.path.join(save_dir, time_start + 'total_epochs_{:d}_best.pt'.format(num_epochs))
                    if os.path.exists(best_save_fname):
                        os.remove(best_save_fname)
                    torch.save(self.model.state_dict(),best_save_fname)

                    info = []
                    info += [f"epoch {best_epoch} has best robustness now"]
                    info += [f"clean acc {best_clean:.4f}"]
                    info += [f"rob acc {best_rob:.4f}"]
                    print(" ".join(info))

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)

        self.test()

        # Close writer
        self._writer.close()

    def post_train(self):
        cfg = self.cfg
        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)
        # freeze finetuned CLIP model 
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.model.eval()
        self.evaluator.reset()

        self.optim = torch.optim.Adam([{"params": self.routing.parameters()}],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        clean_logits_diff = []
        adv_logits_diff = []
        clean_attribution_diff = [1]
        adv_attribution_diff = [1]

        num_epochs = 1
        for epoch_idx in range(num_epochs):
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)

                # 划分 clean 和 adv 样本
                batch_size = image.size(0)
                attack_size = int(batch_size * cfg.attack_ratio)  # attack_ratio 为攻击比例
                clean_size = batch_size - attack_size

                # 随机选择攻击样本的索引
                indices = torch.randperm(batch_size)
                adv_indices = indices[:attack_size]
                clean_indices = indices[attack_size:]

                # 初始化 adv_image，默认和原始 image 相同
                adv_image = image.clone()

                adv_image[adv_indices] = PGD(image[adv_indices], label[adv_indices], self.model, steps=10)
                self.model.eval()

                if cfg.prec == "amp":
                    with autocast():
                        # 分别计算 clean 和 adv 样本的输出和 loss
                        clean_features, clean_output = self.routing(adv_image[clean_indices])
                        adv_features, adv_output = self.routing(adv_image[adv_indices])
                        bce_loss_fn = nn.BCEWithLogitsLoss()
                        clean_loss = bce_loss_fn(clean_output, torch.zeros(clean_output.shape[0], device=clean_output.device))
                        adv_loss = bce_loss_fn(adv_output, torch.ones(adv_output.shape[0], device=adv_output.device))
                        # 总 loss
                        loss = (clean_loss * clean_size + adv_loss * attack_size) / batch_size
                        loss_micro = loss / self.accum_step
                        self.scaler.scale(loss_micro).backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    clean_features, clean_output = self.routing(adv_image[clean_indices])
                    adv_features, adv_output = self.routing(adv_image[adv_indices])
                    bce_loss_fn = nn.BCEWithLogitsLoss()
                    clean_loss = bce_loss_fn(clean_output, torch.zeros(clean_output.shape[0], device=clean_output.device))
                    adv_loss = bce_loss_fn(adv_output, torch.ones(adv_output.shape[0], device=adv_output.device))
                    loss = (clean_loss * clean_size + adv_loss * attack_size) / batch_size
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                            
                clean_logits_diff.extend(clean_features[:, 0].cpu().detach().numpy())
                #clean_attribution_diff.extend(clean_features[:, 1].cpu().detach().numpy())
                adv_logits_diff.extend(adv_features[:, 0].cpu().detach().numpy())
                #adv_attribution_diff.extend(adv_features[:, 1].cpu().detach().numpy()) 
                

                with torch.no_grad():
                    combined_output = torch.cat((clean_output, adv_output), dim=0)
                    pred = (torch.sigmoid(combined_output) > 0.5).int()
                    correct = pred.eq(torch.cat([torch.zeros(clean_output.shape[0], device=pred.device), torch.ones(adv_output.shape[0], device=pred.device)], dim=0)).int()

                acc = correct.sum().item() / correct.numel()
                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                batch_time.update(time.time() - end)

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"clean_loss {clean_loss.item():.4f} adv_loss {adv_loss.item():.4f}"]
                    info += [f"acc {acc:.4e}"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/loss.clean", clean_loss.item(), n_iter)
                self._writer.add_scalar("train/loss.adv", adv_loss.item(), n_iter)
                
                end = time.time()

                if batch_idx + 1 == 200:
                    self.routing.visualize_diff(clean_logits_diff, adv_logits_diff, clean_attribution_diff, adv_attribution_diff, directory=cfg.output_dir)
                    break

            self.sched.step()
            torch.cuda.empty_cache()

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_routing_model(cfg.output_dir)

        # Close writer
        self._writer.close()

    def router_train(self):
        cfg = self.cfg
        # freeze finetuned CLIP model 
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.model.eval()
        self.evaluator.reset()

        num_epochs = 1
        for epoch_idx in range(num_epochs):
            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)

                # 划分 clean 和 adv 样本
                batch_size = image.size(0)
                attack_size = int(batch_size * cfg.attack_ratio)  # attack_ratio 为攻击比例
                clean_size = batch_size - attack_size

                # 随机选择攻击样本的索引
                indices = torch.randperm(batch_size)
                adv_indices = indices[:attack_size]
                clean_indices = indices[attack_size:]

                # 初始化 adv_image，默认和原始 image 相同
                adv_image = image.clone()

                adv_image[adv_indices] = PGD(image[adv_indices], label[adv_indices], self.model, steps=10)
                self.model.eval()

                self.routing.add_data(images=adv_image[clean_indices], is_adv=torch.zeros(clean_size, device=adv_image.device))
                self.routing.add_data(images=adv_image[adv_indices], is_adv=torch.ones(attack_size, device=adv_image.device))

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += ["data added"]
                    print(" ".join(info))
                if batch_idx + 1 == 100:
                    break
                
        self.threshold = self.routing.find_optimal_threshold()
        print(f"The router threshold of dataset {cfg.dataset} is {self.threshold}")
        self.router_test()

    def router_test(self):
        cfg = self.cfg
        # freeze finetuned CLIP model 
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.model.eval()
        self.evaluator.reset()

         # 初始化累计统计变量
        total_clean_correct = 0
        total_adv_correct = 0
        total_clean = 0
        total_adv = 0

        for batch_idx, batch in enumerate(self.test_loader):

            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)
            
            # 划分 clean 和 adv 样本
            batch_size = image.size(0)
            attack_size = int(batch_size * cfg.attack_ratio)  # attack_ratio 为攻击比例
            clean_size = batch_size - attack_size

            # 随机选择攻击样本的索引
            indices = torch.randperm(batch_size)
            adv_indices = indices[:attack_size]
            clean_indices = indices[attack_size:]

            # 初始化 adv_image，默认和原始 image 相同
            adv_image = image.clone()

            adv_image[adv_indices] = PGD(image[adv_indices], label[adv_indices], self.model, steps=10)
            self.model.eval()

            clean_pred, clean_prob = self.routing(adv_image[clean_indices])
            adv_pred, adv_prob = self.routing(adv_image[adv_indices])


            with torch.no_grad():
                combined_pred = torch.cat((clean_pred, adv_pred), dim=0)
                pred_labels = (combined_pred > 0.5).int()
                true_labels = torch.cat([
                torch.zeros(clean_size, device=pred_labels.device),
                torch.ones(attack_size, device=pred_labels.device)
                ])
                correct = pred_labels.eq(true_labels).int()

                # 统计当前batch的正确数
                batch_clean_correct = correct[:clean_size].sum().item()
                batch_adv_correct = correct[clean_size:].sum().item()

                # 累加统计量
                total_clean_correct += batch_clean_correct
                total_adv_correct += batch_adv_correct
                total_clean += clean_size
                total_adv += attack_size
        
        # 计算最终准确率
        clean_acc = total_clean_correct / total_clean if total_clean > 0 else 0
        adv_acc = total_adv_correct / total_adv if total_adv > 0 else 0
        overall_acc = (total_clean_correct + total_adv_correct) / (total_clean + total_adv)
    
        # 打印详细统计信息
        print(f"Clean Samples: Total={total_clean}, Correct={total_clean_correct}, Accuracy={clean_acc:.4f}")
        print(f"Adversarial Samples: Total={total_adv}, Correct={total_adv_correct}, Accuracy={adv_acc:.4f}")
        print(f"All Samples: Total={total_clean + total_adv}, Correct={total_clean_correct + total_adv_correct}, Accuracy={overall_acc:.4f}")

    def create_attack(self, model, num_classes):
        cfg = self.cfg
        attack_map = {
            'pgd': lambda: torchattacks.PGD(model, random_start=True, steps=10),
            'fgsm': lambda: torchattacks.FGSM(model),
            'auto_attack': lambda: torchattacks.AutoAttack(model, n_classes=num_classes)
        }
        
        # 获取攻击构造函数并创建实例
        attack = attack_map.get(cfg.attack_method.lower())
        if attack is None:
            raise ValueError(f"Unsupported attack method: {cfg.attack_method}")
            
        return attack()
    
    def test(self, mode="test"):
        cfg = self.cfg
        # self.visualizer = PGDVisualizer(self.model, self.device)

        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        if self.routing is not None:
            self.routing.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        if cfg.test_attack:
            self.test_attack = self.create_attack(self.model, self.num_classes)

        routing_correct_count = 0

        for idx, batch in enumerate(tqdm(data_loader, ascii=True)):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            if _ncrops <= 5:
                if cfg.test_attack:
                    adv_image = image.clone()
                    # with torch.no_grad():
                    adv_image = self.test_attack(image, label.repeat_interleave(_ncrops))

                    if cfg.use_routing:
                        original_indices = torch.arange(adv_image.size(0), device=self.device)
                        output = self.model(adv_image)
                        pred = output.max(1)[1]
                        _ , attack_prob = self.routing(adv_image, pred.repeat_interleave(_ncrops))
                        is_attacked = attack_prob > 0.5
                        attacked_indices = original_indices[is_attacked]
                        clean_indices = original_indices[~is_attacked]
                        attack_image = adv_image[is_attacked]
                        clean_image = adv_image[~is_attacked]
                        if attack_image.size(0) > 0:
                            attack_output = self.model(attack_image, attack_supervise="adv")
                        if clean_image.size(0) > 0:    
                            clean_output = self.model(clean_image, attack_supervise="clean")
                        output = torch.zeros(adv_image.size(0), attack_output.size(-1), device=self.device)  # 初始化完整 output 张量
                        if attack_image.size(0) > 0:
                            output[attacked_indices] = attack_output
                        if clean_image.size(0) > 0:    
                            output[clean_indices] = clean_output

                        # 更新计数器
                        routing_correct_count += (is_attacked.sum().item())
                    else:
                        output = self.model(adv_image, attack_supervise="adv")
                else:
                    output = self.model(image)
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    if cfg.test_attack:
                        adv_image = self.test_attack(image[:, k], label)
                        output.append(self.model(adv_image, attack_supervise="adv"))
                    else:
                        output.append(self.model(image[:, k]))
                output = torch.stack(output).mean(dim=0)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        if cfg.use_routing:
            print(f"Routing correct classifications: {routing_correct_count}")

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]

    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
            "head": head_dict
        }

        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict, strict=False)

        if head_dict["weight"].shape == self.head.weight.shape:
            self.head.load_state_dict(head_dict, strict=False)

    def save_routing_model(self, directory):
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(self.routing.fc.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_routing_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))
        state_dict = torch.load(load_path, map_location=self.device)
        print("Loading weights to from {}".format(load_path))
        self.routing.fc.load_state_dict(state_dict, strict=True)

    def bi_channel_test(self, mode="test"):
        cfg = self.cfg

        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        if self.routing is not None:
            self.routing.eval()
        self.evaluator.reset()
        self.evaluator_spare.reset()

        # 初始化统计变量
        both_correct = 0
        clean_only = 0
        adv_only = 0
        neither = 0
        total_samples = 0

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        # 定义处理多crops的函数
        def process_crops(image, model, attack_supervise):
            bsz, ncrops, c, h, w = image.size()
            if ncrops <= 5:
                image_flat = image.view(-1, c, h, w)
                output = model(image_flat, attack_supervise=attack_supervise)
                output = output.view(bsz, ncrops, -1).mean(1)
            else:
                outputs = []
                for i in range(ncrops):
                    crop = image[:, i, :, :, :]
                    out = model(crop, attack_supervise=attack_supervise)
                    outputs.append(out)
                output = torch.stack(outputs).mean(0)
            return output

        # 创建攻击实例
        self.test_attack = self.create_attack(self.model, self.num_classes)

        for idx, batch in enumerate(tqdm(data_loader, ascii=True)):
            image = batch[0].to(self.device)
            label = batch[1].to(self.device)
            bsz, ncrops, c, h, w = image.size()
            total_samples += bsz  # 按batch累积样本总数

            # Clean分支处理
            clean_output = process_crops(image, self.model, "clean")
            clean_pred = clean_output.argmax(dim=1)
            clean_correct = (clean_pred == label).cpu().numpy()

            # 生成对抗样本
            image_flat = image.view(bsz * ncrops, c, h, w)
            labels_flat = label.repeat_interleave(ncrops)
            adv_image_flat = self.test_attack(image_flat, labels_flat)
            adv_image = adv_image_flat.view(bsz, ncrops, c, h, w)

            # Adv分支处理
            adv_output = process_crops(adv_image, self.model, "adv")
            adv_pred = adv_output.argmax(dim=1)
            adv_correct = (adv_pred == label).cpu().numpy()

            # 统计四种情况
            batch_both = np.sum(np.logical_and(clean_correct, adv_correct))
            batch_clean_only = np.sum(np.logical_and(clean_correct, ~adv_correct))
            batch_adv_only = np.sum(np.logical_and(~clean_correct, adv_correct))
            batch_neither = np.sum(np.logical_and(~clean_correct, ~adv_correct))

            both_correct += batch_both
            clean_only += batch_clean_only
            adv_only += batch_adv_only
            neither += batch_neither

            self.evaluator.process(clean_output, label)
            self.evaluator_spare.process(adv_output, label)
        
        self.evaluator.evaluate()
        self.evaluator_spare.evaluate()
        
        # 输出统计结果
        print(f"\nClean分支正确数: {both_correct + clean_only}\n")
        print(f"Adv分支正确数: {both_correct + adv_only}\n")
        print(f"两分支都正确: {both_correct}\n")
        print(f"仅Clean正确: {clean_only}\n")
        print(f"仅Adv正确: {adv_only}\n")
        print(f"两分支都错误: {neither}\n")
        print(f"总样本数: {total_samples}")

    def aa_test(self, mode="test"):
        cfg = self.cfg

        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        if self.routing is not None:
            self.routing.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        autoattack = AutoAttack(self.model, norm='Linf', eps=8/255, seed=cfg.seed, version='standard')
        x_total = [x for (x, y) in data_loader]
        y_total = [y for (x, y) in data_loader]
        x_total = torch.cat(x_total, 0)
        y_total = torch.cat(y_total, 0)
        _, AA_acc = autoattack.run_standard_evaluation(x_total, y_total)
        print(f"The accuray under AutoAttack is: {AA_acc}")