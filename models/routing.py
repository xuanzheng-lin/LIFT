from math import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Routing(nn.Module):
    def __init__(self, cfg, finetuned_clip_model, input_dim):
        super(Routing, self).__init__()
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )
        self.model = finetuned_clip_model

    def make_noise(self, x_batch,spread):
        new_x_batch = []
        for x in x_batch:
            x = x.data.cpu().numpy()
            stdev = spread * (np.max(x)-np.min(x))
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            x_plus_noise = np.clip(x_plus_noise, 0, 1)
            x_plus_noise = torch.from_numpy(x_plus_noise).cpu()
            new_x_batch.append(x_plus_noise)
        new_batch = torch.stack(new_x_batch).to(self.device)
        return new_batch

    def calculate_attribution_difference(self, inputs, noisy_inputs, target_label=None):
        """
        计算特征归因差异, 使用Integrated Gradients方法。
        
        参数:
        - model: 微调后的CLIP模型
        - inputs: 原始输入
        - noisy_inputs: 加噪声后的输入
        - target_label: 可选，指定目标类别（如针对分类任务）
        
        返回:
        - 特征归因差异 (batch_size,)
        """
        ig = IntegratedGradients(self.model)
        
        # 计算原始输入的特征归因
        attributions_original = ig.attribute(inputs, target=target_label)
        
        # 计算加噪声输入的特征归因
        attributions_noisy = ig.attribute(noisy_inputs, target=target_label)
        
        # 计算归因差异 (L2范数或者绝对差值)
        attribution_diff = torch.norm(attributions_original - attributions_noisy, p=inf, dim=(1, 2, 3))
        return attribution_diff

    def extract_features_for_routing(self, inputs, spread, target_label=None):
        """
        从微调后的CLIP模型中提取用于路由分类器的特征:
        1. logits差异(高斯噪声前后logits变化)。
        2. 特征归因差异（高斯噪声前后特征变化）。
        """
        # 添加高斯噪声
        noisy_inputs = self.make_noise(inputs, spread=spread)
        
        # 提取logits（模型已微调）
        logits_original = self.model(inputs, return_feature=True)
        logits_noisy = self.model(noisy_inputs, return_feature=True)
        
        # Logits差异
        logits_diff = torch.norm(logits_original - logits_noisy, p=inf, dim=1)
        
        # 特征归因差异
        attribution_diff = self.calculate_attribution_difference(inputs, noisy_inputs, target_label)
        
        # 返回特征向量
        return torch.stack([logits_diff, attribution_diff], dim=1)  # [batch_size, 2]

    def forward(self, images, labels=None, spread=0.35):
        features = self.extract_features_for_routing(images, spread, labels)
        features = features.to(self.fc[0].weight.dtype)
        return features, self.fc(features).squeeze()  # 输出对应概率
    
    def visualize_diff(self, logits_diff_clean, logits_diff_adv, attribution_diff_clean, attribution_diff_adv, directory=None):
        """
        可视化干净样本和对抗样本在高斯噪声下的logits差异和特征归因差异
        
        参数:
        logits_diff_clean: 干净样本的logits差异
        logits_diff_adv: 对抗样本的logits差异
        attribution_diff_clean: 干净样本的特征归因差异
        attribution_diff_adv: 对抗样本的特征归因差异
        """
        # 创建一个新的图形
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 绘制logits差异的分布
        axes[0].hist(logits_diff_clean, bins=30, alpha=0.5, label='Clean Samples', color='blue')
        axes[0].hist(logits_diff_adv, bins=30, alpha=0.5, label='Adversarial Samples', color='red')
        axes[0].set_title('Logits Difference Distribution')
        axes[0].set_xlabel('Logits Difference')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # 绘制特征归因差异的分布
        axes[1].hist(attribution_diff_clean, bins=30, alpha=0.5, label='Clean Samples', color='blue')
        axes[1].hist(attribution_diff_adv, bins=30, alpha=0.5, label='Adversarial Samples', color='red')
        axes[1].set_title('Attribution Difference Distribution')
        axes[1].set_xlabel('Attribution Difference')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        # 显示图形
        plt.tight_layout()
        if directory:
            save_path = os.path.join(directory, "diff_plot.png")
            plt.savefig(save_path)
            print(f"图片已保存至: {save_path}")
        plt.show()