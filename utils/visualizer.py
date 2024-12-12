import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchattacks import PGD

class PGDVisualizer:
    def __init__(self, model, device):
        """
        初始化 PGDVisualizer
        :param model: 目标模型
        :param device: 设备 (如 'cuda' 或 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def _save_combined_image(self, images, titles, filename):
        """将多张图像保存到一个文件中作为子图"""
        num_images = len(images)
        plt.figure(figsize=(4 * num_images, 4))
        for i, (image_tensor, title) in enumerate(zip(images, titles)):
            image = image_tensor.cpu().detach()
            if image.ndimension() == 4:  # Batch of images
                image = image[0]
            image = image.permute(1, 2, 0)  # C, H, W -> H, W, C
            image = image.clamp(0, 1)  # 确保像素值在 [0, 1] 范围内

            plt.subplot(1, num_images, i + 1)
            plt.imshow(image)
            plt.title(title)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def visualize(self, image, label, epsilon=0.3, steps=5, output_dir="./"):
        """
        可视化图像在未攻击和经过PGD攻击后（steps=0 和正常攻击）的对比，并保存到一个文件中
        :param image: 输入图像 (Tensor)
        :param label: 图像对应的标签 (Tensor)
        :param epsilon: 攻击强度 (默认 0.3)
        :param steps: 攻击步数 (默认 5)
        :param output_dir: 保存图像的目录
        """
        # 将图像和标签移动到设备
        image = image.to(self.device)
        label = label.to(self.device)

        # 反归一化变换
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        inv_transform = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )

        # 还原原始图像
        def untransform(image):
            # 逆归一化
            image = inv_transform(image)
            # 逆 Lambda 和 CenterCrop
            image = image.squeeze(0)  # 从 torch.stack 还原单张图像
            return image

        original_image = untransform(image.clone())

        # PGD 攻击实例
        pgd_attack_zero = PGD(self.model, eps=epsilon, steps=0, random_start=False)
        pgd_attack = PGD(self.model, eps=epsilon, steps=steps, random_start=False)

        # 经过 PGD (steps=0) 的图像
        attacked_image_zero = pgd_attack_zero(image, label)
        attacked_image_zero = untransform(attacked_image_zero.clone())

        # 经过 PGD (正常攻击) 的图像
        attacked_image = pgd_attack(image, label)
        attacked_image = untransform(attacked_image.clone())

        # 保存组合图像
        images = [original_image, attacked_image_zero, attacked_image]
        titles = ["Original Image", "PGD Attacked (steps=0)", f"PGD Attacked (steps={steps})"]
        self._save_combined_image(images, titles, f"{output_dir}/combined_pgd_visualization.png")
