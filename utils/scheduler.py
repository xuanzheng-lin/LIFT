import numpy as np

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


class CosineWarmupScheduler:
    def __init__(self, optimizer, base_lr, warmup_length, steps):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_length = warmup_length
        self.steps = steps
        self.current_step = 0  # 内部维护当前步数

    def step(self):
        # 根据当前步数计算学习率并更新优化器
        if self.current_step < self.warmup_length:
            lr = _warmup_lr(self.base_lr, self.warmup_length, self.current_step)
        else:
            e = self.current_step - self.warmup_length
            es = self.steps - self.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.base_lr
        
        assign_learning_rate(self.optimizer, lr)
        self.current_step += 1  # 步数自动递增