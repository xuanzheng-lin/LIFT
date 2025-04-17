import torch
import torch.nn as nn
import torch.nn.functional as F


# some parts of this code are adapted from
# https://github.com/M4xim4l/InNOutRobustness/blob/main/utils/adversarial_attacks/utils.py

data_mean = [0.48145466, 0.4578275, 0.40821073]
data_std = [0.26862954, 0.26130258, 0.27577711]

mean = torch.tensor(data_mean).view(3, 1, 1)
std = torch.tensor(data_std).view(3, 1, 1)

def normalize(X):
    return (X - mean) / std

def clip_img_preprocessing(X):
    img_size = 224
    X = F.interpolate(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)
    return X


def project_perturbation(perturbation, eps, norm):
    if norm in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif norm in [2, 2.0, 'l2', 'L2', '2']:
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    else:
        raise NotImplementedError(f'Norm {norm} not supported')


def normalize_grad(grad, p):
    if p in ['inf', 'linf', 'Linf']:
        return grad.sign()
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        bs = grad.shape[0]
        grad_flat = grad.view(bs, -1)
        grad_normalized = F.normalize(grad_flat, p=2, dim=1)
        return grad_normalized.view_as(grad)
    

def pgd(
        forward,
        data_clean,
        targets,
        text_feature,
        norm="Linf",
        eps=8/255,
        iterations=10,
        stepsize=2/255,
        loss_fn=nn.CrossEntropyLoss(),
        random_start=False,
        mode='max',
        momentum=0.9,
        verbose=False
):
    """
    Minimize or maximize given loss
    """
    # make sure data is in image space
    assert torch.max(data_clean) < 1. + 1e-6 and torch.min(data_clean) > -1e-6
    global mean, std
    mean = mean.to(data_clean.device)
    std = std.to(data_clean.device)

    if not random_start:
        perturbation = torch.zeros_like(data_clean)
    else:
        perturbation = torch.empty_like(data_clean).uniform_(-eps, eps)
    perturbation = torch.clamp(perturbation, 0 - data_clean, 1 - data_clean)
    velocity = torch.zeros_like(data_clean)
    perturbation = perturbation.detach().requires_grad_(True)  # 初始梯度状态
    for i in range(iterations):
        normalized_data = clip_img_preprocessing(data_clean + perturbation)
        # print(normalized_data.requires_grad)  # 应为True
        with torch.enable_grad():
            out = forward(normalized_data, return_feature=True, use_tecoa=True, text_feature=text_feature)
            # print(out.requires_grad)  # 应为True
            loss = loss_fn(out, targets)
            if verbose:
                print(f'[{i}] {loss.item():.5f}')

        gradient = torch.autograd.grad(loss, perturbation, retain_graph=False, create_graph=False)[0]
        grad = gradient.detach()
        if grad.isnan().any():  #
            print(f'attention: nan in gradient ({grad.isnan().sum()})')  #
            grad[grad.isnan()] = 0.
        # normalize
        grad = normalize_grad(grad, p=norm)
        # momentum
        velocity = momentum * velocity + grad
        velocity = normalize_grad(velocity, p=norm)

        with torch.no_grad():
            # update
            if mode == 'min':
                perturbation = perturbation - stepsize * velocity
            elif mode == 'max':
                perturbation = perturbation + stepsize * velocity
            else:
                raise ValueError(f'Unknown mode: {mode}')
            # project
            perturbation = project_perturbation(perturbation, eps, norm)
            perturbation = torch.clamp(
                data_clean + perturbation, 0, 1
            ) - data_clean  # clamp to image space

            # 重新启用梯度跟踪
            perturbation = perturbation.detach().requires_grad_(True)

        assert not perturbation.isnan().any()
        assert torch.max(data_clean + perturbation) < 1. + 1e-6 and torch.min(
                data_clean + perturbation) > -1e-6

            # assert (ctorch.compute_norm(perturbation, p=self.norm) <= self.eps + 1e-6).all()
    # todo return best perturbation
    # problem is that model currently does not output expanded loss
    return perturbation.detach()
