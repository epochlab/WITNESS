#!/usr/bin/env python3

# eps | Higher = Less imperceptible (0.06)
# step | Set smaller than eps (0.02)
# iters | Higher = Stronger (1000)

import torch
from tqdm import trange

def fgsm(x, model, eps=0.1):
    x_adv = x.clone().detach()
    x_adv.requires_grad = True

    loss = model(x_adv).latent_dist.mean.norm()
    grad = torch.autograd.grad(loss, [x_adv])[0]
    x_adv = x_adv.detach() + eps * grad.sign()
    x_adv = torch.clamp(x_adv, -1, 1)
    return x_adv

def bim(x, model, eps=0.1, step_size=0.01, iters=10):
    x_adv = x.clone().detach()

    for _ in (t := trange(iters)):
        x_adv.requires_grad = True

        loss = model(x_adv).latent_dist.mean.norm()
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * grad.sign()

        x_adv = torch.clamp(x_adv, x-eps, x+eps)
        x_adv = torch.clamp(x_adv, -1, 1)

        t.set_description(f"[Running attack] - Loss: {loss.item():.2f}")

    return x_adv

def pgd(x, model, eps=0.1, step_size=0.01, iters=100):
    x_adv = x.clone().detach() + (torch.rand_like(x) * 2 * eps - eps)

    for i in (t := trange(iters)):
        actual_step_size = step_size * (1 - i / iters / 100)

        x_adv.requires_grad = True

        loss = model(x_adv).latent_dist.mean.norm()
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() - grad.sign() * actual_step_size

        x_adv = torch.clamp(x_adv, x-eps, x+eps)
        x_adv = torch.clamp(x_adv, -1, 1)

        t.set_description(f"[Running attack] - Loss: {loss.item():.2f} | Actual Step: {actual_step_size:.4f}")

    return x_adv