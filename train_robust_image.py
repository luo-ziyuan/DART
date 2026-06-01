# -*- coding: utf-8 -*-
'''Train with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import StructuralSimilarityIndexMeasure
import copy

from imageio import imsave

import torchvision
import torchvision.transforms as transforms

import os
import configargparse

from utils import progress_bar
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from run_nerf_helpers import *
# import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
import time
import utils_img
import utils

import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_repo_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def build_image_loss(loss_name):
    if loss_name == 'mse':
        return lambda imgs_w, imgs: torch.mean((imgs_w - imgs) ** 2)
    raise NotImplementedError(
        "This public release supports only --loss_i mse. "
        "Optional perceptual-loss dependencies from the internal research codebase are not bundled."
    )

def setup_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def config_parser():
    parser = configargparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--classifier_path", required=False, default='./checkpoint_cifar100/PreActResNet18_ckpt.pth',
                        help='Classifier path')
    parser.add_argument("--base_log_dir", type=str, required=True,
                        help='base_log_dir')
    parser.add_argument("--msg_decoder_path", type=str, required=True,
                        help='msg_log_dir')
    
    
    parser.add_argument("--test_dir", type=str, required=True,
                        help='Classifier path')
    parser.add_argument("--save_inr", action='store_true',
                        help='save inr weights')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

    parser.add_argument('--trainbs', default=1, type=int, help='trainloader batch size')
    parser.add_argument('--testbs', default=1, type=int, help='testloader batch size')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    
    parser.add_argument("--method", type=str, default='normal', choices=["normal", "robust_inr"],
                        help='INR training method')
    parser.add_argument('--train_net_PGD_epsilon', default=0.0004, type=float, help='attack_noise_std')
    # parser.add_argument('--train_net_PGD_alpha', default=1.6/255, type=float, help='attack_noise_std')
    parser.add_argument('--train_net_PGD_num_iter', default=20, type=int, help='attack_noise_std')

    parser.add_argument("--attack_clean", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--attack_noise", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument('--attack_noise_std', default=0.0, type=float, help='attack_noise_std')

    parser.add_argument("--attack_net_FGSM", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument('--attack_net_FGSM_epsilon', default=0.001, type=float, help='attack_net_FGSM_epsilon')
    parser.add_argument("--attack_net_PGD", action='store_true',
                        help='attack_net_PGD')
    parser.add_argument('--attack_net_PGD_epsilon', default=0.0004, type=float, help='attack_noise_std')
    # parser.add_argument('--attack_net_PGD_alpha', default=1.6/255, type=float, help='attack_noise_std')
    parser.add_argument('--attack_net_PGD_num_iter', default=20, type=int, help='attack_noise_std')

    parser.add_argument('--attack_net_DSPGD_net_epsilon', default=0.0004, type=float, help='attack_noise_std')
    # parser.add_argument('--attack_net_PGD_alpha', default=1.6/255, type=float, help='attack_noise_std')
    parser.add_argument('--attack_net_DSPGD_num_iter', default=20, type=int, help='attack_noise_std')
    parser.add_argument('--attack_net_DSPGD_image_epsilon_start', default=8.0, type=float, help='image_epsilon')
    parser.add_argument('--attack_net_DSPGD_image_epsilon_end', default=8.0, type=float, help='image_epsilon')
    parser.add_argument('--attack_net_DSPGD_image_epsilon_interval', default=2.0, type=float, help='image_epsilon')
    parser.add_argument('--attack_net_DSPGD_image_alpha', default=0.1, type=float, help='image_epsilon')


    parser.add_argument("--attack_net_CW", action='store_true',
                        help='attack_net_CW')
    parser.add_argument('--attack_net_CW_epsilon', default=0.0004, type=float, help='attack_noise_std')
    # parser.add_argument('--attack_net_PGD_alpha', default=1.6/255, type=float, help='attack_noise_std')
    parser.add_argument('--attack_net_CW_num_iter', default=30, type=int, help='attack_noise_std')    

    parser.add_argument("--not_targeted", action='store_true',
                        help='not_targeted_attack')

    parser.add_argument("--attack_net_DSPGD", action='store_true',
                        help='attack_net_DSPGD')

    parser.add_argument("--multires", type=int, default=20,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--H", type=int, default=128,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--W", type=int, default=128,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--N_iters", type=int, default=1000,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument('--lrate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lambda_w', default=1.0, type=float, help='lambda_w')
    parser.add_argument('--lambda_i', default=1.0, type=float, help='lambda_i')

    parser.add_argument('--start_img', type=int, default=0, help='lambda_2')
    parser.add_argument('--end_img', type=int, default=9999, help='lambda_2')

    
    parser.add_argument("--loss_i", type=str, default="mse", help="Type of loss for the image loss. The public release supports only mse.")
    parser.add_argument("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    
    parser.add_argument("--attack_mode", type=str, default="all", help="attack on image")
    
    parser.add_argument("--data_augmentation", action='store_true', help="Type of data augmentation to use at marking time. (Default: combined)")
    # parser.add_argument("--p_crop", type=float, default=1.0, help="Probability of the crop augmentation. (Default: 0.5)")
    # parser.add_argument("--p_res", type=float, default=1.0, help="Probability of the crop augmentation. (Default: 0.5)")
    # parser.add_argument("--p_blur", type=float, default=0.0, help="Probability of the blur augmentation. (Default: 0.5)")
    # parser.add_argument("--p_jpeg", type=float, default=1.0, help="Probability of the diff JPEG augmentation. (Default: 0.5)")
    # parser.add_argument("--p_rot", type=float, default=0.5, help="Probability of the rotation augmentation. (Default: 0.5)")
    # parser.add_argument("--p_color_jitter", type=float, default=0.0, help="Probability of the color jitter augmentation. (Default: 0.5)")
    
    
    args = parser.parse_args()

    return args

def run_network(inputs, fn, embed_fn):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    outputs_flat = fn(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, i=0)
    output_ch = 3
    model = NeRF(D=5, W=256,
                     input_ch=input_ch, output_ch=output_ch).to(device)
    grad_vars = list(model.parameters())

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_iters, eta_min=0.0001)

    return model, embed_fn, grad_vars, optimizer, scheduler

def create_direct_image(args):
#     embed_fn, input_ch = get_embedder(args.multires, i=0)
#     output_ch = 3
    model = torch.randn(1, 3, args.H, args.W, requires_grad=True).to(device)
    grad_vars = [model]

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_iters, eta_min=0.0001)

    return model, grad_vars, optimizer, scheduler

def attack_image_FGSM(msg_decoder, loss_w, loss_i, image, msg, target_msg=None, vqgan_to_imnet=None, epsilon=None, args=None):
    msg_decoder.eval()
    # 将图像和标签移动到设备上
    image = image.to(device)
    msg = msg.to(device)
    # 设置requires_grad属性，以便计算图像梯度
    image.requires_grad = True
    # 前向传播
    output = msg_decoder(vqgan_to_imnet(image))
    if target_msg is not None:
        # 计算损失
        loss = loss_w(output, target_msg)
        # 计算梯度
        msg_decoder.zero_grad()
        loss.backward()
        data_grad = image.grad.data
        # 使用梯度和epsilon计算扰动
        sign_data_grad = data_grad.sign()
        perturbed_image = image - epsilon * sign_data_grad
        # 将图像像素值裁剪到0-1之间
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    else:
        # 计算损失
        loss = loss_w(output, msg)
        # 计算梯度
        msg_decoder.zero_grad()
        loss.backward()
        data_grad = image.grad.data
        # 使用梯度和epsilon计算扰动
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        # 将图像像素值裁剪到0-1之间
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def attack_image_PGD_linf(msg_decoder, loss_w, loss_i, image,
                                                msg, target_msg=None, vqgan_to_imnet=None, epsilon=8 / 255, alpha=1.6 / 255, num_iter=100, args=None):

    image = image.clone().detach()
    if target_msg is not None:
        target_msg = target_msg.clone().detach()
    else:
        msg = msg.clone().detach()

    adv_image = image.clone().detach()

    # Starting at a uniformly random point
    adv_image = adv_image + torch.empty_like(adv_image).uniform_(-epsilon, epsilon)
    adv_image = torch.clamp(adv_image, min=0, max=1).detach()

    if target_msg is not None:
        for _ in range(num_iter):
            adv_image.requires_grad = True
            outputs = msg_decoder(vqgan_to_imnet(adv_image))

            # Calculate loss
            cost = loss_w(outputs, target_msg)
            msg_decoder.zero_grad()
            if adv_image.grad is not None:
                adv_image.grad.zero_()
            # Update adversarial image
            grad = torch.autograd.grad(cost, adv_image,
                                    retain_graph=False, create_graph=False)[0]
            
            adv_image = adv_image.detach() - alpha*grad.sign()
            delta = torch.clamp(adv_image - image,
                                min=-epsilon, max=epsilon)
            adv_image = torch.clamp(image + delta, min=0, max=1).detach()
    else:
        for _ in range(num_iter):
            adv_image.requires_grad = True
            outputs = msg_decoder(vqgan_to_imnet(adv_image))

            # Calculate loss
            cost = loss_w(outputs, msg)
            msg_decoder.zero_grad()
            if adv_image.grad is not None:
                adv_image.grad.zero_()
            # Update adversarial image
            grad = torch.autograd.grad(cost, adv_image,
                                    retain_graph=False, create_graph=False)[0]
            
            adv_image = adv_image.detach() + alpha*grad.sign()
            delta = torch.clamp(adv_image - image,
                                min=-epsilon, max=epsilon)
            adv_image = torch.clamp(image + delta, min=0, max=1).detach()

    return adv_image


def attack_inr_FGSM(msg_decoder, loss_w, loss_i, inr, embed_fn, msg, target_msg=None, vqgan_to_imnet=None, epsilon=None, args=None):
    inr_net = copy.deepcopy(inr)
    msg_decoder.eval()
    for param in inr_net.parameters():
        param.requires_grad = True
    # 将图像和标签移动到设备上
    image = inference_inr(inr_net, embed_fn, args)
    image = image.permute(2, 0, 1).unsqueeze(0)
    # 设置requires_grad属性，以便计算图像梯度
    # image.requires_grad = True
    # 前向传播
    output = msg_decoder(vqgan_to_imnet(image))
    # 计算损失
    if target_msg is not None:
        loss = loss_w(output, target_msg)
        # 计算梯度
        msg_decoder.zero_grad()
        inr_net.zero_grad()
        loss.backward()
        for param in inr_net.parameters():
            data_grad = param.grad.data
            sign_data_grad = data_grad.sign()
            param.data.add_(-1.0 * epsilon * sign_data_grad)
    else:
        loss = loss_w(output, msg)
        # 计算梯度
        msg_decoder.zero_grad()
        inr_net.zero_grad()
        loss.backward()
        for param in inr_net.parameters():
            data_grad = param.grad.data
            sign_data_grad = data_grad.sign()
            param.data.add_(epsilon * sign_data_grad)

    # perturbed_image = inference_inr(inr_net, embed_fn, args)
    # perturbed_image = perturbed_image.permute(2, 0, 1).unsqueeze(0)
    # # 前向传播，使用扰动后的图像计算输出
    # output = model(norm(perturbed_image))
    # _, predicted = output.max(1)
    # 返回扰动后的图像和对应的输出
    return inr_net

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2]
                   * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def CW_msg_loss(decoded, keys): # b k
    ind = keys
    # x_sorted, ind_sorted = x.sort(dim=1)
    # ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(decoded[np.arange(decoded.shape[0]), :] * ind - decoded[np.arange(decoded.shape[0]), :] * (1. - ind))
    return loss_value.mean()

def attack_inr_PGD_linf(msg_decoder, loss_w, loss_i, inr, embed_fn, 
                        msg, target_msg, vqgan_to_imnet, epsilon, 
                        alpha, num_iter,
                        args):
    inr_net = copy.deepcopy(inr)
    msg_decoder.eval()
    for param in inr_net.parameters():
        param.requires_grad = True
    if target_msg is not None:
        target_msg = target_msg.clone().detach()
    else:
        msg = msg.clone().detach()
    
    inr_net_original = copy.deepcopy(inr)
    for param in inr_net_original.parameters():
        param.requires_grad = False
    # params_old = [p.clone().detach() for p in inr_net.parameters()]
    # Starting at a uniformly random point
    for param in inr_net.parameters():
        param.data.add_(torch.empty_like(param.data).uniform_(-epsilon, epsilon))

    # image = image.permute(2, 0, 1).unsqueeze(0)
    # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    if target_msg is not None:
        for _ in range(num_iter):
            image = inference_inr(inr_net, embed_fn, args)
            image = image.permute(2, 0, 1).unsqueeze(0)
            # adv_images.requires_grad = True
            output = msg_decoder(vqgan_to_imnet(image))

            # Calculate loss
            loss = loss_w(output, target_msg)
            msg_decoder.zero_grad()
            inr_net.zero_grad()
            # Update adversarial images
            loss.backward()
            for param, param_original in zip(inr_net.parameters(), inr_net_original.parameters()):
                data_grad = param.grad.data
                sign_data_grad = data_grad.sign()
                param_add = param.data - alpha * sign_data_grad # with target
                delta = torch.clamp(param_add - param_original.data,
                                min=-epsilon, max=epsilon)
                param.data = param_original.data.add(delta)
    else:
        for _ in range(num_iter):
            image = inference_inr(inr_net, embed_fn, args)
            image = image.permute(2, 0, 1).unsqueeze(0)
            # adv_images.requires_grad = True
            output = msg_decoder(vqgan_to_imnet(image))

            # Calculate loss
            loss = loss_w(output, msg)
            msg_decoder.zero_grad()
            inr_net.zero_grad()
            # Update adversarial images
            loss.backward()
            for param, param_original in zip(inr_net.parameters(), inr_net_original.parameters()):
                data_grad = param.grad.data
                sign_data_grad = data_grad.sign()
                param_add = param.data + alpha * sign_data_grad
                delta = torch.clamp(param_add - param_original.data,
                                min=-epsilon, max=epsilon)
                param.data = param_original.data.add(delta)
    return inr_net

def grad_clip_not_targeted(grad, image_original, image_last, image_epsilon, image_alpha):
    diff_image = image_last + image_alpha*grad - image_original
#     print(torch.mean(torch.abs(new_grad)))
    diff_image = torch.clamp(diff_image, -image_epsilon, image_epsilon)
    out_grad = (diff_image + image_original - image_last)/image_alpha
    return out_grad


def grad_clip_targeted(grad, image_original, image_last, image_epsilon, image_alpha):
    diff_image = image_last - image_alpha*grad - image_original
#     print(torch.mean(torch.abs(new_grad)))
    diff_image = torch.clamp(diff_image, -image_epsilon, image_epsilon)
    out_grad = -(diff_image + image_original - image_last)/image_alpha
    return out_grad

def attack_inr_DSPGD_linf(msg_decoder, loss_w, loss_i, inr, embed_fn,
                                                                        msg, target_msg, vqgan_to_imnet, net_epsilon,
                                                                        net_alpha, num_iter, image_epsilon,
                                                                        image_alpha, args):
    image_original = inference_inr(inr, embed_fn, args)
    image_original = image_original.permute(2, 0, 1).unsqueeze(0).detach()
    
    inr_net = copy.deepcopy(inr)
    msg_decoder.eval()
    for param in inr_net.parameters():
        param.requires_grad = True
    if target_msg is not None:
        target_msg = target_msg.clone().detach()
    else:
        msg = msg.clone().detach()
    inr_net_original = copy.deepcopy(inr)
    for param in inr_net_original.parameters():
        param.requires_grad = False
    # params_old = [p.clone().detach() for p in inr_net.parameters()]
    # Starting at a uniformly random point
    for param in inr_net.parameters():
        param.data.add_(torch.empty_like(param.data).uniform_(-net_epsilon, net_epsilon))
        
    # image = image.permute(2, 0, 1).unsqueeze(0)
    # adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    if target_msg is not None:
        for _ in range(num_iter):
            image = inference_inr(inr_net, embed_fn, args)
            image = image.permute(2, 0, 1).unsqueeze(0)
            image.register_hook(lambda grad: grad_clip_targeted(grad, image_original=image_original, image_last = image.detach(), image_epsilon=image_epsilon, image_alpha=image_alpha))
            # adv_images.requires_grad = True
            output = msg_decoder(vqgan_to_imnet(image))

            # Calculate loss
            
            loss = loss_w(output, target_msg)
            msg_decoder.zero_grad()
            inr_net.zero_grad()
            # Update adversarial images
            loss.backward()
            for param, param_original in zip(inr_net.parameters(), inr_net_original.parameters()):
                data_grad = param.grad.data
                sign_data_grad = data_grad.sign()
                param_add = param.data - net_alpha * sign_data_grad # with target
                delta = torch.clamp(param_add - param_original.data,
                                min=-net_epsilon, max=net_epsilon)
                param.data = param_original.data.add(delta)
    else:
        for _ in range(num_iter):
            image = inference_inr(inr_net, embed_fn, args)
            image = image.permute(2, 0, 1).unsqueeze(0)
            image.register_hook(lambda grad: grad_clip_not_targeted(grad, image_original=image_original, image_last = image.detach(), image_epsilon=image_epsilon, image_alpha=image_alpha))
            # adv_images.requires_grad = True
            output = msg_decoder(vqgan_to_imnet(image))

            # Calculate loss
            
            loss = loss_w(output, msg)
            msg_decoder.zero_grad()
            inr_net.zero_grad()
            # Update adversarial images
            loss.backward()
            for param, param_original in zip(inr_net.parameters(), inr_net_original.parameters()):
                data_grad = param.grad.data
                sign_data_grad = data_grad.sign()
                param_add = param.data + net_alpha * sign_data_grad # no target
                delta = torch.clamp(param_add - param_original.data,
                                min=-net_epsilon, max=net_epsilon)
                param.data = param_original.data.add(delta)
    return inr_net

def attack_inr_PGD_linf_image_constrain(msg_decoder, loss_w, loss_i, inr, embed_fn,
                                                                        msg, target_msg, vqgan_to_imnet, net_epsilon,
                                                                        net_alpha, num_iter, image_epsilon,
                                                                        image_alpha, args):
    inr_net = copy.deepcopy(inr)
    msg_decoder.eval()
    
    for param in inr_net.parameters():
        param.requires_grad = True
    if target_msg is not None:
        target_msg = target_msg.clone().detach()
    else:
        msg = msg.clone().detach()
    
    inr_net_original = copy.deepcopy(inr)
    for param in inr_net_original.parameters():
        param.requires_grad = False
    
    image_original = inference_inr(inr_net_original, embed_fn, args)
    image_original = image_original.permute(2, 0, 1).unsqueeze(0)
    
    if target_msg is not None:
        for _ in range(num_iter):
            image = inference_inr(inr_net, embed_fn, args)
            image = image.permute(2, 0, 1).unsqueeze(0)
            perturbed_image = image.clone().detach().requires_grad_(True)
            output = msg_decoder(vqgan_to_imnet(perturbed_image))
            lossw = loss_w(output, target_msg)
            msg_decoder.zero_grad()
            inr_net.zero_grad()
            lossw.backward()
            
            grad_sign = perturbed_image.grad.sign()
            perturbed_image = perturbed_image - image_alpha * grad_sign # target
            delta = torch.clamp(perturbed_image - image_original, min=-image_epsilon, max=image_epsilon)
            perturbed_image = torch.clamp(image_original + delta, min=0, max=1).detach().requires_grad_(True)

            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            lossi = loss_i(image, perturbed_image)
            inr_net.zero_grad()
            # Update adversarial images
            lossi.backward()
            for param, param_original in zip(inr_net.parameters(), inr_net_original.parameters()):
                data_grad = param.grad.data
                sign_data_grad = data_grad.sign()
                param_add = param.data - net_alpha * sign_data_grad
                delta = torch.clamp(param_add - param_original.data,
                                min=-net_epsilon, max=net_epsilon)
                param.data = param_original.data.add(delta)
            # 确保像素值在[0,1]范围内
    else:
        for _ in range(num_iter):
            image = inference_inr(inr_net, embed_fn, args)
            image = image.permute(2, 0, 1).unsqueeze(0)
            perturbed_image = image.clone().detach().requires_grad_(True)
            output = msg_decoder(vqgan_to_imnet(perturbed_image))
            lossw = loss_w(output, msg)
            msg_decoder.zero_grad()
            inr_net.zero_grad()
            lossw.backward()
            
            grad_sign = perturbed_image.grad.sign()
            perturbed_image = perturbed_image + image_alpha * grad_sign # no target
            delta = torch.clamp(perturbed_image - image_original, min=-image_epsilon, max=image_epsilon)
            perturbed_image = torch.clamp(image_original + delta, min=0, max=1).detach().requires_grad_(True)

            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            lossi = loss_i(image, perturbed_image)
            inr_net.zero_grad()
            # Update adversarial images
            lossi.backward()
            for param, param_original in zip(inr_net.parameters(), inr_net_original.parameters()):
                data_grad = param.grad.data
                sign_data_grad = data_grad.sign()
                param_add = param.data - net_alpha * sign_data_grad
                delta = torch.clamp(param_add - param_original.data,
                                min=-net_epsilon, max=net_epsilon)
                param.data = param_original.data.add(delta)
            # 确保像素值在[0,1]范围内
    return inr_net

            

def compute_lpips(img1, img2):
    # 计算LPIPS
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    img1 = img1.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    img2 = img2.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    lpips_val = lpips(img1, img2).item()

    return lpips_val

def compute_ssim(preds, target):
    # 计算LPIPS
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    preds = preds.permute(2, 0, 1).unsqueeze(0)
    target = target.permute(2, 0, 1).unsqueeze(0)
    ssim_val = ssim(preds, target).item()

    return ssim_val

def normal_train(input_img, msg, msg_decoder, loss_w, loss_i, data_aug, vqgan_to_imnet, inr_weights_dir, image_idx, args):
    H = args.H
    W = args.W
    N_iters = args.N_iters
    nerf, embed_fn, grad_vars, optimizer, scheduler = create_nerf(args)
    coords = torch.stack(torch.meshgrid(torch.linspace(1 / 2, H - 1 / 2, H) * 2 / H - 1,
                                        torch.linspace(1 / 2, W - 1 / 2, W) * 2 / W - 1), -1).to(device)  # (H, W, 2)
    input_img = input_img.permute(0, 2, 3, 1).squeeze(0).detach() # 1 c h w -> h w c
    
    start_time = time.time()
    for i in range(0, N_iters):
        rgb = run_network(coords, nerf, embed_fn)
        rgb_aug = data_aug(rgb.permute(2, 0, 1).unsqueeze(0))
        decoded = msg_decoder(vqgan_to_imnet(rgb_aug)) # b c h w -> b k
        lossw = loss_w(decoded, msg) # b k -> 1
        lossi = loss_i(rgb, input_img)
        loss = args.lambda_w*lossw + args.lambda_i*lossi
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f'i: {i}, Loss W: {lossw.item()}, Loss I: {lossi.item()}, Total Loss: {loss.item()}')
        # tqdm.write(f"[TRAIN] Iter: {i} Loss: {img_loss.item()}  PSNR: {psnr.item()}")
    end_time = time.time()
    execution_time = end_time - start_time
    print("normal_time: ", execution_time, 's')

    if args.save_inr:
        rgb = run_network(coords, nerf, embed_fn)
        rgb = torch.clamp(rgb, 0, 1)
        rgb_target = torch.clamp(input_img, 0, 1)
        os.makedirs(os.path.join(inr_weights_dir, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(inr_weights_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(inr_weights_dir, 'images_target'), exist_ok=True)
        save_path = os.path.join(inr_weights_dir, 'weights', '{:06d}.pth'.format(image_idx))
        torch.save({
            'net': nerf.state_dict(),
        }, save_path)
        # rgb8 = to8b(rgb.detach().cpu().numpy())
        filename = os.path.join(inr_weights_dir, 'images', '{:06d}.png'.format(image_idx))
        # imageio.imwrite(filename, rgb8)
        imsave(filename, to8b(rgb.detach().cpu().numpy()))
        # rgb8 = to8b(rgb.detach().cpu().numpy())
        filename_target = os.path.join(inr_weights_dir, 'images_target', '{:06d}.png'.format(image_idx))
        # imageio.imwrite(filename, rgb8)
        imsave(filename_target, to8b(rgb_target.detach().cpu().numpy()))
    return nerf, embed_fn


    # img_loss = img2mse(rgb, input_img)
    # psnr = mse2psnr(img_loss)
    # lpips_val = compute_lpips(rgb, input_img)
    # ssim_val = compute_ssim(rgb, input_img)
    # mse_val = img_loss.item()
    # psnr_val = psnr.item()

    # with open(os.path.join(inr_weights_dir, 'LPIPS.txt'), 'a') as f:
    #     f.write(str(lpips_val) + '\n')

    # with open(os.path.join(inr_weights_dir, 'SSIM.txt'), 'a') as f:
    #     f.write(str(ssim_val) + '\n')

    # with open(os.path.join(inr_weights_dir, 'MSE.txt'), 'a') as f:
    #     f.write(str(mse_val) + '\n')

    # with open(os.path.join(inr_weights_dir, 'PSNR.txt'), 'a') as f:
    #     f.write(str(psnr_val) + '\n')

    # # 读取LPIPS和SSIM的所有值
    # with open(os.path.join(inr_weights_dir, 'LPIPS.txt'), 'r') as f:
    #     lpips_values = [float(line.strip()) for line in f]

    # with open(os.path.join(inr_weights_dir, 'SSIM.txt'), 'r') as f:
    #     ssim_values = [float(line.strip()) for line in f]

    # with open(os.path.join(inr_weights_dir, 'MSE.txt'), 'r') as f:
    #     mse_values = [float(line.strip()) for line in f]

    # with open(os.path.join(inr_weights_dir, 'PSNR.txt'), 'r') as f:
    #     psnr_values = [float(line.strip()) for line in f]

    # # 计算平均值
    # lpips_mean = sum(lpips_values) / len(lpips_values)
    # ssim_mean = sum(ssim_values) / len(ssim_values)
    # mse_mean = sum(mse_values) / len(mse_values)
    # psnr_mean = sum(psnr_values) / len(psnr_values)

    # # 保存平均值
    # with open(os.path.join(inr_weights_dir, 'LPIPS_mean.txt'), 'w') as f:
    #     f.write(str(lpips_mean))

    # with open(os.path.join(inr_weights_dir, 'SSIM_mean.txt'), 'w') as f:
    #     f.write(str(ssim_mean))

    # with open(os.path.join(inr_weights_dir, 'MSE_mean.txt'), 'w') as f:
    #     f.write(str(mse_mean))

    # with open(os.path.join(inr_weights_dir, 'PSNR_mean.txt'), 'w') as f:
    #     f.write(str(psnr_mean))


def adversarial_train(input_img, msg, msg_decoder, loss_w, loss_i, data_aug, vqgan_to_imnet, train_net_PGD_epsilon, train_net_PGD_alpha,
                                                  train_net_PGD_num_iter, inr_weights_dir, image_idx, args):
    H = args.H
    W = args.W
    N_iters = args.N_iters

    # attack_noise_std = args.attack_noise_std

    nerf, embed_fn, grad_vars, optimizer, scheduler = create_nerf(args)
    coords = torch.stack(torch.meshgrid(torch.linspace(1 / 2, H - 1 / 2, H) * 2 / H - 1,
                                        torch.linspace(1 / 2, W - 1 / 2, W) * 2 / W - 1), -1).to(device)  # (H, W, 2)
    input_img = input_img.permute(0, 2, 3, 1).squeeze(0).detach()

#     sys.exit()
    start_time = time.time()
    for i in range(0, N_iters):
        nerf_attack = copy.deepcopy(nerf)
        # for param_attack, param_original in zip(nerf_attack.parameters(), nerf.parameters()):
        #     param_attack.data = param_original.data.clone()

        # params_dict = nerf.state_dict()
        # nerf_attack.load_state_dict(params_dict)

        # noise
        # for param in nerf_attack.parameters():
        #     noise = torch.randn_like(param.data) * attack_noise_std
        #     param.data.add_(noise)

        # attack
        nbit = msg.shape[-1]
        target_msg = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
        nerf_attack_tmp = attack_inr_PGD_linf(msg_decoder, loss_w, loss_i, nerf_attack, embed_fn,
                                                msg, target_msg, vqgan_to_imnet, train_net_PGD_epsilon,
                                                train_net_PGD_alpha, train_net_PGD_num_iter,
                                                args)
        for param_attack, param_attack_tmp in zip(nerf_attack.parameters(), nerf_attack_tmp.parameters()):
            param_attack.data.add_(param_attack_tmp.data.detach() - param_attack.data.detach().clone())

        rgb = run_network(coords, nerf, embed_fn)
        lossi = loss_i(rgb, input_img)

        rgb_attack = run_network(coords, nerf_attack, embed_fn)
        rgb_attack = rgb_attack.permute(2, 0, 1).unsqueeze(0)
        
        # rgb_aug = rgb_attack
        rgb_aug = data_aug(rgb_attack)
        
        decoded = msg_decoder(vqgan_to_imnet(rgb_aug)) # b c h w -> b k

        lossw = loss_w(decoded, msg)

        loss = args.lambda_w * lossw + args.lambda_i*lossi
        # psnr = mse2psnr(img_loss)
        optimizer.zero_grad()
        loss.backward()
        for param_attack, param_original in zip(nerf_attack.parameters(), nerf.parameters()):
            param_original.grad.add_(param_attack.grad.clone())
        
        optimizer.step()
        scheduler.step()
        print(f'i: {i}, Loss W: {lossw.item()}, Loss I: {lossi.item()}, Total Loss: {loss.item()}')
        # tqdm.write(f"[TRAIN] Iter: {i} Loss: {img_loss.item()}  PSNR: {psnr.item()}")
    end_time = time.time()
    execution_time = end_time - start_time
    print("adversarial_time: ", execution_time, 's')

    if args.save_inr:
        rgb = run_network(coords, nerf, embed_fn)
        rgb = torch.clamp(rgb, 0, 1)
        rgb_target = torch.clamp(input_img, 0, 1)
        os.makedirs(os.path.join(inr_weights_dir, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(inr_weights_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(inr_weights_dir, 'images_target'), exist_ok=True)
        save_path = os.path.join(inr_weights_dir, 'weights', '{:06d}.pth'.format(image_idx))
        torch.save({
            'net': nerf.state_dict(),
        }, save_path)
        # rgb8 = to8b(rgb.detach().cpu().numpy())
        filename = os.path.join(inr_weights_dir, 'images', '{:06d}.png'.format(image_idx))
        # imageio.imwrite(filename, rgb8)
        imsave(filename, to8b(rgb.detach().cpu().numpy()))
        # rgb8 = to8b(rgb.detach().cpu().numpy())
        filename_target = os.path.join(inr_weights_dir, 'images_target', '{:06d}.png'.format(image_idx))
        # imageio.imwrite(filename, rgb8)
        imsave(filename_target, to8b(rgb_target.detach().cpu().numpy()))
    return nerf, embed_fn

    # img_loss = img2mse(rgb, input_img)
    # psnr = mse2psnr(img_loss)
    # lpips_val = compute_lpips(rgb, input_img)
    # ssim_val = compute_ssim(rgb, input_img)
    # mse_val = img_loss.item()
    # psnr_val = psnr.item()

    # with open(os.path.join(inr_weights_dir, 'LPIPS.txt'), 'a') as f:
    #     f.write(str(lpips_val) + '\n')

    # with open(os.path.join(inr_weights_dir, 'SSIM.txt'), 'a') as f:
    #     f.write(str(ssim_val) + '\n')

    # with open(os.path.join(inr_weights_dir, 'MSE.txt'), 'a') as f:
    #     f.write(str(mse_val) + '\n')

    # with open(os.path.join(inr_weights_dir, 'PSNR.txt'), 'a') as f:
    #     f.write(str(psnr_val) + '\n')

    # # 读取LPIPS和SSIM的所有值
    # with open(os.path.join(inr_weights_dir, 'LPIPS.txt'), 'r') as f:
    #     lpips_values = [float(line.strip()) for line in f]

    # with open(os.path.join(inr_weights_dir, 'SSIM.txt'), 'r') as f:
    #     ssim_values = [float(line.strip()) for line in f]

    # with open(os.path.join(inr_weights_dir, 'MSE.txt'), 'r') as f:
    #     mse_values = [float(line.strip()) for line in f]

    # with open(os.path.join(inr_weights_dir, 'PSNR.txt'), 'r') as f:
    #     psnr_values = [float(line.strip()) for line in f]

    # # 计算平均值
    # lpips_mean = sum(lpips_values) / len(lpips_values)
    # ssim_mean = sum(ssim_values) / len(ssim_values)
    # mse_mean = sum(mse_values) / len(mse_values)
    # psnr_mean = sum(psnr_values) / len(psnr_values)

    # # 保存平均值
    # with open(os.path.join(inr_weights_dir, 'LPIPS_mean.txt'), 'w') as f:
    #     f.write(str(lpips_mean))

    # with open(os.path.join(inr_weights_dir, 'SSIM_mean.txt'), 'w') as f:
    #     f.write(str(ssim_mean))

    # with open(os.path.join(inr_weights_dir, 'MSE_mean.txt'), 'w') as f:
    #     f.write(str(mse_mean))

    # with open(os.path.join(inr_weights_dir, 'PSNR_mean.txt'), 'w') as f:
    #     f.write(str(psnr_mean))


# def train_inr_method(args):
#     if args.method == "normal":
#         return normal_train
#     elif args.method == "robust_inr":
#         return adversarial_train


def adversarial_train_image(input_img, msg, msg_decoder, loss_w, loss_i, data_aug, vqgan_to_imnet, train_net_PGD_epsilon, train_net_PGD_alpha,
                                                  train_net_PGD_num_iter, inr_weights_dir, image_idx, args):
    H = args.H
    W = args.W
    N_iters = args.N_iters

    # attack_noise_std = args.attack_noise_std

    nerf, grad_vars, optimizer, scheduler = create_direct_image(args)
    input_img = input_img.permute(0, 2, 3, 1).squeeze(0).detach()

#     sys.exit()
    start_time = time.time()
    for i in range(0, N_iters):
        # nerf_attack = nerf.clone().detach()
        # nerf_attack.requires_grad=True
        nerf_attack = copy.deepcopy(nerf)
        
        # for param_attack, param_original in zip(nerf_attack.parameters(), nerf.parameters()):
        #     param_attack.data = param_original.data.clone()

        # params_dict = nerf.state_dict()
        # nerf_attack.load_state_dict(params_dict)

        # noise
        # for param in nerf_attack.parameters():
        #     noise = torch.randn_like(param.data) * attack_noise_std
        #     param.data.add_(noise)

        # attack
        nbit = msg.shape[-1]
        target_msg = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
        
        nerf_attack_tmp = attack_image_PGD_linf(msg_decoder, loss_w, loss_i, nerf_attack,
                                                msg, target_msg, vqgan_to_imnet, train_net_PGD_epsilon,
                                                train_net_PGD_alpha, train_net_PGD_num_iter)
        
#         nerf_attack_tmp = attack_inr_PGD_linf(msg_decoder, loss_w, loss_i, nerf_attack, embed_fn,
#                                                 msg, target_msg, vqgan_to_imnet, train_net_PGD_epsilon,
#                                                 train_net_PGD_alpha, train_net_PGD_num_iter,
#                                                 args)
        
        nerf_attack = nerf_attack.add(nerf_attack_tmp.detach() - nerf_attack.detach().clone())

        rgb = nerf.permute(0, 2, 3, 1).squeeze(0)
        lossi = loss_i(rgb, input_img)

        rgb_attack = nerf_attack
        # rgb_attack = rgb_attack.permute(0, 2, 3, 1)
        
        # rgb_aug = rgb_attack
        rgb_aug = data_aug(rgb_attack)
        
        decoded = msg_decoder(vqgan_to_imnet(rgb_aug)) # b c h w -> b k

        lossw = loss_w(decoded, msg)

        loss = args.lambda_w * lossw + args.lambda_i*lossi
        # psnr = mse2psnr(img_loss)
        optimizer.zero_grad()
        nerf_attack.retain_grad()
        loss.backward()
        nerf.grad.add_(nerf_attack.grad.clone())
        
        optimizer.step()
        scheduler.step()
        print(f'i: {i}, Loss W: {lossw.item()}, Loss I: {lossi.item()}, Total Loss: {loss.item()}')
        # tqdm.write(f"[TRAIN] Iter: {i} Loss: {img_loss.item()}  PSNR: {psnr.item()}")
    end_time = time.time()
    execution_time = end_time - start_time
    print("adversarial_time: ", execution_time, 's')

    if args.save_inr:
        rgb = nerf.permute(0, 2, 3, 1).squeeze(0)
        rgb = torch.clamp(rgb, 0, 1)
        rgb_target = torch.clamp(input_img, 0, 1)
#         os.makedirs(os.path.join(inr_weights_dir, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(inr_weights_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(inr_weights_dir, 'images_target'), exist_ok=True)
#         save_path = os.path.join(inr_weights_dir, 'weights', '{:06d}.pth'.format(image_idx))
#         torch.save({
#             'net': nerf.state_dict(),
#         }, save_path)
        # rgb8 = to8b(rgb.detach().cpu().numpy())
        filename = os.path.join(inr_weights_dir, 'images', '{:06d}.png'.format(image_idx))
        # imageio.imwrite(filename, rgb8)
        imsave(filename, to8b(rgb.detach().cpu().numpy()))
        # rgb8 = to8b(rgb.detach().cpu().numpy())
        filename_target = os.path.join(inr_weights_dir, 'images_target', '{:06d}.png'.format(image_idx))
        # imageio.imwrite(filename, rgb8)
        imsave(filename_target, to8b(rgb_target.detach().cpu().numpy()))
    return nerf

def inference_inr(inr_net, embed_fn, args):
    H = args.H
    W = args.W
    # inr_net.eval()
    # with torch.no_grad():
    coords = torch.stack(torch.meshgrid(torch.linspace(1 / 2, H - 1 / 2, H) * 2 / H - 1,
                                        torch.linspace(1 / 2, W - 1 / 2, W) * 2 / W - 1), -1).to(device)  # (H, W, 2)
    rgb = run_network(coords, inr_net, embed_fn)
    return rgb

def noise_inr(model, mean=0, std=0.01):
    model_noise = copy.deepcopy(model)

    for param in model_noise.parameters():
        noise = torch.randn_like(param.data) * std + mean
        param.data.add_(noise)
    return model_noise

class ImageClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
#         mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
#         std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
#         self.mean = list(mean)
#         self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, torch.Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
#         img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
#         format_string += f"\n    mean={self.mean}"
#         format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )

def train(args):
    args.attack_net_DSPGD_image_epsilon = np.arange(args.attack_net_DSPGD_image_epsilon_start,
                           args.attack_net_DSPGD_image_epsilon_end + args.attack_net_DSPGD_image_epsilon_interval,
                           args.attack_net_DSPGD_image_epsilon_interval)
# get testloader
    vqgan_transform = transforms.Compose([
        transforms.Resize(args.H),
        transforms.CenterCrop(args.H),
        transforms.ToTensor(),
        ])
    testloader = utils.get_dataloader(args.test_dir, vqgan_transform, 1, shuffle=False, num_workers=1, collate_fn=None)
    vqgan_to_imnet = transforms.Compose([utils_img.normalize_img])


# attack parameters
    attack_net_FGSM_epsilon = args.attack_net_FGSM_epsilon/255.0
    # attack_noise_std = args.attack_noise_std

    attack_net_PGD_epsilon = args.attack_net_PGD_epsilon/255.0
    attack_net_PGD_num_iter = args.attack_net_PGD_num_iter
    attack_net_PGD_alpha = 2.5 * attack_net_PGD_epsilon / attack_net_PGD_num_iter

    
    attack_net_CW_epsilon = args.attack_net_CW_epsilon/255.0
    attack_net_CW_num_iter = args.attack_net_CW_num_iter
    attack_net_CW_alpha = 2.5 * attack_net_CW_epsilon / attack_net_CW_num_iter

    train_net_PGD_epsilon = args.train_net_PGD_epsilon/255.0
    train_net_PGD_num_iter = args.train_net_PGD_num_iter
    train_net_PGD_alpha = 2.5 * train_net_PGD_epsilon / train_net_PGD_num_iter





# image-level attack mode
    if args.attack_mode == 'all':
        attacks = {
            'none': lambda x: x,
            # 'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            'crop_25': lambda x: utils_img.center_crop(x, 0.25),
            # 'rot_25': lambda x: utils_img.rotate(x, 25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            # 'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
            'jpeg_50': lambda x: utils_img.REALJPEG(x, 50),
            'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
            # 'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            # 'contrast_1p5': lambda x: utils_img.adjust_contrast(x, 1.5),
            'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
            # 'saturation_1p5': lambda x: utils_img.adjust_saturation(x, 1.5),
            'saturation_2': lambda x: utils_img.adjust_saturation(x, 2),
            # 'sharpness_1p5': lambda x: utils_img.adjust_sharpness(x, 1.5),
            'sharpness_2': lambda x: utils_img.adjust_sharpness(x, 2),
            'resize_05': lambda x: utils_img.resize(x, 0.5),
            # 'resize_01': lambda x: utils_img.resize(x, 0.1),
            # 'overlay_text': lambda x: utils_img.overlay_text(x, [76,111,114,101,109,32,73,112,115,117,109]),
            # 'comb': lambda x: utils_img.REALJPEG(utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80),
        }
    elif args.attack_mode == 'few':
        attacks = {
            'none': lambda x: x,
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
            'jpeg_50': lambda x: utils_img.REALJPEG(x, 50),
            # 'comb': lambda x: utils_img.REALJPEG(utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80),
        }
    else:
        attacks = {'none': lambda x: x}
    
        # Construct data augmentation seen at train time
    if args.data_augmentation:
        attacks_aug = {
            'none': lambda x: x,
            'crop_25': lambda x: utils_img.center_crop(x, 0.25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'jpeg_50': lambda x: utils_img.JPEGSS(x, 50),
            'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
            'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
            # 'saturation_2': lambda x: utils_img.adjust_saturation(x, 2),
            # 'sharpness_2': lambda x: utils_img.adjust_sharpness(x, 2),
            'resize_05': lambda x: utils_img.resize(x, 0.5),
        }
        p = {
            'none': 2.0,
            'crop_25': 1.0,
            'rot_90': 1.0,
            'jpeg_50': 1.0,
            'brightness_1p5': 1.0,
            'contrast_2': 0.5,
            # 'saturation_2': lambda x: utils_img.adjust_saturation(x, 2),
            # 'sharpness_2': lambda x: utils_img.adjust_sharpness(x, 2),
            'resize_05': 1.0,
        }
        # attacks_aug['comb'] = lambda x: utils_img.JPEGSS(utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80)
        data_aug = utils_img.ImageAug(attacks_aug, p).to(device)
    else:
        data_aug = nn.Identity().to(device)

# choose the training type
    if args.method == "normal":
        method = "normal" + "_lambda_w_" + str(args.lambda_w) + "_lambda_i_" + str(args.lambda_i)
    elif args.method == "robust_inr":
        method = "robust_inr_epsilon_{:.3f}_alpha_{:.3f}_num_iter_{:.3f}_lambda_w_{:.3f}_lambda_i_{:.3f}".format(
            train_net_PGD_epsilon, train_net_PGD_alpha, train_net_PGD_num_iter, args.lambda_w, args.lambda_i)
    else:
        raise Exception("method wrong!")

# choose the attack types
    attack_net_str = ""
    if args.attack_net_FGSM:
        attack_net_str = attack_net_str + "_attack_net_FGSM"
    if args.attack_net_PGD:
        attack_net_str = attack_net_str + "_attack_net_PGD"
    if args.attack_net_CW:
        attack_net_str = attack_net_str + "_attack_net_CW"

# generate the experiment name  
    expname = "{}_{}_{}_{}".format(method,
                                   attack_net_str,
                                   str(args.N_iters),
                                   "seed_" + str(args.seed))
    i = 0
    base_log_dir = args.base_log_dir
    exp_dir = os.path.join(base_log_dir, expname)
    while True:
        try:
            if i == 0:
                # 第一次尝试创建文件夹时，直接使用指定的文件夹名
                os.makedirs(os.path.join(base_log_dir, expname), exist_ok=False)
            else:
                # 如果文件夹已经存在，则在文件夹名后面加上数字后缀
                os.makedirs(f"{os.path.join(base_log_dir, expname)}_{i}", exist_ok=False)
                exp_dir = f"{os.path.join(base_log_dir, expname)}_{i}"
            break
        except FileExistsError:
            i += 1

    f = os.path.join(exp_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

# save INR to exp_dir/INR
    inr_weights_dir = os.path.join(exp_dir, 'INR')
    os.makedirs(inr_weights_dir)

# save tensorboard log to exp_dir/tensorboard
    summary_writer = SummaryWriter(os.path.join(exp_dir, 'tensorboard'))

# load the msg decoder
    msg_decoder = torch.jit.load(args.msg_decoder_path).to(device)
    msg_decoder.eval()
    nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]
    
    for param in [*msg_decoder.parameters()]:
        param.requires_grad = False

# Create losses
    print(f'>>> Creating losses...')
    print(f'Losses: {args.loss_w} and {args.loss_i}...')
    if args.loss_w == 'mse':        
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2) # b k - b k
    elif args.loss_w == 'bce':
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')
    else:
        raise NotImplementedError
    
    loss_i = build_image_loss(args.loss_i)

# initial the results
    # total_clean = 0
    # acc_clean_all = 0

    total_attack_net_FGSM = 0
    acc_attack_net_FGSM_all = 0

    total_attack_net_PGD = 0
    acc_attack_net_PGD_all = 0
    
    total_attack_net_DSPGD = 0
    acc_attack_net_DSPGD_all = 0

    total_attack_net_CW = 0
    acc_attack_net_CW_all = 0
    
    # lpips_val_clean_all = 0
    # ssim_val_clean_all = 0
    # mse_val_clean_all = 0
    # psnr_val_clean_all = 0

    lpips_val_attack_net_FGSM_all = 0
    ssim_val_attack_net_FGSM_all = 0
    mse_val_attack_net_FGSM_all = 0
    psnr_val_attack_net_FGSM_all = 0

    lpips_val_attack_net_PGD_all = 0
    ssim_val_attack_net_PGD_all = 0
    mse_val_attack_net_PGD_all = 0
    psnr_val_attack_net_PGD_all = 0

    lpips_val_attack_net_CW_all = 0
    ssim_val_attack_net_CW_all = 0
    mse_val_attack_net_CW_all = 0
    psnr_val_attack_net_CW_all = 0 

    lpips_val_attack_net_DSPGD_all = 0
    ssim_val_attack_net_DSPGD_all = 0
    mse_val_attack_net_DSPGD_all = 0
    psnr_val_attack_net_DSPGD_all = 0


    start_img = args.start_img
    end_img = args.end_img

    log_stats = {}
# start training
    for image_idx, inputs in tqdm(enumerate(testloader), total=len(testloader)):
# if run part of the whole test dataset
        if image_idx < start_img:
            continue
        if image_idx > end_img:
            break
        inputs = inputs.to(device) # 1 c h w

# generate rand message
        print(f'\n>>> Creating key with {nbit} bits...')
        msg = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
        msg_str = "".join([ str(int(ii)) for ii in msg.tolist()[0]])
        # print(f'Key: {key_str}')
        
        summary_writer.add_text('target/message', msg_str, global_step=image_idx)
        summary_writer.add_image('target/image',
                                 to8b(inputs.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()),
                                 global_step=image_idx, dataformats='HWC')

        input_img = inputs

        # input_img = un_norm(input_img)
        if args.method == "normal":
            inr_net, embed_fn = normal_train(input_img, msg, msg_decoder, loss_w, loss_i, data_aug, vqgan_to_imnet, inr_weights_dir, image_idx, args)
        elif args.method == "robust_inr":
            inr_net = adversarial_train_image(input_img, msg, msg_decoder, loss_w, loss_i, data_aug, vqgan_to_imnet, train_net_PGD_epsilon, train_net_PGD_alpha,
                                                  train_net_PGD_num_iter, inr_weights_dir, image_idx, args)

        # target_msg = None
        if args.not_targeted:
            target_msg = None
        else:
            target_msg = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
        
        if args.attack_clean:
            # os.makedirs(os.path.join(exp_dir, "attack_clean"), exist_ok=True)
            img_clean = inr_net
            for name, attack in attacks.items():
                imgs_aug = attack(img_clean)
                decoded = msg_decoder(vqgan_to_imnet(imgs_aug)) # b c h w -> b k
                diff = (~torch.logical_xor(decoded>0, msg>0)) # b k -> b k
                diff_str = "".join([ str(int(ii)) for ii in diff.tolist()[0]])
                bit_acc = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
                log_stats.setdefault(f'bit_acc_{name}', []).append(bit_acc.item())
                log_stats[f'average_acc_{name}'] = np.mean(log_stats[f'bit_acc_{name}'])

                summary_writer.add_image(f'{name}/image', to8b(imgs_aug.detach().cpu().squeeze(0).numpy()), global_step=image_idx, dataformats='CHW')
                summary_writer.add_text(f'{name}/bit_diff', diff_str, global_step=image_idx)
                summary_writer.add_scalar(f'{name}/bit_acc', bit_acc.item(), global_step=image_idx)
                summary_writer.add_scalar(f'{name}/average_acc', log_stats[f'average_acc_{name}'], global_step=image_idx)

                img = torch.clamp(img_clean.permute(0, 2, 3, 1).squeeze(0), 0, 1)
                ref = inputs.permute(0, 2, 3, 1).squeeze(0)
                img_loss = img2mse(img, ref)
                psnr = mse2psnr(img_loss)
                lpips_val = compute_lpips(img, ref)
                ssim_val = compute_ssim(img, ref)
                mse_val = img_loss.item()
                psnr_val = psnr.item()
                
                log_stats.setdefault(f'lpips_val_{name}', []).append(lpips_val)
                log_stats[f'lpips_val_mean_{name}'] = np.mean(log_stats[f'lpips_val_{name}'])
                log_stats.setdefault(f'ssim_val_{name}', []).append(ssim_val)
                log_stats[f'ssim_val_mean_{name}'] = np.mean(log_stats[f'ssim_val_{name}'])
                log_stats.setdefault(f'mse_val_{name}', []).append(mse_val)
                log_stats[f'mse_val_mean_{name}'] = np.mean(log_stats[f'mse_val_{name}'])
                log_stats.setdefault(f'psnr_val_{name}', []).append(psnr_val)
                log_stats[f'psnr_val_mean_{name}'] = np.mean(log_stats[f'psnr_val_{name}'])     
                
                summary_writer.add_scalar(f'{name}/lpips_val', lpips_val, global_step=image_idx)
                summary_writer.add_scalar(f'{name}/ssim_val', ssim_val, global_step=image_idx)
                summary_writer.add_scalar(f'{name}/mse_val', mse_val, global_step=image_idx)
                summary_writer.add_scalar(f'{name}/psnr_val', psnr_val, global_step=image_idx)
                summary_writer.add_scalar(f'{name}/lpips_val_mean', log_stats[f'lpips_val_mean_{name}'], global_step=image_idx)
                summary_writer.add_scalar(f'{name}/ssim_val_mean', log_stats[f'ssim_val_mean_{name}'], global_step=image_idx)
                summary_writer.add_scalar(f'{name}/mse_val_mean', log_stats[f'mse_val_mean_{name}'], global_step=image_idx)
                summary_writer.add_scalar(f'{name}/psnr_val_mean', log_stats[f'psnr_val_mean_{name}'], global_step=image_idx)

        
        if args.attack_net_FGSM:
            name = 'attack_net_FGSM'
            # os.makedirs(os.path.join(exp_dir, "FGSM_epsilon_{:.3f}".format(attack_net_FGSM_epsilon)), exist_ok=True)
            inr_net_attack = attack_image_FGSM(msg_decoder, loss_w, loss_i, inr_net,
                                                                        msg, target_msg, vqgan_to_imnet, attack_net_FGSM_epsilon, args)
            img_attack = inr_net_attack
            decoded = msg_decoder(vqgan_to_imnet(img_attack))
            # loss = criterion(outputs, targets)
            # test_loss += loss.item()
            diff = (~torch.logical_xor(decoded>0, msg>0))
            diff_str = "".join([ str(int(ii)) for ii in diff.tolist()[0]])
            bit_acc = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            log_stats.setdefault(f'bit_acc_{name}', []).append(bit_acc.item())
            log_stats[f'average_acc_{name}'] = np.mean(log_stats[f'bit_acc_{name}'])

            summary_writer.add_image(f'{name}/image', to8b(imgs_aug.detach().cpu().squeeze(0).numpy()), global_step=image_idx, dataformats='CHW')
            summary_writer.add_text(f'{name}/bit_diff', diff_str, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/bit_acc', bit_acc.item(), global_step=image_idx)
            summary_writer.add_scalar(f'{name}/average_acc', log_stats[f'average_acc_{name}'], global_step=image_idx)

            img = torch.clamp(img_attack.permute(0, 2, 3, 1).squeeze(0), 0, 1)
            ref = inputs.permute(0, 2, 3, 1).squeeze(0)
            img_loss = img2mse(img, ref)
            psnr = mse2psnr(img_loss)
            lpips_val = compute_lpips(img, ref)
            ssim_val = compute_ssim(img, ref)
            mse_val = img_loss.item()
            psnr_val = psnr.item()
            
            
            log_stats.setdefault(f'lpips_val_{name}', []).append(lpips_val)
            log_stats[f'lpips_val_mean_{name}'] = np.mean(log_stats[f'lpips_val_{name}'])
            log_stats.setdefault(f'ssim_val_{name}', []).append(ssim_val)
            log_stats[f'ssim_val_mean_{name}'] = np.mean(log_stats[f'ssim_val_{name}'])
            log_stats.setdefault(f'mse_val_{name}', []).append(mse_val)
            log_stats[f'mse_val_mean_{name}'] = np.mean(log_stats[f'mse_val_{name}'])
            log_stats.setdefault(f'psnr_val_{name}', []).append(psnr_val)
            log_stats[f'psnr_val_mean_{name}'] = np.mean(log_stats[f'psnr_val_{name}'])     
            
            summary_writer.add_scalar(f'{name}/lpips_val', lpips_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/ssim_val', ssim_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/mse_val', mse_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/psnr_val', psnr_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/lpips_val_mean', log_stats[f'lpips_val_mean_{name}'], global_step=image_idx)
            summary_writer.add_scalar(f'{name}/ssim_val_mean', log_stats[f'ssim_val_mean_{name}'], global_step=image_idx)
            summary_writer.add_scalar(f'{name}/mse_val_mean', log_stats[f'mse_val_mean_{name}'], global_step=image_idx)
            summary_writer.add_scalar(f'{name}/psnr_val_mean', log_stats[f'psnr_val_mean_{name}'], global_step=image_idx)


        if args.attack_net_PGD:
            name = 'attack_net_PGD'
            inr_net_attack = attack_image_PGD_linf(msg_decoder, loss_w, loss_i, inr_net,
                                                                        msg, target_msg, vqgan_to_imnet, attack_net_PGD_epsilon,
                                                                        attack_net_PGD_alpha, attack_net_PGD_num_iter,
                                                                        args)
            img_attack = inr_net_attack
            decoded = msg_decoder(vqgan_to_imnet(img_attack))
            # loss = criterion(outputs, targets)
            # test_loss += loss.item()
            diff = (~torch.logical_xor(decoded>0, msg>0))
            diff_str = "".join([ str(int(ii)) for ii in diff.tolist()[0]])
            bit_acc = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            log_stats.setdefault(f'bit_acc_{name}', []).append(bit_acc.item())
            log_stats[f'average_acc_{name}'] = np.mean(log_stats[f'bit_acc_{name}'])

            summary_writer.add_image(f'{name}/image', to8b(imgs_aug.detach().cpu().squeeze(0).numpy()), global_step=image_idx, dataformats='CHW')
            summary_writer.add_text(f'{name}/bit_diff', diff_str, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/bit_acc', bit_acc.item(), global_step=image_idx)
            summary_writer.add_scalar(f'{name}/average_acc', log_stats[f'average_acc_{name}'], global_step=image_idx)

            img = torch.clamp(img_attack.permute(0, 2, 3, 1).squeeze(0), 0, 1)
            ref = inputs.permute(0, 2, 3, 1).squeeze(0)
            img_loss = img2mse(img, ref)
            psnr = mse2psnr(img_loss)
            lpips_val = compute_lpips(img, ref)
            ssim_val = compute_ssim(img, ref)
            mse_val = img_loss.item()
            psnr_val = psnr.item()
            
            
            log_stats.setdefault(f'lpips_val_{name}', []).append(lpips_val)
            log_stats[f'lpips_val_mean_{name}'] = np.mean(log_stats[f'lpips_val_{name}'])
            log_stats.setdefault(f'ssim_val_{name}', []).append(ssim_val)
            log_stats[f'ssim_val_mean_{name}'] = np.mean(log_stats[f'ssim_val_{name}'])
            log_stats.setdefault(f'mse_val_{name}', []).append(mse_val)
            log_stats[f'mse_val_mean_{name}'] = np.mean(log_stats[f'mse_val_{name}'])
            log_stats.setdefault(f'psnr_val_{name}', []).append(psnr_val)
            log_stats[f'psnr_val_mean_{name}'] = np.mean(log_stats[f'psnr_val_{name}'])     
            
            summary_writer.add_scalar(f'{name}/lpips_val', lpips_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/ssim_val', ssim_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/mse_val', mse_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/psnr_val', psnr_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/lpips_val_mean', log_stats[f'lpips_val_mean_{name}'], global_step=image_idx)
            summary_writer.add_scalar(f'{name}/ssim_val_mean', log_stats[f'ssim_val_mean_{name}'], global_step=image_idx)
            summary_writer.add_scalar(f'{name}/mse_val_mean', log_stats[f'mse_val_mean_{name}'], global_step=image_idx)
            summary_writer.add_scalar(f'{name}/psnr_val_mean', log_stats[f'psnr_val_mean_{name}'], global_step=image_idx)


                
        if args.attack_net_CW:
            name = 'attack_net_CW'
            inr_net_attack = attack_image_PGD_linf(msg_decoder, CW_msg_loss, loss_i, inr_net,
                                                                        msg, target_msg, vqgan_to_imnet, attack_net_CW_epsilon,
                                                                        attack_net_CW_alpha, attack_net_CW_num_iter,
                                                                        args)
            img_attack = inr_net_attack
            decoded = msg_decoder(vqgan_to_imnet(img_attack))
            # loss = criterion(outputs, targets)
            # test_loss += loss.item()
            diff = (~torch.logical_xor(decoded>0, msg>0))
            diff_str = "".join([ str(int(ii)) for ii in diff.tolist()[0]])
            bit_acc = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            log_stats.setdefault(f'bit_acc_{name}', []).append(bit_acc.item())
            log_stats[f'average_acc_{name}'] = np.mean(log_stats[f'bit_acc_{name}'])

            summary_writer.add_image(f'{name}/image', to8b(imgs_aug.detach().cpu().squeeze(0).numpy()), global_step=image_idx, dataformats='CHW')
            summary_writer.add_text(f'{name}/bit_diff', diff_str, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/bit_acc', bit_acc.item(), global_step=image_idx)
            summary_writer.add_scalar(f'{name}/average_acc', log_stats[f'average_acc_{name}'], global_step=image_idx)

            img = torch.clamp(img_attack.permute(0, 2, 3, 1).squeeze(0), 0, 1)
            ref = inputs.permute(0, 2, 3, 1).squeeze(0)
            img_loss = img2mse(img, ref)
            psnr = mse2psnr(img_loss)
            lpips_val = compute_lpips(img, ref)
            ssim_val = compute_ssim(img, ref)
            mse_val = img_loss.item()
            psnr_val = psnr.item()
            
            
            log_stats.setdefault(f'lpips_val_{name}', []).append(lpips_val)
            log_stats[f'lpips_val_mean_{name}'] = np.mean(log_stats[f'lpips_val_{name}'])
            log_stats.setdefault(f'ssim_val_{name}', []).append(ssim_val)
            log_stats[f'ssim_val_mean_{name}'] = np.mean(log_stats[f'ssim_val_{name}'])
            log_stats.setdefault(f'mse_val_{name}', []).append(mse_val)
            log_stats[f'mse_val_mean_{name}'] = np.mean(log_stats[f'mse_val_{name}'])
            log_stats.setdefault(f'psnr_val_{name}', []).append(psnr_val)
            log_stats[f'psnr_val_mean_{name}'] = np.mean(log_stats[f'psnr_val_{name}'])     
            
            summary_writer.add_scalar(f'{name}/lpips_val', lpips_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/ssim_val', ssim_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/mse_val', mse_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/psnr_val', psnr_val, global_step=image_idx)
            summary_writer.add_scalar(f'{name}/lpips_val_mean', log_stats[f'lpips_val_mean_{name}'], global_step=image_idx)
            summary_writer.add_scalar(f'{name}/ssim_val_mean', log_stats[f'ssim_val_mean_{name}'], global_step=image_idx)
            summary_writer.add_scalar(f'{name}/mse_val_mean', log_stats[f'mse_val_mean_{name}'], global_step=image_idx)
            summary_writer.add_scalar(f'{name}/psnr_val_mean', log_stats[f'psnr_val_mean_{name}'], global_step=image_idx)
        
        results = {}
        for name, attack in attacks.items():
            result = {
                "accuracy": log_stats[f'average_acc_{name}'],
                "psnr": log_stats[f'psnr_val_mean_{name}'],
                "ssim": log_stats[f'ssim_val_mean_{name}'],
                "lpips": log_stats[f'lpips_val_mean_{name}']
            }
            results[f'{name}'] = result
        
        if args.attack_net_FGSM:
            name = 'attack_net_FGSM'
            result = {
                "net_epsilon": attack_net_FGSM_epsilon,
                "accuracy": log_stats[f'average_acc_{name}'],
                "psnr": log_stats[f'psnr_val_mean_{name}'],
                "ssim": log_stats[f'ssim_val_mean_{name}'],
                "lpips": log_stats[f'lpips_val_mean_{name}']
            }
            results["FGSM"] = result
        if args.attack_net_PGD:
            name = 'attack_net_PGD'
            result = {
                "net_epsilon": attack_net_PGD_epsilon,
                "net_alpha": attack_net_PGD_alpha,
                "num_iter": attack_net_PGD_num_iter,
                "accuracy": log_stats[f'average_acc_{name}'],
                "psnr": log_stats[f'psnr_val_mean_{name}'],
                "ssim": log_stats[f'ssim_val_mean_{name}'],
                "lpips": log_stats[f'lpips_val_mean_{name}']
            }
            results[f"{name}"] = result
        if args.attack_net_CW:
            name = 'attack_net_CW'
            result = {
                "net_epsilon": attack_net_CW_epsilon,
                "net_alpha": attack_net_CW_alpha,
                "num_iter": attack_net_CW_num_iter,
                "accuracy": log_stats[f'average_acc_{name}'],
                "psnr": log_stats[f'psnr_val_mean_{name}'],
                "ssim": log_stats[f'ssim_val_mean_{name}'],
                "lpips": log_stats[f'lpips_val_mean_{name}']
            }
            results[f"{name}"] = result

        with open(os.path.join(exp_dir, 'average_results.json'), 'w') as json_file:
            json.dump(results, json_file, indent=4)
            
        with open(os.path.join(exp_dir, 'log_stats.json'), 'w') as json_file:
            json.dump(log_stats, json_file, indent=4)

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled GPU is required for the public DART image release.")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args = config_parser()
    args.base_log_dir = resolve_repo_path(args.base_log_dir)
    args.msg_decoder_path = resolve_repo_path(args.msg_decoder_path)
    args.test_dir = resolve_repo_path(args.test_dir)
    setup_seed(args.seed)
    train(args)
