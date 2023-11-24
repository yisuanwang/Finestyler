from PIL import Image
import numpy as np

import torch
import torch.nn
import torch.optim as optim
from torchvision import transforms, models

import style.StyleNet as StyleNet
import style.utils as utils
import clip
import torch.nn.functional as F
from style.template import imagenet_templates

from PIL import Image 
import PIL 
from torchvision import utils as vutils
from torchvision.transforms.functional import adjust_contrast
import utils.config as config
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def img_denormalize(image,device):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = image*std +mean
    return image

def img_normalize(image,device):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

    
def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return loss_var_l2

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


def getTotalLoss(args, content_features, text_features,source_features,text_source,target,device,VGG,clip_model,st=None):
    cropper = transforms.Compose([
        transforms.RandomCrop(args.crop_size)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
        transforms.Resize(224)
    ])

    target_features = utils.get_features(img_normalize(target,device), VGG)
    content_loss = 0
    content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)
    
    loss_patch=0 
    img_proc =[]
    for n in range(args.num_crops):
        target_crop = cropper(target)
        target_crop = augment(target_crop)
        img_proc.append(target_crop)
    img_proc = torch.cat(img_proc,dim=0)
    img_aug = img_proc
    image_features = clip_model.encode_image(clip_normalize(img_aug,device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

    
    img_direction = (image_features-source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    text_direction = (text_features-text_source).repeat(image_features.size(0),1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
    loss_temp[loss_temp<args.thresh] =0
    loss_patch+=loss_temp.mean()
    
    glob_features = clip_model.encode_image(clip_normalize(target,device))
    glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
    glob_direction = (glob_features-source_features)
    glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
    loss_glob = (1- torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()
    
    reg_tv = args.lambda_tv*get_image_prior_losses(target)
    total_loss = args.lambda_patch*loss_patch + args.lambda_c * content_loss + reg_tv + args.lambda_dir*loss_glob

    return total_loss

import numpy as np
import cv2
import torch

def save_image(tensor, filename):
    if isinstance(tensor, torch.Tensor):
        image = tensor.to("cpu").clone().detach()
        image = image.numpy()
    elif isinstance(tensor, np.ndarray):
        image = tensor
    else:
        raise TypeError("Input must be a PyTorch tensor or a NumPy array")

    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)

    image = np.transpose(image, (1, 2, 0))
    image = (image * 255).astype(np.uint8)

    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))



def getStyleImg(config_path,content_image,source="a Photo",prompt="a Photo",seed=42,get_total_loss=getTotalLoss,mask=None,save_epoch=False,path=''):
    setup_seed(seed)
    args = config.load_cfg_from_cfg_file(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = content_image.to(device)
    VGG = models.vgg19(pretrained=True).features
    VGG.to(device)
    for parameter in VGG.parameters():
        parameter.requires_grad_(False)
    
    content_features = utils.get_features(img_normalize(content_image,device), VGG)
    target = content_image.clone().requires_grad_(True).to(device)
    
    style_net = StyleNet.UNet()
    style_net.to(device)

    optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    steps = args.max_step

    output_image = content_image
    m_cont = torch.mean(content_image,dim=(2,3),keepdim=False).squeeze(0)
    m_cont = [m_cont[0].item(),m_cont[1].item(),m_cont[2].item()]

    cropper = transforms.Compose([
        transforms.RandomCrop(args.crop_size)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
        transforms.Resize(224)
    ])

    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

    with torch.no_grad():
        template_text = compose_text_with_templates(prompt, imagenet_templates)
        tokens = clip.tokenize(template_text).to(device)
        text_features = clip_model.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        template_source = compose_text_with_templates(source, imagenet_templates)
        tokens_source = clip.tokenize(template_source).to(device)
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)
        source_features = clip_model.encode_image(clip_normalize(content_image,device))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
    
    for epoch in range(0, steps+1):
        scheduler.step()
        target = style_net(content_image,use_sigmoid=True).to(device)
        target.requires_grad_(True)
        ###############
        total_loss, detail_loss = get_total_loss(args,content_features,text_features,source_features,text_source,target,device,VGG,clip_model,content_image,mask)
        ###############
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch %1 ==0:
            print('======={}/{}========'.format(epoch, steps+1))
            print('Total loss: ', total_loss.item())
            for k,v in detail_loss.items():
                print(f"{k}:{v}")
        if save_epoch :
            output_image = target.clone()
            output_image = torch.clamp(output_image,0,1)
            output_image = adjust_contrast(output_image,1.5)
            # output_image = utils.im_convert2(output_image)
            save_image(output_image, f'{path}epoch={epoch}_loss={total_loss.item()}.png')
            
    output_image = target.clone()
    output_image = torch.clamp(output_image,0,1)
    output_image = adjust_contrast(output_image,1.5)
    # output_image = utils.im_convert2(output_image)
    return output_image
