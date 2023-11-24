from predict import getMask
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import style.utils as utils
from style.CLIPstyler import getStyleImg
from torchvision import transforms, models
import torch.nn.functional as F
import clip
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

prompt_list = [
    {'style':"cloud", 'seed':0},
    {'style':"white wool", 'seed':2},
    {'style':"a sketch with crayon", 'seed':3},
    {'style':"oil painting of flowers", 'seed':7},
    {'style':'pop art of night city', 'seed':6},
    {'style':"Starry Night by Vincent van gogh", 'seed':0},
    {'style':"neon light", 'seed':5},
    {'style':"mosaic", 'seed':4},
    {'style':"green crystal", 'seed':1},
    {'style':"Underwater", 'seed':0},
    {'style':"fire", 'seed':0},
    {'style':'a graffiti style painting', 'seed':2},
    {'style':'The great wave off kanagawa by Hokusai', 'seed':0},
    {'style':'Wheatfield by Vincent van gogh', 'seed':2},
    {'style':'a Photo of white cloud', 'seed':3},
    {'style':'golden', 'seed':2},
    {'style':'Van gogh', 'seed':0},
    {'style':'pop art', 'seed':2},
    {'style':'a monet style underwater', 'seed':3},
    {'style':'A fauvism style painting', 'seed':2}
]

input_data  = [
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/ship.jpg",
        'cris_prompt' : "A white sailboat with three blue sails floating on the sea",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/911.jpg",
        'cris_prompt' : "a plane",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/1.jpg",
        'cris_prompt' : "a flower",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/house.jpg",
        'cris_prompt' : "a house",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/people.jpg",
        'cris_prompt' : "the face",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/Napoleon.jpg",
        'cris_prompt' : "a White War Horse",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/apple.png",
        'cris_prompt' : "a red apple",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/bigship.png",
        'cris_prompt' : "White Large Luxury Cruise Ship",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/car.png",
        'cris_prompt' : "White sports car.",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/lena.png",
        'cris_prompt' : "A woman's face.",
        'style_prompts' : prompt_list
    },

    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/mountain.png",
        'cris_prompt' : "mountain peak",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/tjl.jpeg",
        'cris_prompt' : "The White House at the Taj Mahal",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/man.jpg",
        'cris_prompt' : "The Men's face",
        'style_prompts' : prompt_list
    },

]

config_path = "./config/refcoco+/test.yaml"
model_pth = "./best_model.pth"

def getMaskImg(img,config_path,model_pth,sent=None,isMask=False,):
    if not isMask:
        img_style1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_style2 = img_style1/255.0
        img_style3 = np.transpose(img_style2, (2,0,1))
        img_style4 = torch.Tensor(img_style3)
        img_style = torch.unsqueeze(img_style4, 0)
    
        mask0 = getMask(img,sent,config_path,model_pth)
        mask1 = np.stack((mask0, mask0,mask0), axis=2)
        mask_img = np.array(mask1*255, dtype=np.uint8)
        return mask_img
    else:
        return img 

def getCVImg2Torch(img):
    img_style1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_style1 = img
    img_style2 = img_style1/255.0
    img_style3 = np.transpose(img_style2, (2,0,1))
    img_style4 = torch.Tensor(img_style3)
    img_style = torch.unsqueeze(img_style4, 0)
    return img_style

def load_image(img,mode="PLT"):
    if mode == "CV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])   
    image = transform(img)[:3, :, :].unsqueeze(0)
    return image.to(device)

def squeeze_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)
    image = torch.Tensor(image)
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

def clip_normalize2(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    return image

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return loss_var_l2

def getClipFeature(image,clip_model):
    image = F.interpolate(image,size=224,mode='bicubic')
    image = clip_model.encode_image(image.to(device))
    image = image.mean(axis=0, keepdim=True)
    image /= image.norm(dim=-1, keepdim=True)
    return image

def getVggFeature(image,device,VGG):
    return utils.get_features(img_normalize(image,device), VGG)

def getLoss(text_feature,img_feature):
    return 1-torch.cosine_similarity(text_feature, img_feature)

# 原始的
def getCropImgAndFeature(img,mask,target,clip_model,size=128,batch=64,pot_part=0.9,sizePose=None):
    print('getCropImgAndFeature begin')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img, mask, target = img.to(device), mask.to(device), target.to(device)
    
    back_crop, pot_crop ,pot_aug,extra_pot = [], [],[],[]
    cropper = transforms.RandomCrop(size)
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
        transforms.Resize(400)
    ])

    max_iterations = 2000  
    iteration = 0

    while len(pot_crop)<batch and iteration < max_iterations :
        iteration += 1
        # print(f'iteration={iteration}, len(pot_crop)={len(pot_crop)}')
        # print('in while 1: while len(pot_crop)<batch ')
        if sizePose:
            (i, j, h, w) = sizePose
        else:
            (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
        # print(f'(i, j, h, w)={(i, j, h, w)}')
        mask_crop = transforms.functional.crop(mask, i, j, h, w)
        img_crop = transforms.functional.crop(img, i, j, h, w)
        target_crop = transforms.functional.crop(target, i, j, h, w)
        
        if int(mask_crop[0].sum())/(3*size*size) >= 0.8:
            if len(pot_crop)<batch :
                # print('in while 1: pot_crop.append(img_crop)')
                pot_crop.append(img_crop)
                pot_aug.append(augment(target_crop))
    
    pot_allCrop = torch.cat(pot_crop,dim=0).to(device)
    pot_all_crop = pot_allCrop
    pot_crop_feature = clip_model.encode_image(clip_normalize2(pot_all_crop, device))

    while len(back_crop) < batch and iteration < max_iterations:
        iteration += 1
        # print(f'iteration={iteration}, len(back_crop)={len(back_crop)}')
        # print('in while 2: len(back_crop) < batch')
        (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
        # print(f'(i, j, h, w)={(i, j, h, w)}')
        mask_crop = transforms.functional.crop(mask, i, j, h, w)
        img_crop = transforms.functional.crop(img, i, j, h, w)
        target_crop = transforms.functional.crop(target, i, j, h, w)
        if int(mask_crop[0].sum())/(3*size*size) < 0.1:
            img_crop_feature = clip_model.encode_image(clip_normalize2(img_crop,device))
            cos = (1- torch.cosine_similarity(img_crop_feature, pot_crop_feature))
            if torch.numel(cos[cos>0.12]) > 0.8*batch:
                # print('in while 2: back_crop.append')
                back_crop.append([target_crop,img_crop])
        
    while len(extra_pot) < 0.1*batch and iteration < max_iterations:
        iteration += 1
        # print(f'iteration={iteration}, len(extra_pot)={len(extra_pot)}')
        # print('in while 3: len(extra_pot) < 0.1*batch')
        (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
        # print(f'(i, j, h, w)={(i, j, h, w)}')
        mask_crop = transforms.functional.crop(mask, i, j, h, w)
        img_crop = transforms.functional.crop(img, i, j, h, w)
        target_crop = transforms.functional.crop(target, i, j, h, w)
        if int(mask_crop[0].sum())/(3*size*size) < 0.1:
            img_crop_feature = clip_model.encode_image(clip_normalize2(img_crop,device))
            cos = (1- torch.cosine_similarity(img_crop_feature, pot_crop_feature))
            # if torch.numel(cos[cos<0.06]) > 0.2*batch:
            if torch.numel(cos[cos<0.10]) > 0.1*batch: # plane
                # print('in while 3: back_crop.append')
                extra_pot.append(augment(target_crop))
                pot_aug.append(augment(target_crop))

    print('getCropImgAndFeature end')
    return pot_aug, back_crop


import time

def getTotalLoss1(args, content_features,text_features,source_features,text_source,target,device,VGG,clip_model,img,mask):
    print('getTotalLoss1 begin')
    target_features = utils.get_features(img_normalize(target,device), VGG)
    content_loss = 0
    content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

    cropper = transforms.Compose([
        transforms.RandomCrop(args.crop_size)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
        transforms.Resize(224)
    ])

    loss_patch=0 

    img_crop, back_crop = getCropImgAndFeature(img, mask, target, clip_model, size=64, batch=64, pot_part=args.pot_part, sizePose=None)
    img_crop = torch.cat(img_crop,dim=0)
    img_aug = img_crop

    image_features = clip_model.encode_image(clip_normalize(img_aug,device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
    img_direction = (image_features-source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    text_direction = (text_features-text_source).repeat(image_features.size(0),1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
    loss_temp[loss_temp<args.thresh] =0
    loss_patch+=loss_temp.mean()
    
    # print('compute loss_back')
    loss_back = 0
    lossToBack = torch.nn.MSELoss()
    for i in back_crop:
        a = i[0]
        b = i[1]
        loss_back += lossToBack(a, b)

    # glob_features = clip_model.encode_image(clip_normalize(target,device))
    # glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
    # glob_direction = (glob_features-source_features)
    # glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
    # loss_glob = (1- torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

    loss_glob=0
    
    reg_tv = args.lambda_tv*get_image_prior_losses(target)
    total_loss = args.lambda_patch*loss_patch + args.lambda_c * content_loss+ reg_tv+ args.lambda_dir*loss_glob + args.lambda_c * loss_back

    detail_loss = {
        "loss_patch":loss_patch,
        "content_loss":content_loss,
        "reg_tv":reg_tv,
        "loss_glob":loss_glob,
        "loss_back":loss_back,
    }
    
    # print('getTotalLoss1 end')
    return total_loss,detail_loss

import torch
import numpy as np
import cv2

def save_image(input, filename):
    if isinstance(input, torch.Tensor):
        image = input.to("cpu").clone().detach()
        image = image.numpy().squeeze(0)
        image = np.transpose(image, (1, 2, 0))
    elif isinstance(input, np.ndarray):
        image = input
    else:
        raise TypeError("Input must be a PyTorch Tensor or a NumPy array")

    image = (image * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

import threading
import os

def StyleProcess(mask_path, img_path, cris_prompt, style_prompt, seed, save_epoch=False, size=128, pot_part=0.8):
    tmp_cris = cris_prompt.replace('.','').replace(' ','_')
    tmp_style = style_prompt.replace('.','').replace(' ','_')
    
    base_path = f'/data15/chenjh2309/soulstyler_org/outputs/size={size}/pot_part={pot_part}/{tmp_cris}/seed={seed}_{tmp_style}'
    img_output_image_path = os.path.join(base_path, 'ori_img.png')
    mask_output_image_path = os.path.join(base_path, 'mask_img.png')
    result_output_image_path = os.path.join(base_path, 'result_img.png')
    result_epoch_output_image_path = os.path.join(base_path, 'epoch/')

    
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    if save_epoch and not os.path.exists(result_epoch_output_image_path):
        os.makedirs(result_epoch_output_image_path)


    if os.path.exists(result_output_image_path):
        print(f"File '{result_output_image_path}' already exists. Exiting function.")
        return
    
    img = cv2.imread(img_path)
    # mask_img = cv2.imread(mask_path)
    mask = getMaskImg(img,config_path,model_pth,cris_prompt,isMask=False)

    img = load_image(img,mode="CV")
    mask = load_image(mask)

    # img = load2_image(img)
    # mask = load2_image(mask)

    img = img.to(device)
    mask = mask.to(device)
    # plt.imshow(utils.im_convert2(img))
    # plt.show()
    # plt.imshow(utils.im_convert2(mask))
    # plt.show()

    
    save_image(img, img_output_image_path)
    save_image(mask, mask_output_image_path)

    print("style img start", '='*50)

    output_image = getStyleImg(
        config_path, img, source="a Photo",
        prompt=style_prompt,
        seed=seed,
        get_total_loss=getTotalLoss1,
        mask=mask,
        save_epoch=save_epoch,
        path = result_epoch_output_image_path
    )

    save_image(output_image, result_output_image_path)


def main(case, stylelist):
    global input_data
    for item in input_data[case:case+1]:
        threading_list = []
        print(item)
        # print(item['style_prompts'])
        print(stylelist)
        for sty in item['style_prompts'][stylelist[0]: stylelist[1]]:
            threading_list.append(threading.Thread(target=StyleProcess, 
                                                   args=(item['mask_path'],
                                                         item['img_path'],
                                                         item['cris_prompt'] ,
                                                         sty['style'], 
                                                         sty['seed'], 
                                                         True,
                                                         64, 
                                                         0.95)))



        for t in threading_list:
            t.start()


        for t in threading_list:
            t.join()


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--case', type=int, help='An integer for the case')
    parser.add_argument('--stylelist', type=lambda s: [int(item) for item in s.split(',')], help='A list of integers for the style')
    args = parser.parse_args()

    main(args.case, args.stylelist)
