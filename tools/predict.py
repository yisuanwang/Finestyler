import argparse
import os
import warnings

import cv2
import torch
import torch.nn.parallel
import torch.utils.data
from loguru import logger

import utils.config as config
from engine.engine import mask, predict
from model import build_segmenter
from utils.dataset import RefDataset
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def getMask(img,sent,config_path,model_pth):
    args = config.load_cfg_from_cfg_file(config_path)
    # build model
    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model).cuda()
    #load model
    checkpoint = torch.load(model_pth)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    # inference
    return mask(img,sent, model, args)

@logger.catch
def main():
    args = get_parser()
    args.output_dir = os.path.join(args.output_folder, args.exp_name+"_predict")
    args.vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(args.vis_dir, exist_ok=True)
    # logger
    setup_logger(args.output_dir,
                 distributed_rank=0,
                 filename="test.log",
                 mode="a")
    logger.info(args)
    # build dataset & dataloader
    img = cv2.imread("./testimg/1.jpg")
    print(img.shape)
    print(type(img))
    sent = "a white flower in the middle is made of glass"
    # build model
    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model).cuda()
    logger.info(model)

    args.model_dir = os.path.join(args.output_dir, "best_model.pth")
    if os.path.isfile(args.model_dir):
        logger.info("=> loading checkpoint '{}'".format(args.model_dir))
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_dir))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.model_dir))

    # inference
    predict(img,sent, model, args)


if __name__ == '__main__':
    main()
