# FineStyler

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eCM4wMZJAMQ0Z99TKVIxVygIRs6TzV0?usp=sharing)

---

**We provide a demo on colab that can be easily run !**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eCM4wMZJAMQ0Z99TKVIxVygIRs6TzV0?usp=sharing)


``⚠: demo in colab needs to download the model from the internet 
and may run slowly due to GPU limitations (30s-60s an iter, 
usually need to train about 200 iters to get better results)``


The top left is the original image, and the bottom left is the mask generated using the stylized objects via CRIS. The rest are stylized images generated with different stylized content. Our stylized translation results in various text conditions. The stylized images have the spatial structure of the content images with realistic textures corresponding to the text, while retaining the original style of the non-target regions.

![soulstyler examples](./img/examples3.jpg)


![soulstyler comp](./img/comp.jpg)

❤A more detailed description of the source code is in the process of being organized and will be posted in a readme in this repository when the paper is accepted.

## Pipeline

The overall architecture of the system.
![](./img/soulstructure.jpg)


## Run

### Environment
Python 3.10.13 & ptyorch 1.12.0+cu116 & ubuntu 20.04.1
```
$ conda create -n soulstyler python=3.10
$ conda activate soulstyler
$ pip install -r requirements.txt
$ pip install git+https://github.com/openai/CLIP.git
```
If you clone this repository locally, you will need to download this weight file to the root directory before running it. Running colab directly will automatically download the weights.
```
https://drive.google.com/file/d/10wo4R7sGWw5ITHpjtv3dIbIbkGpvkMiJ/view?usp=sharing
```


### 1. Single Image Style Transfer
Use this colab->[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eCM4wMZJAMQ0Z99TKVIxVygIRs6TzV0?usp=sharing)


### 2. Multiple Image Style Transfer
If you think it's long for him to generate a graph (due to the presence of random cropping, this takes a long time to reduce the loss, we'll follow up with optimizations 😳), you can run demo.py, which is a multi-threaded batch-run script that allows for multiple Stylized Content trainings to be performed simultaneously on a single GPU.

Some commands for testing can be found in democases.md (this is just a temporary draft file of commands, a more detailed description of the training, inference detail steps will follow in this GitHub repository)

```
CUDA_VISIBLE_DEVICES=0 python demo.py --case=0 --style=0,7
```

### To Do List
✅1. Colab online running demo

🔘2. CUDA version FineStyler

# License
This code and model are available only for non-commercial research purposes as defined in the LICENSE (i.e., MIT LICENSE). 
[Check the LICENSE](./LICENSE)

# Acknowledgment
This implementation is mainly built based on [CRIS](https://github.com/DerrickWang005/CRIS.pytorch), [CLIPstyler](https://github.com/cyclomon/CLIPstyler).

# ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yisuanwang/Finestyler&type=Date)](https://star-history.com/#yisuanwang/Finestyler&Date)