
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
        'cris_prompt' : "The white horse under Napoleon.",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/Napoleon.jpg",
        'cris_prompt' : "white horse",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/Napoleon.jpg",
        'cris_prompt' : "horse",
        'style_prompts' : prompt_list
    },
    # ddddddddddddd
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
        'img_path' : "./testimg/tjl.png",
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


# 0-21 Style
cd soulstyler_org/
conda activate soulstyler

# ship finish
# CUDA_VISIBLE_DEVICES=0 python demo.py --case=0 --style=0,7 d02-cuda0 
# CUDA_VISIBLE_DEVICES=4 python demo.py --case=0 --style=8,14 d01-cuda4 finish
# CUDA_VISIBLE_DEVICES=5 python demo.py --case=0 --style=15,21 d01-cuda5
# CUDA_VISIBLE_DEVICES=2 python demo.py --case=0 --style=0,20 补齐剩下的 d01-cuda02 runing

# plane
# CUDA_VISIBLE_DEVICES=3 python demo.py --case=1 --style=0,4 # d01-cuda3
# CUDA_VISIBLE_DEVICES=4 python demo.py --case=1 --style=4,8 # d02-cuda4
# CUDA_VISIBLE_DEVICES=5 python demo.py --case=1 --style=8,12 # d01-cuda5
# CUDA_VISIBLE_DEVICES=6 python demo.py --case=1 --style=12,16 # d02-cuda6
# CUDA_VISIBLE_DEVICES=7 python demo.py --case=1 --style=16,20 # d01-cuda7

# flower
# CUDA_VISIBLE_DEVICES=5 python demo.py --case=2 --style=0,7 # d02-cuda5 finish
# CUDA_VISIBLE_DEVICES=6 python demo.py --case=2 --style=7,14 # d02-cuda6 finish
# CUDA_VISIBLE_DEVICES=6 python demo.py --case=2 --style=15,21 # d01-cuda6 finish



 <!-- runing -->
# house finish
# CUDA_VISIBLE_DEVICES=4 python demo.py --case=3 --style=0,7 # d01-cuda4 finish
# CUDA_VISIBLE_DEVICES=5 python demo.py --case=3 --style=8,14 # d01-cuda5 finish
# CUDA_VISIBLE_DEVICES=0 python demo.py --case=3 --style=14,20 # d02-cuda1
CUDA_VISIBLE_DEVICES=1 python demo.py --case=3 --style=0,20 runing


<!-- v100 -->
cd /data3/chenjh/soulstyler/code/soulstyler_org
conda activate /data3/chenjh/conda_envs/soulstyler

# women
# CUDA_VISIBLE_DEVICES=0 python demo.py --case=4 --style=0,5 # v100-cuda0-tmux16
# CUDA_VISIBLE_DEVICES=1 python demo.py --case=4 --style=6,10 # v100-cuda1-tmux17
# CUDA_VISIBLE_DEVICES=2 python demo.py --case=4 --style=11,15 # v100-cuda2-tmux18
# CUDA_VISIBLE_DEVICES=3 python demo.py --case=4 --style=16,21 # v100-cuda3-tmux19

# Napoleon 
# CUDA_VISIBLE_DEVICES=6 python demo.py --case=5 --style=0,7 # d02-cuda5
# CUDA_VISIBLE_DEVICES=5 python demo.py --case=5 --style=7,14 # d02-cudda6
# CUDA_VISIBLE_DEVICES=6 python demo.py --case=5 --style=14,20 # d02-cuda1

# apple
CUDA_VISIBLE_DEVICES=5 python demo.py --case=6 --style=0,7 
CUDA_VISIBLE_DEVICES=6 python demo.py --case=6 --style=7,14 
CUDA_VISIBLE_DEVICES=5 python demo.py --case=6 --style=14,20

# bigship
# CUDA_VISIBLE_DEVICES=1 python demo.py --case=7 --style=0,5 # v100-cuda0-tmux16
# CUDA_VISIBLE_DEVICES=2 python demo.py --case=7 --style=6,10 # v100-cuda1-tmux17
# CUDA_VISIBLE_DEVICES=3 python demo.py --case=7 --style=11,15 # v100-cuda2-tmux18
# CUDA_VISIBLE_DEVICES=7 python demo.py --case=7 --style=16,20 # v100-cuda3-tmux19


CUDA_VISIBLE_DEVICES=5 python demo.py --case=9 --style=0,7 
CUDA_VISIBLE_DEVICES=6 python demo.py --case=9 --style=7,14 
CUDA_VISIBLE_DEVICES=2 python demo.py --case=9 --style=14,20

CUDA_VISIBLE_DEVICES=1 python demo.py --case=11 --style=0,5 
CUDA_VISIBLE_DEVICES=2 python demo.py --case=11 --style=5,10 
CUDA_VISIBLE_DEVICES=3 python demo.py --case=11 --style=10,15
CUDA_VISIBLE_DEVICES=7 python demo.py --case=11 --style=15,20

CUDA_VISIBLE_DEVICES=5 python demo.py --case=12 --style=0,7 
CUDA_VISIBLE_DEVICES=6 python demo.py --case=12 --style=7,14 
CUDA_VISIBLE_DEVICES=5 python demo.py --case=12 --style=14,20