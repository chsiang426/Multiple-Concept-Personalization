import json
import argparse
import os
import random
import torch
from diffusers import DPMSolverMultistepScheduler

from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline
import re

def preprocess_prompt(prompt):
    special_tokens = re.findall(r'<.*?>', prompt)
    for token in special_tokens:
        prompt = prompt.replace(token, f"<{token[1:-1]}1> <{token[1:-1]}2>")
    return prompt

def load_model(model_path):
    # pretrained_model_path = '/tmp2/lins901121/DLCV_final/Mix-of-Show/experiments/merge/combined_model_base'
    enable_edlora = True  # True for edlora, False for lora
    pipe = EDLoRAPipeline.from_pretrained(model_path, scheduler=DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler'), torch_dtype=torch.float16).to('cuda')
    with open(f'{model_path}/new_concept_cfg.json', 'r') as fr:
        new_concept_cfg = json.load(fr)
    pipe.set_new_concept_cfg(new_concept_cfg)
    return pipe

def generate_image(pipe, prompt, save_dir, num_images):
    print(prompt)
    negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    for i in range(num_images):
        image = pipe(prompt, negative_prompt=negative_prompt, height=512, width=512, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(os.path.join(save_dir, f"{i}.png"))



def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--prompt', help='input prompt', required=True, type=str)
    parser.add_argument('--save_path', help='folder to save file', required=True, type=str)
    parser.add_argument('--checkpoint', help='checkpoint', required=True, type=str)
    parser.add_argument('--num_images', help='Number of images to generate', required=False, type=int, default=10)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    prompt = preprocess_prompt(args.prompt)
    pipe = load_model(args.checkpoint)
    generate_image(pipe, prompt, args.save_path, args.num_images)
    
