import json
import os
import argparse
import re
import hashlib

import torch
from diffusers import DPMSolverMultistepScheduler
from diffusers.models import T2IAdapter
from PIL import Image

from mixofshow.pipelines.pipeline_regionally_t2iadapter import RegionallyT2IAdapterPipeline


def prepare_text(prompt, region_prompts, height, width):
    '''
    Args:
        prompt_entity: [subject1]-*-[attribute1]-*-[Location1]|[subject2]-*-[attribute2]-*-[Location2]|[global text]
    Returns:
        full_prompt: subject1, attribute1 and subject2, attribute2, global text
        context_prompt: subject1 and subject2, global text
        entity_collection: [(subject1, attribute1), Location1]
    '''
    print(height, width)
    region_collection = []

    regions = region_prompts.split('|')

    for region in regions:
        if region == '':
            break
        prompt_region, neg_prompt_region, pos = region.split('-*-')
        prompt_region = prompt_region.replace('[', '').replace(']', '')
        neg_prompt_region = neg_prompt_region.replace('[', '').replace(']', '')
        pos = eval(pos)
        if len(pos) == 0:
            pos = [0, 0, 1, 1]
        else:
            pos[0], pos[2] = pos[0] / height, pos[2] / height
            pos[1], pos[3] = pos[1] / width, pos[3] / width

        region_collection.append((prompt_region, neg_prompt_region, pos))
    return (prompt, region_collection)

def sample_image(pipe,
    input_prompt,
    input_neg_prompt=None,
    generator=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    sketch_adaptor_weight=1.0,
    region_sketch_adaptor_weight='',
    keypose_adaptor_weight=1.0,
    region_keypose_adaptor_weight='',
    **extra_kargs
):

    keypose_condition = extra_kargs.pop('keypose_condition')
    if keypose_condition is not None:
        keypose_adapter_input = [keypose_condition] * len(input_prompt)
    else:
        keypose_adapter_input = None

    sketch_condition = extra_kargs.pop('sketch_condition')
    if sketch_condition is not None:
        sketch_adapter_input = [sketch_condition] * len(input_prompt)
    else:
        sketch_adapter_input = None

    images = pipe(
        prompt=input_prompt,
        negative_prompt=input_neg_prompt,
        keypose_adapter_input=keypose_adapter_input,
        keypose_adaptor_weight=keypose_adaptor_weight,
        region_keypose_adaptor_weight=region_keypose_adaptor_weight,
        sketch_adapter_input=sketch_adapter_input,
        sketch_adaptor_weight=sketch_adaptor_weight,
        region_sketch_adaptor_weight=region_sketch_adaptor_weight,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        **extra_kargs).images
    return images



def build_model(pretrained_model, device, mode):
    pipe = RegionallyT2IAdapterPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16).to(device)
    assert os.path.exists(os.path.join(pretrained_model, 'new_concept_cfg.json'))
    with open(os.path.join(pretrained_model, 'new_concept_cfg.json'), 'r') as json_file:
        new_concept_cfg = json.load(json_file)
    pipe.set_new_concept_cfg(new_concept_cfg)
    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(pretrained_model, subfolder='scheduler')
    pipe.keypose_adapter = T2IAdapter.from_pretrained('TencentARC/t2iadapter_openpose_sd14v1', torch_dtype=torch.float16).to(device)
    if mode == 'edge':
        pipe.sketch_adapter = T2IAdapter.from_pretrained('TencentARC/t2iadapter_canny_sd14v1', torch_dtype=torch.float16).to(device)
    else:
        pipe.sketch_adapter = T2IAdapter.from_pretrained('TencentARC/t2iadapter_sketch_sd14v1', torch_dtype=torch.float16).to(device)
    pipe.adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_canny_sd14v1", torch_dtype=torch.float16).to(device)
    return pipe

def processing_region_prompt(bbox_list, tokens, negative_prompt):
    prompt_list = []
    for b_list in bbox_list:
        prompt = ""
        for token in tokens:
            region_prompt = f"[a <{token[1:-1]}1> <{token[1:-1]}2>]"
            for k in b_list.keys():
                clean_k = re.sub(r"[0-9]", "", k)
                if clean_k in token:
                    region = [b_list[k][1], b_list[k][0], b_list[k][3], b_list[k][2]]
                    region = f"{region}"
                    b_list.pop(k)
                    break
            # {region1_prompt}-*-{region1_neg_prompt}-*-{region1}
            if prompt == "":
                prompt += f"{region_prompt}-*-{negative_prompt}-*-{region}"
            else:
                prompt += f"|{region_prompt}-*-{negative_prompt}-*-{region}"
        prompt_list.append(prompt)
    # print(prompt_list)
    return prompt_list

def generate_images(prompt, region_prompt, negative_prompt, save_path, num_images, condition, adaptor_weight, iter):
    condition = Image.open(condition).convert('L')
    width, height = condition.size
    print('use sketch condition')
    print(f'save to: {save_path}')
    os.makedirs(save_path, exist_ok=True)
    for i in range(iter * num_images, (iter + 1) * num_images):
        kwargs = {
            'sketch_condition': condition,
            'keypose_condition': None,
            'height': height,
            'width': width,
        }
    
        prompts = [prompt]
        prompts_rewrite = [region_prompt]
        input_prompt = [prepare_text(p, p_w, height, width) for p, p_w in zip(prompts, prompts_rewrite)]
        image = sample_image(
            pipe,
            input_prompt=input_prompt,
            input_neg_prompt=[negative_prompt] * len(input_prompt),
            sketch_adaptor_weight=adaptor_weight,
            **kwargs)
        save_name = f'result_{i}.png'
        image[0].resize((512, 512)).save(os.path.join(save_path, save_name))

def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--prompt', help='input prompt', required=True, type=str)
    parser.add_argument('--token', help="token list", required=True, type=str)
    parser.add_argument('--save_path', help='folder to save file', required=True, type=str)
    parser.add_argument('--checkpoint', help='checkpoint', required=True, type=str)
    parser.add_argument("--bbox_file", help="Bounding box json", required=True, type=str)
    parser.add_argument('--num_images', help='Number of images to generate', required=False, type=int, default=10)
    parser.add_argument("--control_mode", default="edge", type=str)
    parser.add_argument('--guidance_path', help="path to guidance folder", required=True, default=None, type=str)
    parser.add_argument('--guidance_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_guidance_adaptor_weight', default='', type=str)
    parser.add_argument('--keypose_condition', default=None, type=str)
    parser.add_argument('--keypose_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_keypose_adaptor_weight', default='', type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    with open(args.bbox_file, 'r') as file:
        bbox = json.load(file)
    context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    prompt_list = processing_region_prompt(bbox, args.token.split("+"), context_neg_prompt)
    
    pipe = build_model(args.checkpoint, device, args.control_mode)

    for i, prompt in enumerate(prompt_list):
        condition = os.path.join(args.guidance_path, f"{i}.png")
        adaptor_weight = args.guidance_adaptor_weight
        generate_images(args.prompt, prompt, context_neg_prompt, args.save_path, args.num_images, condition, adaptor_weight, i)

        # cmd = f"python ./src/regionally_controlable_sampling.py --pretrained_model={args.checkpoint} --save_dir='{args.save_path}' --prompt='{args.prompt}' \
        # --negative_prompt='{context_neg_prompt}' --prompt_rewrite='{prompt}' --suffix='baseline' \
        # --seed=42 --sketch_adaptor_weight={sketch_adaptor_weight} --sketch_condition={sketch_condition} --iter {i}"
        # print(cmd)
        # returned_value = os.system("CUDA_VISIBLE_DEVICES=3 " + cmd)
        # print(returned_value)