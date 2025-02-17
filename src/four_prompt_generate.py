import json
import argparse
import os


def processing_token(tokens):
    s = ""
    for i, token in enumerate(tokens):
        if i == 0:
            s += token
        else:
            s += "+" + token
    return s

def generate_with_control(prompt_dict, save_path, checkpoint, num_images, k):
    prompt = prompt_dict["prompt_4_clip_eval"]
    token = processing_token(prompt_dict["token_name"])
    code = os.path.join("src", "generate_with_control.py")
    bbox_file = os.path.join("mask", k, "bbox.json")
    condition_path = os.path.join("mask", k, "edge")
    cmd = f"python3 {code} --checkpoint {checkpoint} --num_images {num_images} \
            --prompt '{prompt}' --save_path {save_path} --bbox_file {bbox_file} \
            --guidance_path {condition_path} --token '{token}'"
    print(cmd)
    returned_value = os.system(cmd)
    print(returned_value)


def generate_without_control(prompt_dict, save_path, checkpoint, num_images):
    code = os.path.join("src", "generate_without_control.py")
    cmd = f"python3 {code} --checkpoint {checkpoint} --num_images {num_images * 10} --save_path {save_path} --prompt '{prompt_dict['prompt']}'"
    print(cmd)
    returned_value = os.system(cmd)
    print(returned_value)

def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--json_file', help='json file', required=True, type=str)
    parser.add_argument('--save_path', help='folder to save file', required=True, type=str)
    parser.add_argument('--checkpoint', help='checkpoint', required=True, type=str)
    parser.add_argument('--num_images', help='Number of images to generate', required=False, type=int, default=10)
    return parser.parse_args()

if __name__ in "__main__":
    args = parse_args()
    with open(args.json_file, 'r') as file:
        prompt_file = json.load(file)
    for k in prompt_file.keys():
        save_path = os.path.join(args.save_path, k)
        if k == "0" or k == "2":
            generate_with_control(prompt_file[k], save_path, args.checkpoint, args.num_images, k)
        else:
            generate_without_control(prompt_file[k], save_path, args.checkpoint, args.num_images)