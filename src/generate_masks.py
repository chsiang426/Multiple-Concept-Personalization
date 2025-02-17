import torch
from diffusers import StableDiffusion3Pipeline
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, predict
from segment_anything import build_sam, SamPredictor
import groundingdino.datasets.transforms as T
import numpy as np
import os
from typing import Tuple, List
from torchvision.utils import save_image
import json
import argparse
import cv2


def load_image_dino(image_source) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    args = SLConfig.fromfile(ckpt_config_filename)
    model = build_model(args)
    args.device = device
    checkpoint = torch.load(os.path.join(repo_id, filename), map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(filename, log))
    _ = model.eval()
    return model

def load_model(ckpt_repo_id, sam_checkpoint):
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = os.path.join(ckpt_repo_id, "GroundingDINO_SwinB.cfg.py")
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.cuda()
    sam_predictor = SamPredictor(sam)
    return pipe, groundingdino_model, sam_predictor

def predict_mask(segmentmodel, sam, image, TEXT_PROMPT, num_of_object):
    image_source, image = load_image_dino(image)
    boxes, logits, phrases = predict(
        model=segmentmodel,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=0.4,
        text_threshold=0.25
    )
    # print(boxes, logits, phrases)
    
    if boxes.shape[0] < num_of_object:
        print("pass")
        return None, None
    _, idx = torch.topk(logits, num_of_object)
    boxes = boxes[idx]
    sam.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    
    transformed_boxes = sam.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
    masks, _, _ = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks, boxes_xyxy

def same_box(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    bb1_rate = intersection_area / float(bb1_area)
    bb2_rate = intersection_area / float(bb2_area)
    if bb1_rate > 0.5 or bb2_rate > 0.5:
        return True
    return False

def sketch_and_edge(image, masks):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blur_gray, 150, 200)
    sketch = np.zeros_like(image).astype("uint8")
    for mask in masks:
        mask = mask.squeeze(0).cpu().numpy().astype("uint8") * 255
        print(np.max(mask))
        hole = cv2.erode(mask, kernel)
        sketch += (mask - hole)
    return edges, sketch.astype("uint8")
    


def generate_control(pipe, detect_model, sam, prompt, tokens, num_images, save_dir):
    idx = 0
    bbox_list = []
    os.makedirs(os.path.join(save_dir, "sketch"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "edge"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "image"), exist_ok=True)

    while idx < num_images:
        image = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0
        ).images[0]
        masks, bbox_xyxy_list, token_list = [], [], []
        no_match = False
        for token in tokens:
            n = prompt.count(token)
            print(token, n)
            mask, bbox_xyxy = predict_mask(detect_model, sam, image, token, n)
            if bbox_xyxy is None:
                no_match = True
                break
            for i in range(n):
                masks.append(mask[i])
                bbox_xyxy_list.append(torch.floor(bbox_xyxy[i]).to(torch.int))
                token_list.append(f"{token}{i}")
        if no_match:
            continue
        
        Is_sameBox = False
        for i in range(len(bbox_xyxy_list)):
            for j in range(i+1, len(bbox_xyxy_list)):
                if same_box(bbox_xyxy_list[i], bbox_xyxy_list[j]):
                    Is_sameBox = True
                    break
            if Is_sameBox:
                break
        if Is_sameBox:
            print("same box")
            continue
        edge, sketch = sketch_and_edge(image, masks)
        region = dict({})
        for token, bbox_xyxy in zip(token_list, bbox_xyxy_list):
            region[token] = bbox_xyxy.tolist()
        bbox_list.append(region)
        image.save(os.path.join(save_dir, "image", f"{idx}.png"))
        cv2.imwrite(os.path.join(save_dir, "sketch", f"{idx}.png"), sketch)
        cv2.imwrite(os.path.join(save_dir, "edge", f"{idx}.png"), edge)
        idx += 1
    with open(os.path.join(save_dir, "bbox.json"), "w") as f:
	    json.dump(bbox_list, f, indent=1)


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--prompt', help='input prompt', required=True, type=str)
    parser.add_argument("--token", help='Replace token name', required=True, type=str)
    parser.add_argument('--save_path', help='folder to save file', required=True, type=str)
    parser.add_argument('--DINO_checkpoint', help='GroundingDINO_checkpoint', required=True, type=str)
    parser.add_argument('--SAM_checkpoint', help='SAM_checkpoint', required=True, type=str)
    parser.add_argument('--num_images', help='Number of images to generate', required=False, type=int, default=10)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # pipe, detect_model, sam = build_model("/tmp2/lins901121/DLCV_final/OMG/checkpoint/GroundingDINO", "/tmp2/lins901121/DLCV_final/OMG/checkpoint/sam/sam_vit_h_4b8939.pth")
    pipe, detect_model, sam = load_model(args.DINO_checkpoint, args.SAM_checkpoint)
    generate_control(pipe, detect_model, sam, args.prompt, args.token.split("+"), args.num_images, args.save_path)