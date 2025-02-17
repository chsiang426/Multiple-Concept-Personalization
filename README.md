# DLCV Final Project: Multiple Concept Personalization

<div align="center">
  <img src="https://github.com/chsiang426/Multiple-Concept-Personalization/blob/master/poster.png" alt="Project Poster" width="700">
</div>

## Environment Setup

### 1. Create and activate the Conda environment
```shell
conda create -n OMG python=3.10
conda activate OMG
```

### 2. Install dependencies
```shell
conda install nvidia/label/cuda-12.4.0::cuda-toolkit -y
pip install diffusers transformers
pip install torch torchvision
pip install einops gdown accelerate protobuf sentencepiece omegaconf ipython
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 3. For GroundingDINO (Optional)
If you're using GroundingDINO for region control, clone and install it:
```shell
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
```

---

## How to Run

### 1. Download Required Checkpoints and Data
To get the necessary model checkpoints and data:
```shell
bash download.sh
```

> [!TIP]
> After downloading the checkpoints, you can run the inference code directly based on these checkpoints. For details, check [HERE](https://github.com/chsiang426/Multiple-Concept-Personalization/edit/master/README.md#inference).

### 2. Training
To train a new model based on a YAML configuration file:
```shell
bash train.sh <Path to yaml file>
```
- **Path to YAML file**: The path to the YAML file for the new concept (e.g., `configs/cat2.yaml`).
- The model will be saved under `experiments/<name>`.

#### Model Method Choices
- **For original Mix-of-Show results**: Use `<name>.yaml` with the method set to `mix-of-show`.
- **For training only the cross-attention k, v layer**: Use `<name>_kv.yaml` with the method set to `attn2-kv`.
- **For orthogonal LoRA**: Use `<name>_ortho.yaml` with the method set to `orthogonal`.

---

### 3. Fusing Models

#### For Gradient Fusion
To fuse LoRA models via gradient fusion:
```shell
bash fuse.sh <Concept configure> <Saved path>
```
- **Concept configure**: JSON file for new concepts to merge (e.g., `datasets/data_cfgs/Data/cat2+dog6.json`).
- **Saved path**: Path where the merged model will be saved (e.g., `experiments/composed_edlora/cat2+dog6`).

To merge all 8 concepts, use `datasets/data_cfgs/Data/all.json`.

Ensure the trained model path for each concept is `experiments/<name>`.

#### For Orthogonal Fusion
To fuse models using orthogonal fusion (add LoRA weights directly):
```shell
bash fuse_orthogonal.sh <Concept configure> <Saved path>
```
- **Concept configure**: JSON file for new concepts to merge (e.g., `datasets/data_cfgs/Data/cat2+dog6_ortho.json`).
- **Saved path**: Path to save the merged model (e.g., `experiments/composed_edlora/cat2+dog6_ortho`).

---

### 4. Inference

#### For Generating Images with Region Control
To run inference using pre-defined region control (e.g., masks) and pretrained models:
```shell
bash inference.sh <Path to annotation file> <Path to output image folder>
```
- **Path to annotation file**: JSON file with prompts and special tokens (e.g., `prompt.json`).
- **Path to output image folder**: Folder where the generated images will be saved.

This will generate 100 images per prompt.

#### For Generating Images Without Region Control
To generate images without region control:
```shell
python3 ./src/generate_without_control.py --prompt <prompt> --save_path <save path> --checkpoint <pretrain weights> --num_images <number of images>
```
- **prompt**: Prompt with special tokens.
- **save path**: Folder to store the generated images.
- **checkpoint**: Path to pretrained weights.
- **num images**: Number of images to generate.

Example:
```shell
python3 ./src/generate_without_control.py --prompt "a <cat2>" --save_path ./output_images --checkpoint ./ckpt/model/combined_model_base --num_images 100
```

#### For Generating Images with Region Control
Region control requires generating the condition (e.g., edge or sketch with bounding boxes). First, generate the conditions:
```shell
python3 ./src/generate_masks.py --prompt <prompt> --token <token> --save_path <save path> --DINO_checkpoint <pretrained DINO> --SAM_checkpoint <pretrained SAM> --num_images <number of images>
```
- **prompt**: Prompt without special tokens (e.g., `prompt_4_clip_eval`).
- **token**: Tokens to be detected, concatenated with a `+` (e.g., `cat+dog`).
- **save path**: Folder where the generated data will be stored.
- **DINO checkpoint**: Path to pretrained weights for GroundingDINO.
- **SAM checkpoint**: Path to pretrained weights for SAM.
- **num images**: Number of images to generate.

Example:
```shell
python3 ./src/generate_masks.py --prompt "a cat with a dog" --token "cat+dog" --save_path ./region --DINO_checkpoint ./ckpt/GroundingDINO --SAM_checkpoint ./ckpt/sam_vit_h_4b8939.pth --num_images 10
```

#### For Generating Images with Region Control Conditions
Once the conditions are ready, generate images with region control:
```shell
python3 ./src/generate_with_control.py --prompt <prompt> --token <token> --save_path <save path> --checkpoint <checkpoint> --bbox_file <bbox json file> --control_mode <control mode> --guidance_path <guidance path> --num_images <number of images>
```
- **prompt**: Prompt without special tokens (e.g., `prompt_4_clip_eval`).
- **token**: Special tokens, concatenated with `+`.
- **save path**: Folder to store generated images.
- **checkpoint**: Path to pretrained weights.
- **bbox file**: Path to the bounding box JSON file generated by `./src/generate_masks.py`.
- **control mode**: Either `edge` or `sketch` guidance.
- **guidance path**: Folder of generated region control images (e.g., `./region/sketch` or `./region/edge`).
- **num images**: Number of images to generate per region condition.

Example:
```shell
python3 ./src/generate_with_control.py --prompt "a cat with a dog" --token "<cat2>+<dog6>" --save_path ./output_images --checkpoint ./ckpt/model/combined_model_base --bbox_file ./region/bbox.json --control_mode sketch --guidance_path ./region/sketch --num_images 10
```
