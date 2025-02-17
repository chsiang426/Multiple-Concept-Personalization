# DLCV Final Project ( Multiple Concept Personalization )

# Environment
```shell script=
conda create -n OMG python=3.10
conda activate OMG
conda install nvidia/label/cuda-12.4.0::cuda-toolkit -y
pip install diffusers transformers
pip install torch torchvision
pip install einops gdown accelerate protobuf sentencepiece omegaconf ipython
pip install git+https://github.com/facebookresearch/segment-anything.git
```

For GroundingDINO(Optional)
```shell script=
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
```
# How to run
## Download required checkpoint and data
```shell script=
bash download.sh
```

**🚨 After downloading checkpoints, you can directly run inference code based on our checkpoints.**  
**🚨 Please check [HERE](https://github.com/DLCV-Fall-2024/DLCV-Fall-2024-Final-2-darkmagic?tab=readme-ov-file#inference) (Inference part)**

## Training
```shell script=
bash train.sh <Path to yaml file>
```
* **Path to yaml file**: Yaml file for new concept(e.g. configs/cat2.yaml)
It will generate the model for new concept. The model is saved at experiments/<name>.

For original Mix-of-Show result, the path should be <name>.yaml, and the method in models should be mix-of-show.
For only train cross attention k, v layer, the path should be <name>_kv.yaml, and the method in models should be attn2-kv.
For orthogonal LoRA, the path should be <name>_ortho.yaml, and the method in models should be orthogonal.


## Fusing
### For gradient fusion
```shell script=
bash fuse.sh <Concept configure> <Saved path>
```
* **Concept configure**: Json file for new concepts to merge(e.g. datasets/data_cfgs/Data/cat2+dog6.json, datasets/data_cfgs/Data/cat2+dog6_kv.json)
* **Saved path**: Path for the merged model to save(e.g. experiments/composed_edlora/cat2+dog6, experiments/composed_edlora/cat2+dog6_kv)
It will merge the LoRAs for new concepts by gradient fusion.

datasets/data_cfgs/Data/all.json for merge all 8 concepts.
Make sure the trained model path for specific concept is experiments/<name>

### For orthogonal fusion
```shell script=
bash fuse_orthogonal.sh <Concept configure> <Saved path>
```
* **Concept configure**: Json file for new concepts to merge(e.g. datasets/data_cfgs/Data/cat2+dog6_ortho.json)
* **Saved path**: Path for the merged model to save(e.g. experiments/composed_edlora/cat2+dog6_ortho)
It will merge the LoRAs for new concepts by add the LoRA weights directily.


## Inference
### For four prompts with pre-defined region control condition(e.g. ./mask) and pretrained model
```shell script=
bash inference.sh <Path to annot file> <Path to output image folder>
```
* **Path to anno file**: Json file with all prompt and special token(e.g. prompt.json)  
* **Path to output image folder**: Folder to store output images  
It will generate 100 images per prompt.  
  
### For generate images without region control  
```shell script=
python3 ./src/generate_without_control.py --prompt <prompt> --save_path <save path> --checkpoint <pretrain weights> --num_images <number of images>
```
* **prompt**: Prompt with special tokens  
* **save path**: Folder to store all generated images  
* **checkpoint**: Pretrained Weights path  
* **num images**: Number of images to generate  

example:  
```shell script=
python3 ./src/generate_without_control.py --prompt "a <cat2>" --save_path ./output_images --checkpoint ./ckpt/model/combined_model_base --num_images 100
```
  
### For generate region control condition(require about 30G GPU memory)  
Region control generation needs region conditions(edge or sketch, with bounding boxes of tokens) to generate images, so we need to generate these condition first  
```shell script=
python3 ./src/generate_masks.py --prompt <prompt> --token <token> --save_path <save path> --DINO_checkpoint <pretrained DINO> --SAM_checkpoint <pretrained SAM> --num_images <number of image>
```
* **prompt**: Prompt without special prompt(e.g. prompt_4_clip_eval) to generate image template  
* **token**: Tokens to be detected, concatenating with '+'  
* **save path**: Folder to store all generated data  
* **DINO checkpoint**:  Pretrained weights for GroundingDINO  
* **SAM checkpoint**: Pretrained weights for SAM  
* **num images**: Number of images to generate  

example:  
```shell script=
python3 ./src/generate_masks.py --prompt "a cat with a dog" --token "cat+dog" --save_path ./region --DINO_checkpoint ./ckpt/GroundingDINO --SAM_checkpoint ./ckpt/sam_vit_h_4b8939.pth --num_images 10
```
It will generate sketch images and edge images, with a bbox.json file recording all bounding boxes, named  
  
### For generate images with region control condition  
```shell script=
python3 ./src/generate_with_control.py --prompt <prompt> --token <token> --save_path <save path> --checkpoint <checkpoint> --bbox_file <bbox json file> --control_mode <control mode> --guidance_path <guidance path> --num_images <number of images>
```
* **prompt**: Prompt without special prompt(e.g. prompt_4_clip_eval)  
* **token**: Special tokens, concatenating with '+'  
* **save path**: Folder to store all generated images  
* **checkpoint**: Pretrained Weights path  
* **bbox file**: Bbox json file path, generated from ./src/generate_masks.py  
* **control mode**: Edge guidance or Sketch guidance  
* **guidance path**: Folder of all region control condition images(e.g. ./region/sketch or ./region/edge)  
* **num images**: Number of images to generate per region condition 

example:  
```shell script=
python3 ./src/generate_with_control.py --prompt "a cat with a dog" --token "<cat2>+<dog6>" --save_path ./output_images --checkpoint ./ckpt/model/combined_model_base --bbox_file ./region/bbox.json --control_mode sketch --guidance_path ./region/sketch --num_images 10
```
