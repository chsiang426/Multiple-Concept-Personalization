#!/bin/bash

python3 ./src/four_prompt_generate.py --json_file $1 --save_path $2 --checkpoint ./ckpt/model/combined_model_base --num_images 10