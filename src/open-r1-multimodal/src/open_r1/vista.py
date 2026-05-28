# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from open_r1.trainer.vista_config import VISTAConfig
from open_r1.trainer.vista_trainer import VISTATrainer
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import yaml
import json
import random
import math
from qwen_vl_utils import smart_resize

from open_r1.dynamic_crop import dynamic_crop_with_gt, generate_weighted_number

# ----------------------- Main Script -----------------------
@dataclass
class VISTAScriptArguments(ScriptArguments):
    """
    Script arguments for the VISTA training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        # default_factory=lambda: ["accuracy", "format", "length"],
        default_factory=lambda: ["point","format"],
        metadata={"help": "List of reward functions. Possible values: 'point', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )


@dataclass
class VISTAModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: VISTAScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        self.current_step = 0
        self.num_generations = 8

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):


        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["instruction"]},
                ],
            }
        example = self.list_data_dict[i]

        instruction = example["instruction"]

        if instruction[-1] == '.':
            instruction = instruction[:-1]

        QUESTION_TEMPLATE = 'Output the center point of the position corresponding to the instruction: {Question}. The output should just be the coordinates of a point, in the format [x,y].'
        def make_conversation_image(example):
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=instruction)},
                        ],
                    },
                ],
            }
        image_root = self.script_args.image_root
        if 'image_path' in example:
            image_path = example['image_path']
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                print(example)
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
            image = Image.open(image_path).convert("RGB")
            solution = example['gt']
            w, h = image.size
            sw_list = [w for i in range(self.num_generations)]
            sh_list = [h for i in range(self.num_generations)]

            images = [image for i in range(self.num_generations)]
            solutions = [solution for i in range(self.num_generations)]

            is_augmentation = generate_weighted_number(i)
            if is_augmentation:
                sw, sh = smart_resize(w,h,factor=32,min_pixels=1280*720,max_pixels=1280*720)
                image = image.resize((sw, sh))
                sw, sh = smart_resize(w,h,factor=32,min_pixels=568*1008,max_pixels=568*1008)
                images, gt_boxes = dynamic_crop_with_gt(image, solution, [sw, sh])

                for i in range(len(images)):
                    w, h = images[i].size
                    sw, sh = smart_resize(w,h,factor=32,min_pixels=568*1008,max_pixels=568*1008)
                    sw_list[i] = sw
                    sh_list[i] = sh
                    images[i] = images[i].resize((sw, sh))
                    solutions[i] = [int(gt_boxes[i][0] *1000),int(gt_boxes[i][1]*1000 ),int(gt_boxes[i][2]*1000 ),int(gt_boxes[i][3]*1000)]
            else:
                for i in range(len(images)):
                    w, h = images[i].size
                    sw, sh = smart_resize(w,h,factor=32,min_pixels=568*1008,max_pixels=568*1008)
                    sw_list[i] = sw
                    sh_list[i] = sh
                    images[i] = images[i].resize((sw, sh))
                    solutions[i] = [int(solutions[i][0] *1000),int(solutions[i][1]*1000 ),int(solutions[i][2]*1000 ),int(solutions[i][3]*1000)]

        else:
            image = None
            sw, sh = None, None
            solution = NotImplemented

        self.current_step += 1
        return {
            'image': images,
            'image_path': image_path,
            'width_resized': sw_list,
            'height_resized': sh_list,
            'problem': example['instruction'],
            'solution': solutions,
            'prompt': make_conversation_image(example)['prompt'] if 'image_path' in example else make_conversation(example)['prompt'],
        }

'''
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.5, the reward is 1.0, otherwise 0.0 .
    This is a hard reward, maybe the soft reward is better and could be used in the future .
'''
def iou_reward(completions, solution, **kwargs):
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        IOU = 0.0
        content = content.split('assistant\n')[-1]
        bbox_match = re.search(bbox_pattern, content.strip(), re.DOTALL)
        try:
            if bbox_match:
                bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)), float(bbox_match.group(3)), float(bbox_match.group(4))]
                sol = [float(num) for num in sol]
                IOU = iou(bbox, sol)
                if IOU > 0.5:
                    reward = 1.0
        except Exception:
            print(Exception, content, sol)

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"\n---------------------------------------------------- RANK: {dist.get_rank()}, iou: {IOU}, iou reward: {reward} ----------------------------------------------------\n")
                f.write(f"Image Path: \n{kwargs['image_path'][0]}\n")
                f.write(f"Resized Width: {kwargs['width_resized'][0]}, Resized Height: {kwargs['height_resized'][0]}\n")
                f.write(f"\nInstruction: \n{kwargs['problem'][0]}\n")
                f.write(f"Content: \n{content}\n")
                f.write(f"\nSolution: \n{sol}\n")
    return rewards


def point_reward(completions, solution, **kwargs):
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union

    def is_inbox(point, gt_coord):
        # print(point, gt_coord)
        x, y = point
        x1, y1, x2, y2 = gt_coord
        return x > x1 and x < x2 and y > y1 and y < y2

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        bbox_match = re.search(bbox_pattern, content.strip())
        is_match = 0
        try:
            if bbox_match:
                is_match = 1
                bbox = [float(bbox_match.group(1)), float(bbox_match.group(2))]
                point = bbox
                reward = is_inbox(point, sol)

        except Exception:
            print(Exception, content, sol)
            pass  
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"\[\s*\d+\s*,\s*\d+\s*\]"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content.split('assistant\n')[-1], re.DOTALL) for content in completion_contents]
    for i, num in enumerate([1.0 if match else 0.0 for match in matches]):
        if num < 1:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a") as f:
                    f.write(f"\n|||||||||||||||||||||||||||||||||||||||||||||||||||| RANK: {dist.get_rank()}, match: {num} ||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
                    f.write(f"Image Path: \n{kwargs['image_path'][0]}\n")
                    f.write(f"Resized Width: {kwargs['width_resized'][0]}, Resized Height: {kwargs['height_resized'][0]}\n")
                    f.write(f"\nInstruction: \n{kwargs['problem'][0]}\n")
                    f.write(f"\nformat not matched\n")
                    f.write(f"completion_contents: \n{completion_contents[i]}\n")
    return [1.0 if match else 0.0 for match in matches]

def normal_distribution(x):
    mu = 512
    sigma = 1 / np.sqrt(2 * np.pi)
    return np.exp(-np.pi * (x - mu)**2)


def length_reward(completions, **kwargs):
    length_predict = 256
    sigma = 64
    completion_contents = [completion[0]["content"] for completion in completions]
    reward_list = []
    for content in completion_contents:
        print(len(content))
        alpha = -math.log(0.5) / (128 ** 2)
        reward_list.append(math.exp(-alpha * ((len(content) - 256) ** 2)))
        return reward_list


def object_to_dict(obj):
    """
    将类实例的属性转换为字典。
    """
    return {key: value for key, value in obj.__dict__.items()}

def write_configs_to_txt(filename, *configs):
    """
    将多个配置字典写入 txt 文件。

    参数：
    - filename: 输出文件名。
    - configs: 多个配置字典。
    """
    with open(filename, 'a', encoding='utf-8') as f:
        for i, config in enumerate(configs):
            # 写入每个配置的标题
            if i == 0:
                f.write("=== VISTAScriptArguments ===\n")
            elif i == 1:
                f.write("\n=== VISTAConfig ===\n")
            elif i == 2:
                f.write("\n=== ModelConfig ===\n")

            # 写入配置内容
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

reward_funcs_registry = {
    "point": point_reward,
    "iou": iou_reward,
    "format": format_reward,
}

def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        from open_r1.vlm_modules import Qwen3VLModule

        return Qwen3VLModule
    elif "internvl" in model_name_or_path.lower():
        from open_r1.vlm_modules import InvernVLModule

        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")


def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)
    print(script_args.max_pixels, script_args.min_pixels)

    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    trainer_cls = VISTATrainer
    # Initialize the VISTA trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    # import rpdb; rpdb.set_trace(port=4444)
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((VISTAScriptArguments, VISTAConfig, VISTAModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # import rpdb; rpdb.set_trace(port=4444+int(dist.get_rank()))
    import torch.distributed as dist
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        if dist.get_rank()==0:
            write_configs_to_txt(log_path, object_to_dict(script_args), object_to_dict(training_args), object_to_dict(model_args))

    main(script_args, training_args, model_args)
