# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import math
import os
import glob
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}

class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        mixed_data: bool = False,
        text_only: bool = False,
        max_prompt_length: int = 2048,
        truncation: str = "error",
        format_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.mixed_data = mixed_data
        self.text_only = text_only
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.format_prompt = format_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.shuffle = shuffle
        self.seed = seed

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        def get_data_files(data_path, data_split):
            # Ensure path exists
            if not os.path.exists(data_path):
                raise ValueError(f"[ERROR] Path does not exist: {data_path}")
            
            base_dir = data_path  # already the correct base directory

            # Glob for split-specific files
            pattern = os.path.join(base_dir, f"{data_split}-*.parquet")
            
            files = glob.glob(pattern)
            files.sort()

            if not files:
                raise ValueError(f"No files found for split '{data_split}' at path '{data_path}'")

            data_files = {data_split: files}

            # If train, add val files too (for inspection or eval)
            if data_split == "train":
                val_pattern = os.path.join(base_dir, "val-*.parquet")
                val_files = glob.glob(val_pattern)
                val_files.sort()
                if val_files:
                    data_files["val"] = val_files

            return data_files
        
        if os.path.isdir(data_path):
            self.dataset = load_dataset(
                "parquet", 
                data_files=get_data_files(data_path, data_split),
                split=data_split
            )
        elif os.path.isfile(data_path):
            self.dataset = load_dataset(
                "parquet", 
                data_files=get_data_files(data_path, data_split),
                split=data_split
            )
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)
            
        # remove image_key value for every odd row in dataset
        if self.mixed_data:
            def remove_image_column(example, idx):
                if idx % 2 == 0 and self.image_key in example and "<image>" in example[self.prompt_key]:
                    example[self.prompt_key] = example[self.prompt_key].replace("<image>", "").strip()
                return example

            self.dataset = self.dataset.map(
                remove_image_column,
                with_indices=True,
                desc="Removing <image> from prompt_key"
            )

        # Shuffle the dataset if requested
        if self.shuffle:
            import random
            random.seed(self.seed)
            self.dataset = self.dataset.shuffle(seed=self.seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        prompt_str: str = row_dict[self.prompt_key]
        if self.format_prompt:
            # prompt_str = prompt_str + " " + self.format_prompt.strip()
            prompt_str = self.format_prompt.strip() + " " + prompt_str
            
        if self.text_only:
            prompt_str = prompt_str.replace("<image>", "").strip()
            
        # print(f'TEXT ONLY: {self.text_only}')
        # prompt_str = prompt_str.replace("<image>", "").strip()
        # print(f'REMOVED <image> from prompt_str: {prompt_str[:20]}')
        
        # if self.image_key in row_dict:
        if ("<image>" in prompt_str) and (self.image_key in row_dict or "image" in row_dict or "images" in row_dict) and row_dict[self.image_key] is not None:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            # print(f"[DEBUG DATA] Row: {index}")
            # Ensure <image> is only at the start
            prompt_str = prompt_str.replace("<image>", "").strip()
            prompt_str = "<image> " + prompt_str

            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages = [{"role": "user", "content": content_list}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            image_or_images = row_dict.pop(self.image_key)
            if not isinstance(image_or_images, list):
                image_or_images = [image_or_images]  # wrap single image
            if any(im is None for im in image_or_images):
                raise ValueError(f"Image is None at index {index} despite <image> token present. Check data logic.")
            images = [self.process_image(image) for image in image_or_images]
            # model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            model_inputs = self.processor(images, [prompt], return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            row_dict["multi_modal_data"] = {"image": images}
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)
            
        else:
            # print(f"[DEBUG DATA] Text-only Row: {index}, image_key: {self.image_key}")
            
            messages = [{"role": "user", "content": prompt_str}]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = row_dict.pop(self.answer_key)
        
        return row_dict
