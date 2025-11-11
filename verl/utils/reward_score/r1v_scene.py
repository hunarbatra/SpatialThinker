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

import re
from typing import Dict

from mathruler.grader import grade_answer


# def r1v_format_reward(predict_str: str) -> float:
#     # pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
#     pattern = re.compile(r"<think>.*?</think>\s*<scene>.*?</scene>\s*<answer>.*?</answer>", re.DOTALL)
#     format_match = re.fullmatch(pattern, predict_str)
#     return 1.0 if format_match else 0.0

def r1v_format_reward(predict_str: str) -> float:
    # pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    pattern = re.compile(r"<observe>.*?</observe>\s*<scene>.*?</scene>\s*<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def acc_reward(pred: str, gt: str) -> float:
    return float(pred.strip().lower() == gt.strip().lower())

def r1v_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    pred_answer = extract_answer(predict_str)
    gt_answer = extract_answer(ground_truth)
    return acc_reward(pred_answer, gt_answer)

def r1v_scene_compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    format_score = r1v_format_reward(predict_str)
    if format_score == 0.0:
        return {
            "overall": 0.0,
            "format": 0.0,
            "accuracy": 0.0,
        }
    
    accuracy_score = r1v_accuracy_reward(predict_str, ground_truth)
    overall = 0.5 * accuracy_score + 0.5 * format_score
    
    return {
        "overall": overall,
        "format": format_score,
        "accuracy": accuracy_score,
    }
