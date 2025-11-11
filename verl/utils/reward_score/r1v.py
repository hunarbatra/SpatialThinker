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


def r1v_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def r1v_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        # Extract ground truth from <answer>...</answer> if present; otherwise, use raw string
        if "<answer>" in ground_truth and "</answer>" in ground_truth:
            gt_match = re.search(r"<answer>(.*?)</answer>", ground_truth)
            ground_truth_clean = gt_match.group(1).strip() if gt_match else ground_truth.strip()
        else:
            ground_truth_clean = ground_truth.strip()

        # Extract predicted answer from <answer>...</answer>
        pred_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        predicted_answer = pred_match.group(1).strip() if pred_match else predict_str.strip()

        if grade_answer(predicted_answer, ground_truth_clean):
            return 1.0

    except Exception:
        pass

    return 0.0

def r1v_compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    format_score = r1v_format_reward(predict_str)
    accuracy_score = r1v_accuracy_reward(predict_str, ground_truth)
    total_score = 0.5 * accuracy_score + 0.5 * format_score
    
    # print(f"[DEBUG] pred_str: {predict_str} \n\n gt_str: {ground_truth}, format: {format_score}, accuracy: {accuracy_score}, total_score: {total_score}")
    
    return {
        "overall": total_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
