import re
import json
import math
from typing import Dict, List


REQUIRED_KEYS = {"id", "object_name", "x", "y", "z"}

def is_valid_object(obj: Dict) -> bool:
    if not isinstance(obj, dict):
        return False
    if not REQUIRED_KEYS.issubset(obj.keys()):
        return False
    if not isinstance(obj["id"], int):
        return False
    if not isinstance(obj["object_name"], str):
        return False
    if not all(isinstance(obj[k], (int, float)) for k in ["x", "y", "z"]):
        return False
    return True

def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_scene(text: str) -> List[Dict]:
    match = re.search(r"<scene>(.*?)</scene>", text, re.DOTALL)
    try:
        return json.loads(match.group(1).strip()) if match else []
    except:
        return []

def format_reward(text: str) -> float:
    try:
        think = re.search(r"<think>.*?</think>", text, re.DOTALL)
        scene = re.search(r"<scene>(.*?)</scene>", text, re.DOTALL)
        answer = re.search(r"<answer>.*?</answer>", text, re.DOTALL)

        if not (think and scene and answer):
            return 0.0

        scene_json = json.loads(scene.group(1).strip())
        if not isinstance(scene_json, list) or not all(is_valid_object(obj) for obj in scene_json):
            return 0.0

        return 1.0
    except:
        return 0.0

def count_reward(pred_scene: List[Dict], gt_scene: List[Dict]) -> float:
    # return 0 reward if predicted scene is not a valid json! (format issue - hence, 0 for counting reward)
    if not isinstance(pred_scene, list) or not all(isinstance(obj, dict) for obj in pred_scene):
        return 0.0
    
    if len(pred_scene) == len(gt_scene):
        return 1.0
    else:
        return max(0.0, 1 - abs(len(pred_scene) - len(gt_scene)) / max(len(gt_scene), 1))

def acc_reward(pred: str, gt: str) -> float:
    return float(pred.strip().lower() == gt.strip().lower())

def spatial_reward( # euclidean distance spatial reward function
    pred_scene: List[Dict], 
    gt_scene: List[Dict],
    max_x: int = 6,
    max_y: int = 6,
    max_z: int = 1
) -> float:
    if not pred_scene or not gt_scene:
        return 0.0

    if not isinstance(pred_scene, list) or not all(is_valid_object(obj) for obj in pred_scene):
        return 0.0

    # Convert scenes to dicts keyed by object_name
    pred_dict = {obj["object_name"]: obj for obj in pred_scene}
    gt_dict = {obj["object_name"]: obj for obj in gt_scene}

    # Define maximum possible distance based on room bounds
    max_dist = math.sqrt((max_x)**2 + (max_y)**2 + (max_z)**2)  # x: [-3,3], y: [-3,3], z: [0,1]
    total_score = 0.0
    total_objects = len(gt_dict)

    for name, gt_obj in gt_dict.items():
        pred_obj = pred_dict.get(name)
        if pred_obj:
            dx = pred_obj["x"] - gt_obj["x"]
            dy = pred_obj["y"] - gt_obj["y"]
            dz = pred_obj["z"] - gt_obj["z"]
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            normalized_score = max(0.0, 1 - (dist / max_dist))
            total_score += normalized_score
        else:
            total_score += 0.0  # unmatched object = 0 score

    return total_score / total_objects if total_objects > 0 else 0.0

def spatial_compute_score(predict_str: str, ground_truth_str: str) -> Dict[str, float]:
    pred_answer = extract_answer(predict_str)
    gt_answer = extract_answer(ground_truth_str)
    
    pred_scene = extract_scene(predict_str)
    gt_scene = extract_scene(ground_truth_str)
    
    fr = format_reward(predict_str)
    cr = count_reward(pred_scene, gt_scene)
    ar = acc_reward(pred_answer, gt_answer)
    sr = spatial_reward(pred_scene, gt_scene)

    return {
        "overall": 0.25 * fr + 0.25 * cr + 0.25 * ar + 0.25 * sr,
        "format": fr,
        "count": cr,
        "accuracy": ar,
        "spatial": sr,
    }
