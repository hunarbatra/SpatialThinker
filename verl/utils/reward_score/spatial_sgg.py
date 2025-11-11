import json
import math
import re
from functools import lru_cache
from typing import Dict, List

import numpy as np
import spacy
from scipy.optimize import linear_sum_assignment


# NLP model for label similarity 
nlp = spacy.load("en_core_web_md", disable=["parser","ner","tagger"])

OBJ_WEIGHT  = 0.5 # 0.2 overall (0.5 * 0.4)
REL_WEIGHT  = 0.5 # 0.2 overall
SEM_W = 2.0 # label similarity weight
IOU_W = 1.0 # spatial overlap IoU weight
L1_W = 5.0 # L1 distance weight

def scale_box(box, scale):
    sw, sh = scale
    return [box[0]*sw, box[1]*sh, box[2]*sw, box[3]*sh]

def refine_node_edge(label: str) -> str:
    """unify case/punct so 'fire-hydrant' == 'fire hydrant'."""
    return label.replace("_", " ").replace("-", " ").strip().lower()

@lru_cache(maxsize=4096)
def _doc(tok: str):
    return nlp(tok)

def sem_sim(a: str, b: str) -> float:
    """
    Cosine similarity between *labels* (ignores trailing “.123” id).
    """
    clean_a = refine_node_edge(a.split('.')[0]) # e.g. keep “chair” from “chair.5”
    clean_b = refine_node_edge(b.split('.')[0])
    return _doc(clean_a).similarity(_doc(clean_b))

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return 0.0 if union == 0 else inter / union

def compute_giou(boxA, boxB):
    # intersection
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)

    # union
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0

    # enclosing box
    cx1, cy1 = min(boxA[0], boxB[0]), min(boxA[1], boxB[1])
    cx2, cy2 = max(boxA[2], boxB[2]), max(boxA[3], boxB[3])
    c_area = (cx2 - cx1) * (cy2 - cy1)

    if c_area == 0:
        return iou

    giou = iou - (c_area - union_area) / c_area
    mapped_giou = (giou + 1.0) / 2.0
    return mapped_giou


def compute_ciou(boxA, boxB, eps=1e-7):
    """
    Compute Complete IoU (CIoU) between two bounding boxes,
    and return the value mapped to [0, 1] range via (ciou + 1) / 2.

    boxA, boxB: [x1, y1, x2, y2] in normalized coordinates (0–1).
    """
    # Widths and heights
    wA, hA = boxA[2] - boxA[0], boxA[3] - boxA[1]
    wB, hB = boxB[2] - boxB[0], boxB[3] - boxB[1]

    # Areas
    areaA = wA * hA
    areaB = wB * hB

    # Intersection
    xI1 = max(boxA[0], boxB[0])
    yI1 = max(boxA[1], boxB[1])
    xI2 = min(boxA[2], boxB[2])
    yI2 = min(boxA[3], boxB[3])

    inter_w = max(0.0, xI2 - xI1)
    inter_h = max(0.0, yI2 - yI1)
    inter_area = inter_w * inter_h

    # Union
    union = areaA + areaB - inter_area + eps
    iou = inter_area / union

    # Center points
    cxA = (boxA[0] + boxA[2]) / 2
    cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2
    cyB = (boxB[1] + boxB[3]) / 2

    # Center distance squared
    center_dist_sq = (cxA - cxB) ** 2 + (cyA - cyB) ** 2

    # Smallest enclosing box
    enclose_x1 = min(boxA[0], boxB[0])
    enclose_y1 = min(boxA[1], boxB[1])
    enclose_x2 = max(boxA[2], boxB[2])
    enclose_y2 = max(boxA[3], boxB[3])
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # Aspect ratio consistency term v
    v = (4 / (math.pi ** 2)) * (math.atan(wB / (hB + eps)) - math.atan(wA / (hA + eps))) ** 2

    # Trade-off factor α
    with_v = (1 - iou) + v
    alpha = v / with_v if with_v != 0 else 0.0

    # Final CIoU
    ciou = iou - (center_dist_sq / enclose_diag_sq + alpha * v)

    # Map to [0,1]
    mapped_ciou = (ciou + 1) / 2

    return mapped_ciou

def box_L1(a, b):
    # calculate the sum of absolute differences between the coordinates of the two boxes
    l1_dist = sum(abs(a - b) for a, b in zip(a, b))
    return l1_dist

# Hungarian matcher (object level)
def _freeze_objs(objs):
    """
    Turn a list of scene-graph objects into a hashable key:
    ('id', (x1, y1, x2, y2))
    """ 
    return tuple(
        (o["id"], tuple(o["bbox"])) for o in objs
    )
    
def _cost(gt, pr):
    """Lower = better (for Hungarian)."""
    iou = compute_ciou(pr["bbox"], gt["bbox"])
    # l1 = box_L1(pr["bbox=  gt["bbox"])
    sim = sem_sim(pr["id"], gt["id"])
    
    return (
        SEM_W * (1.0 - sim) + # label similarity
        IOU_W * (1.0 - iou)  # spatial overlap
        # L1_W  *  l1 # L1 distance for fine-grained pixel-level matching
    )

def _hungarian(gt_objs, pr_objs):
    P, G = len(pr_objs), len(gt_objs)
    pad = max(0, G - P) # add rows if preds < GT
    C = np.zeros((P + pad, G))

    for i, p in enumerate(pr_objs): # only real preds filled
        for j, g in enumerate(gt_objs):
            C[i, j] = _cost(p, g)

    if pad:
        C[P:, :] = 1e5 # high cost for padded rows

    rows_idx, cols_idx = linear_sum_assignment(C)

    mapping = [None] * G # GT-indexed result
    for r, c in zip(rows_idx, cols_idx):
        if r < P: # ignore assignments on dummy rows
            mapping[c] = r # GT j → pred i
            
    return mapping # len == #GT

@lru_cache(maxsize=4096)
def _bi_match_cached(gt_key, pr_key):
    """
    Hashable wrapper so we can cache across identical calls.
    Converts the frozen keys back to lists of dicts for the impl function
    """
    gt_objs = [
        {"id": obj_id, "bbox": list(bbox)}
        for obj_id, bbox in gt_key
    ]
    pr_objs = [
        {"id": obj_id, "bbox": list(bbox)}
        for obj_id, bbox in pr_key
    ]
    return _hungarian(gt_objs, pr_objs)

def bi_match(gt_objs, pr_objs):
    """
    Bi-directional matching between GT and pred objs.
    Thin cached front-end that keeps the original signature.
    """
    gt_objs = _freeze_objs(gt_objs) # freeze GT objs - hashable key
    pr_objs = _freeze_objs(pr_objs) # freeze pred objs - hashable key
    
    return _bi_match_cached(gt_objs, pr_objs)

def bi_match_triplets(gt_rels, pred_rels):
    """
    Bi-directional semantic matching between ground-truth and predicted relation triplets.
    Returns best alignment (pred ↔ gt) with semantic similarity cost.
    """
    num_gt = len(gt_rels)
    num_pred = len(pred_rels)
    pad = max(0, num_gt - num_pred)

    cost_matrix = np.zeros((num_pred + pad, num_gt))

    for i, pr in enumerate(pred_rels):
        for j, gt in enumerate(gt_rels):
            subj_sim = sem_sim(pr["subject"], gt["subject"])
            obj_sim = sem_sim(pr["object"], gt["object"])
            pred_sim = sem_sim(pr["predicate"], gt["predicate"])
            weighted_sim = (
                0.3 * subj_sim +
                0.3 * obj_sim +
                0.4 * pred_sim
            )
            cost_matrix[i, j] = 1.0 - weighted_sim

    if pad:
        cost_matrix[num_pred:, :] = 1e5  # High cost for dummy preds

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < num_pred:  # Ignore dummy preds
            matches.append({
                "groundtruth": gt_rels[c],
                "prediction": pred_rels[r],
                "cost": cost_matrix[r, c],
                "similarity": 1.0 - cost_matrix[r, c],
            })
    return matches

def spatial_reward(pred_scene: dict, gt_scene: dict, w: int, h: int) -> tuple[float, float]:
    if not isinstance(pred_scene, dict) or not isinstance(gt_scene, dict):
        return 0.0, 0.0

    gt_objs = gt_scene.get("objects") or []
    pr_objs = pred_scene.get("objects") or []
    gt_rels = gt_scene.get("relationships") or []
    pr_rels = pred_scene.get("relationships") or []
    
    # SAFETY CHECKS 
    if not isinstance(pr_objs, list) or not isinstance(gt_objs, list):
        return 0.0, 0.0
    if not isinstance(pr_rels, list) or not isinstance(gt_rels, list):
        return 0.0, 0.0

    if not all(is_valid_object(o) for o in pr_objs):
        return 0.0, 0.0
    if not all(is_valid_relation(r) for r in pr_rels):
        return 0.0, 0.0

    #### preprocess predictions ####
    gt_objs = [
        {
            **o,
            "id": refine_node_edge(o["id"]),
            "bbox": scale_box(o["bbox"], (1.0 / w, 1.0 / h))
        }
        for o in gt_objs
    ]
    pr_objs = [
        {
            **o,
            "id": refine_node_edge(o["id"]),
            "bbox": scale_box(o["bbox"], (1.0 / w, 1.0 / h))
        }
        for o in pr_objs
    ]

    gt_triplets = [
        {
            **r,
            "subject": refine_node_edge(r["subject"]),
            "object": refine_node_edge(r["object"])
        }
        for r in gt_rels
    ]

    pr_triplets = [
        {
            **r,
            "subject": refine_node_edge(r["subject"]),
            "object": refine_node_edge(r["object"])
        }
        for r in pr_rels
    ]

    if not gt_objs:
        obj_score = 1.0 if not pr_objs else 0.0
    else:
        assign = bi_match(gt_objs, pr_objs)
        per_gt_box = []
        per_gt_id_sim = []
        for g_idx, p_idx in enumerate(assign):
            if p_idx is None: # unmatched GT
                per_gt_box.append(0.0)
                per_gt_id_sim.append(0.0)
                continue
            g, p = gt_objs[g_idx], pr_objs[p_idx]
            sim = sem_sim(g["id"], p["id"])
            iou = compute_iou(g["bbox"], p["bbox"])
            l1 = math.exp(-box_L1(g["bbox"], p["bbox"])) # 0-1
            obj_score = ((IOU_W * iou) + (L1_W * l1)) / (IOU_W + L1_W)
            # obj_score = iou
            per_gt_box.append(obj_score)
            per_gt_id_sim.append(sim)
        obj_box_score = sum(per_gt_box) / len(gt_objs) if gt_objs else 1.0
        obj_id_sim_score = sum(per_gt_id_sim) / len(gt_objs) if gt_objs else 1.0
        obj_score = 0.5 * obj_box_score + 0.5 * obj_id_sim_score

        # print(f"[DEBUG] assign: {assign}, obj_box_score: {obj_box_score}, obj_id_sim_score: {obj_id_sim_score}, obj_score: {obj_score}")

    #### Relation score - Edge Matching ####
    if not gt_rels:
        rel_score = 1.0 if not pr_rels else 0.0
    else:
        # Run semantic triplet-level Hungarian matching
        matches = bi_match_triplets(gt_triplets, pr_triplets)
        # print(f'[DEBUG] matches: {matches}')
        
        # Compute soft similarity score for each matched pair
        scores = [1.0 - m["cost"] for m in matches]

        # Final score: average similarity over all GT relations
        rel_score = sum(scores) / len(gt_triplets) if gt_triplets else 1.0

        # # Map GT id → matched pred id
        # id_map = {}
        # for g_idx, p_idx in enumerate(assign): # map gt id's to pred id's based on hungarian bbox + semantic match
        #     if p_idx is not None: # matched GT
        #         id_map[gt_objs[g_idx]["id"]] = pr_objs[p_idx]["id"]

        # # Pred triplet index keyed by (subj, obj)
        # pred_by_pair = {}
        # for p in pr_rels:
        #     pred_by_pair[(refine_node_edge(p["subject"]),
        #                     refine_node_edge(p["object"]))] = p["predicate"]

        # # try exact-matched triplet alignment first
        # matched_gts = set()
        # scores = []

        # for i, g in enumerate(gt_rels):
        #     gs, go, gp = g["subject"], g["object"], g["predicate"]

        #     if gs in id_map and go in id_map:
        #         ps, po = id_map[gs], id_map[go]
        #         pp = pred_by_pair.get((refine_node_edge(ps), refine_node_edge(po)))
        #         if pp is not None: # found valid triplet via object matching (bbox + sem)
        #             s_score = sem_sim(gs, ps)
        #             o_score = sem_sim(go, po)
        #             p_score = sem_sim(gp, pp)
        #             scores.append(s_score * o_score * p_score)
        #             matched_gts.add(i)
        #             continue

        # unmatched_gt_rels = [gt_rels[i] for i in range(len(gt_rels)) if i not in matched_gts]

        # # fallback to allow soft relation matching: semantic triplet matching
        # if unmatched_gt_rels:
        #     fallback_matches = bi_match_triplets(unmatched_gt_rels, pr_rels)
        #     for m in fallback_matches:
        #         score = 1.0 - m["cost"] # soft similarity
        #         print(f'score: {score}')
        #         scores.append(score)

        # print(f'scores: {scores}')

        # rel_score = sum(scores) / len(gt_rels) if len(gt_rels) else 1.0

    #### Combine ####
    return obj_score, rel_score

def compute_obj_score(gt_objs: list, pr_objs: list) -> float:
    assign = bi_match(gt_objs, pr_objs)
    per_gt_box = []
    per_gt_id_sim = []
    for g_idx, p_idx in enumerate(assign):
        if p_idx is None: # unmatched GT
            per_gt_box.append(0.0)
            per_gt_id_sim.append(0.0)
            continue
        g, p = gt_objs[g_idx], pr_objs[p_idx]
        iou = compute_ciou(g["bbox"], p["bbox"])
        sim = sem_sim(g["id"], p["id"])
        # l1 = math.exp(-box_L1(g["bbox"], p["bbox"])) # 0-1
        # obj_score = ((IOU_W * iou) + (L1_W * l1)) / (IOU_W + L1_W)
        obj_score = iou
        per_gt_box.append(obj_score) 
        per_gt_id_sim.append(sim)
    obj_box_score = sum(per_gt_box) / len(gt_objs) if gt_objs else 1.0
    obj_id_sim_score = sum(per_gt_id_sim) / len(gt_objs) if gt_objs else 1.0
    # print(f"[DEBUG] assign: {assign}, per_gt_box: {per_gt_box}, obj_score: {obj_score}, pr_objs: {pr_objs}, gt_objs: {gt_objs}")
    
    # obj_score = 0.7 * obj_box_score + 0.3 * obj_id_sim_score
    obj_score = obj_box_score
    
    return obj_score

def compute_rel_score(gt_rels: list, pr_rels: list) -> float:
    matches = bi_match_triplets(gt_rels, pr_rels)
    scores = [1.0 - m["cost"] for m in matches]
    rel_score = sum(scores) / len(gt_rels) if gt_rels else 1.0
    return rel_score

def relaxed_spatial_reward(pred_scene: dict, gt_scene: dict, w: int, h: int, threshold: float = 0.0, rel_gating: bool = False) -> float:
    if not isinstance(pred_scene, dict) or not isinstance(gt_scene, dict):
        return 0.0

    gt_objs = gt_scene.get("objects") or []
    pr_objs = pred_scene.get("objects") or []
    gt_rels = gt_scene.get("relationships") or []
    pr_rels = pred_scene.get("relationships") or []
    
    # SAFETY CHECKS 
    if not isinstance(pr_objs, list) or not isinstance(gt_objs, list):
        return 0.0
    if not isinstance(pr_rels, list) or not isinstance(gt_rels, list):
        return 0.0

    if not all(is_valid_object(o) for o in pr_objs):
        return 0.0
    if not all(is_valid_relation(r) for r in pr_rels):
        return 0.0

    #### preprocess predictions ####
    gt_objs = [
        {
            **o,
            "id": refine_node_edge(o["id"]),
            "bbox": scale_box(o["bbox"], (1.0 / w, 1.0 / h))
        }
        for o in gt_objs
    ]
    pr_objs = [
        {
            **o,
            "id": refine_node_edge(o["id"]),
            "bbox": scale_box(o["bbox"], (1.0 / w, 1.0 / h))
        }
        for o in pr_objs
    ]

    gt_triplets = [
        {
            **r,
            "subject": refine_node_edge(r["subject"]),
            "object": refine_node_edge(r["object"])
        }
        for r in gt_rels
    ]

    pr_triplets = [
        {
            **r,
            "subject": refine_node_edge(r["subject"]),
            "object": refine_node_edge(r["object"])
        }
        for r in pr_rels
    ]
    
    if not gt_rels: # if there are no GT triplets, simply return matched IoU
        if not gt_objs:
            obj_score = 1.0 if not pr_objs else 0.0
        else:
            obj_score = compute_obj_score(gt_objs, pr_objs)
            
    else: # if there are GT triplets, check if the correct triplet is generated or not, if it is then return IoU, else 0
        # run semantic triplet-level Hungarian matching
        matches = bi_match_triplets(gt_triplets, pr_triplets)
        
        # compute soft similarity score for each matched pair
        scores = [1.0 - m["cost"] for m in matches]
        obj_score = compute_obj_score(gt_objs, pr_objs)
        
        if matches:
            avg_similarity = sum(m['similarity'] for m in matches) / len(matches)
            obj_score = obj_score
            # if avg_similarity >= threshold:
            #     obj_score = compute_obj_score(gt_objs, pr_objs)
        else:
            obj_score = 0.0 if rel_gating else obj_score
            # print(f'[DEBUG] no relation triplet matched, matches: {matches}, pred_rels: {pr_rels}, gt_rels: {gt_rels}')
            
    return obj_score   
        

REQUIRED_KEYS_OBJ = {"id", "bbox"}
REQUIRED_KEYS_REL = {"subject", "predicate", "object"}

def is_valid_id_format(s):
    return bool(re.fullmatch(r"[a-zA-Z_]+\.\d+", s))

def is_valid_object(obj: Dict) -> bool:
    if not isinstance(obj, dict):
        return False
    if not REQUIRED_KEYS_OBJ.issubset(obj.keys()):
        return False
    # if any other key than required_keys is present
    if not all(key in REQUIRED_KEYS_OBJ for key in obj.keys()):
        return False
    if not isinstance(obj["id"], str):
        return False
    # check if format of id is valid i.e [str].[number]
    if not is_valid_id_format(obj["id"]):
        return False
    if not isinstance(obj["bbox"], list):
        return False
    if not len(obj["bbox"]) == 4:
        return False
    if not all(isinstance(val, (int, float)) for val in obj["bbox"]):
        return False
    return True

def is_valid_relation(rel: Dict) -> bool:
    if not isinstance(rel, dict):
        return False
    if not REQUIRED_KEYS_REL.issubset(rel.keys()):
        return False
    if not isinstance(rel["subject"], str):
        return False
    if not isinstance(rel["predicate"], str):
        return False
    if not isinstance(rel["object"], str):
        return False
    # check if subject is valid i.e [str].[number]
    if not is_valid_id_format(rel["subject"]):
        return False
    # check if object is valid i.e [str].[number]
    if not is_valid_id_format(rel["object"]):
        return False
    return True

def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_scene(text: str) -> List[Dict]:
    match = re.search(r"<scene>(.*?)</scene>", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1).strip())
        return parsed if isinstance(parsed, dict) else {}
    except:
        return {}

def format_reward(text: str) -> float:
    try:
        observe = re.search(r"<observe>.*?</observe>", text, re.DOTALL)
        think = re.search(r"<think>.*?</think>", text, re.DOTALL)
        scene = re.search(r"<scene>(.*?)</scene>", text, re.DOTALL)
        answer = re.search(r"<answer>.*?</answer>", text, re.DOTALL)

        if not (observe and think and scene and answer):
            return 0.0
        
        # check for counts of tags: think, scene, think, answser
        observe_count = text.count("<observe>")
        think_count = text.count("<think>")
        scene_count = text.count("<scene>")
        answer_count = text.count("<answer>")
        
        if think_count != 1 or scene_count != 1 or answer_count != 1 or observe_count != 1:
            return 0.0
        
        scene = extract_scene(text)
        
        if not scene or not isinstance(scene, dict):
            return 0.0
        
        objs, rels = scene.get("objects", []), scene.get("relationships", [])

        if not isinstance(objs, list) or not isinstance(rels, list):
            return 0.0
        
        if not all(is_valid_object(o) for o in objs):
            return 0.0
        
        if not all(is_valid_relation(r) for r in rels):
            return 0.0
        
        # repeated-ID penalty
        ids = [o.get("id", "") for o in objs]
        if len(ids) != len(set(ids)):
            return 0.0
        
        return 1.0
    except:
        return 0.0
    
def acc_reward(pred: str, gt: str) -> float:
    return float(pred.strip().lower() == gt.strip().lower())

def count_reward(pred_scene, gt_scene) -> float:
    # return 0 reward if predicted scene is not a valid json! (format issue - hence, 0 for counting reward)
    if not isinstance(pred_scene, dict) or not isinstance(gt_scene, dict):
        return 0.0
    
    pred_objs = pred_scene.get("objects")
    gt_objs = gt_scene.get("objects")
    
    pred_rels = pred_scene.get("relationships") or []
    gt_rels = gt_scene.get("relationships") or []
    
    if not isinstance(pred_objs, list) or not isinstance(gt_objs, list):
        return 0.0
    
    obj_count_reward = max(0.0, 1 - abs(len(pred_objs) - len(gt_objs)) / max(len(gt_objs), 1))
    
    if not len(gt_rels): 
        final_count_reward = obj_count_reward
    else:
        rel_count_reward = max(0.0, 1 - abs(len(pred_rels) - len(gt_rels)) / max(len(gt_rels), 1))
        final_count_reward = (obj_count_reward * 0.7) + (rel_count_reward * 0.3)
    
    return final_count_reward
    # return obj_count_reward

def extract_image_size(problem: str) -> tuple[int, int]:
    match = re.search(r"Image size: \((.*?) x (.*?)\)", problem)
    if not match:
        raise ValueError("Image size not found in problem!!! Required for spatial_sgg reward scoring.")
    width = int(match.group(1))
    height = int(match.group(2))
    return width, height

def spatial_sgg_compute_score(predict_str: str, ground_truth_str: str, problem: str) -> Dict[str, float]:
    pred_answer = extract_answer(predict_str)
    gt_answer = extract_answer(ground_truth_str)
    
    pred_scene = extract_scene(predict_str)
    gt_scene = extract_scene(ground_truth_str)
    
    image_width, image_height = extract_image_size(problem)
        
    FORMAT_WEIGHT = 0.1
    COUNT_WEIGHT = 0.2
    ACCURACY_WEIGHT = 0.5
    SPATIAL_WEIGHT = 0.2
    
    THRESHOLD = 0.0
    REL_GATING = False
                
    fr = format_reward(predict_str)
    
    if fr == 1.0:
        cr = count_reward(pred_scene, gt_scene)
        ar = acc_reward(pred_answer, gt_answer)
        
        obj_score = 0.0
        if ar == 1.0:
            obj_score = relaxed_spatial_reward(pred_scene, gt_scene, image_width, image_height, THRESHOLD, REL_GATING)
    else:
        cr = 0.0
        ar = 0.0
        obj_score = 0.0
    
    
    total_reward = (
        fr * FORMAT_WEIGHT + 
        cr * COUNT_WEIGHT + 
        ar * ACCURACY_WEIGHT + 
        obj_score * SPATIAL_WEIGHT
    )
    
    # print(f"[DEBUG] pred_str: {predict_str} \n\n gt_str: {ground_truth_str} \n obj_score: {obj_score}, format: {fr}, count: {cr}, ar: {ar}, total_reward: {total_reward} \n")

    return {
        "overall": total_reward,
        "format": fr,
        "count": cr,
        "accuracy": ar,
        "spatial_score": obj_score
    }
