import os.path as osp
import os
import numpy as np
from tqdm import tqdm
import pickle
import sys
sys.path.append("/home/data_sata/vlmloc/vlm-loc")
import random
random.seed(42)
import json
from typing import Any, Dict, List, Optional

from utils import project_points_to_pixels
from datapreparation.kitti360pose.utils import (
    SCENE_NAMES_TRAIN,
    SCENE_NAMES_VAL,
    SCENE_NAMES_TEST,
    STUFF_CLASSES,
)
from datapreparation.args import parse_arguments

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    

def create_train_info(scene_info):
    image_path = scene_info["image_path"]
    user_message = scene_info["user_message"]
    assistant_message = scene_info["assistant_message"]
    scene_graph_message = scene_info["scene_graph_message"]

    user_content = f"<image> {user_message}"
    # user_content = user_message
    assistant_content = assistant_message

    return {
        "messages": [
            {"role": "user", "content": scene_graph_message},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [image_path],
    }


def create_scene_graph(
    centers_info: List[Optional[List[Dict[str, Any]]]],
) -> Dict[str, Any]:

    # Collect raw nodes
    raw_nodes: List[Dict[str, Any]] = []
    node_world: List[np.ndarray] = []
    node_id = 0

    items = centers_info

    for it in items:
        lbl = str(it.get("label", "")).lower()
        pc = it.get("pixel_center")
        wc = it.get("world_center")

        px, py = int(pc[0]), int(pc[1])
        wx, wy = float(wc[0]), float(wc[1])

        raw_nodes.append({
            "node_id": node_id,  # Raw index reference only
            "label": lbl,
            "pixel_center": [px, py],  # Single-pixel coordinate
            "world_center": [wx, wy],  # Keep original precision
        })
        node_world.append(np.array([wx, wy], dtype=float))
        node_id += 1

    return {"nodes": raw_nodes}

def ground_pose_to_image_scene_graph(pose, scene_graph, objects_in_cell, object_instance_ids) -> List[Dict[str, Any]]:
    """
    Matching rule:
      1) Only match within nodes with the same semantic label.
      2) If the semantic label does not match, it is unmatched directly.
      3) If the label matches, iterate over all candidate nodes of that label,
         compute distances from all object points to each node, and select
         the nearest node if it is within the threshold.
    """
    grounded_info: List[Dict[str, Any]] = []

    # Preprocess scene-graph nodes
    pre_nodes = []
    for nd in scene_graph.get("nodes", []):
        lbl = str(nd.get("label", "")).lower()
        wc = nd.get("world_center", None)
        if wc is None or len(wc) < 2:
            continue
        wx, wy = float(wc[0]), float(wc[1])
        pre_nodes.append({
            "node_id": nd.get("node_id"),
            "label": lbl,
            "pixel_center": nd.get("pixel_center"),
            "world_center": np.array([wx, wy], dtype=float),
        })

    # label -> list of nodes
    from collections import defaultdict
    label2nodes = defaultdict(list)
    for nd in pre_nodes:
        label2nodes[nd["label"]].append(nd)

    # Iterate over each described object
    for desc in getattr(pose, "descriptions", []):
        label_raw = getattr(desc, "object_label", "")
        label = str(label_raw).lower()
        instance_id = getattr(desc, "object_instance_id", None)

        matched = {
            "object_label": label_raw,
            "grounded": False,
            "matched_node": None,
        }

        # Object not found
        if instance_id not in object_instance_ids:
            grounded_info.append(matched)
            continue

        # Get the point set of the current object
        obj = objects_in_cell[object_instance_ids.index(instance_id)]
        obj_xyz = obj[0].xyz[obj[1]]
        if obj_xyz.ndim != 2 or obj_xyz.shape[1] < 2 or len(obj_xyz) == 0:
            grounded_info.append(matched)
            continue
        obj_xy = obj_xyz[:, :2]

        # Semantic match
        cand_nodes = label2nodes.get(label, [])
        if not cand_nodes:
            grounded_info.append(matched)
            continue

        # Distance computation
        best_dist, best_nd = float("inf"), None
        for nd in cand_nodes:
            wc = nd["world_center"]
            # Compute distances from all object points to the node and keep the minimum
            dists = np.linalg.norm(obj_xy - wc, axis=1)
            min_dist = float(np.min(dists))
            if min_dist < best_dist:
                best_dist = min_dist
                best_nd = nd

        # Threshold
        thr = 1.0 if label in STUFF_CLASSES else 1.0
        if best_dist <= thr:
            matched["grounded"] = True
            matched["matched_node"] = {
                "node_id": best_nd["node_id"],
                "label": best_nd["label"],
                "pixel_center": best_nd["pixel_center"],
                "world_center": best_nd["world_center"].tolist(),
                "distance_m": best_dist,
                "threshold_m": thr,
            }

        grounded_info.append(matched)

    return grounded_info

def ground_pose_to_image_scene_graph_v2(pose, scene_graph, objects_in_cell, object_instance_ids) -> List[Dict[str, Any]]:
    """
    Matching rule:
      1) Only match within nodes with the same semantic label.
      2) If the semantic label does not match, it is unmatched directly.
      3) If the label matches, iterate over all candidate nodes of that label,
         compute distances, and select the nearest node if it is within the threshold.
    """
    grounded_info: List[Dict[str, Any]] = []

    # Preprocess scene-graph nodes
    pre_nodes = []
    for nd in scene_graph.get("nodes", []):
        lbl = str(nd.get("label", "")).lower()
        wc = nd.get("world_center", None)
        if wc is None or len(wc) < 2:
            continue
        wx, wy = float(wc[0]), float(wc[1])
        pre_nodes.append({
            "node_id": nd.get("node_id"),
            "label": lbl,
            "pixel_center": nd.get("pixel_center"),
            "world_center": np.array([wx, wy], dtype=float),
        })

    # label -> list of nodes
    from collections import defaultdict
    label2nodes = defaultdict(list)
    for nd in pre_nodes:
        label2nodes[nd["label"]].append(nd)

    # Iterate over each described object
    for desc in getattr(pose, "descriptions", []):
        label_raw = getattr(desc, "object_label", "")
        label = str(label_raw).lower()
        instance_id = getattr(desc, "object_instance_id", None)

        matched = {
            "object_label": label_raw,
            "grounded": False,
            "matched_node": None,
        }

        # Object not found
        if instance_id not in object_instance_ids:
            grounded_info.append(matched)
            continue

        # Get the current pose-object center
        pose_obj_center_incell = desc.object_center[0:2]
        pose_obj_center_world = pose_obj_center_incell * 50 + np.array(pose.pose_w[0:2]) - 25.0

        # Semantic match
        cand_nodes = label2nodes.get(label, [])
        if not cand_nodes:
            grounded_info.append(matched)
            continue

        # Distance computation
        dists = [float(np.hypot(*(nd["world_center"] - pose_obj_center_world))) for nd in cand_nodes]
        best_i = int(np.argmin(dists))
        best_nd = cand_nodes[best_i]
        best_dist = dists[best_i]

        # Threshold setup
        thr = 15.0 if label in STUFF_CLASSES else 5.0
        if label == 'road':
            thr = 50.0

        # Match decision
        if best_dist <= thr:
            matched["grounded"] = True
            matched["matched_node"] = {
                "node_id": best_nd["node_id"],
                "label": best_nd["label"],
                "pixel_center": best_nd["pixel_center"],
                "world_center": best_nd["world_center"].tolist(),
                "distance_m": best_dist,
                "threshold_m": thr,
            }

        grounded_info.append(matched)
    return grounded_info
def create_text_image_pairs(
    poses,
    cells, 
    objects,
    scene,
    centers_info,
    BEV_RANGE,
    image_paths
):

    def _join_phrases_with_and(phrases):
        """Join phrases with commas and 'and': a, b, and c."""
        n = len(phrases)
        if n == 0:
            return ""
        if n == 1:
            return phrases[0]
        if n == 2:
            return f"{phrases[0]} and {phrases[1]}"
        return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    scene_info = []
    cell_ids = [cell.id for cell in cells]
    
    for pid, pose in tqdm(enumerate(poses), desc=f"Creating text-video pairs for scene {scene}", total=len(poses)):

        eval_cell_id = pose.eval_cell_id
        descriptions = pose.descriptions

        idx = cell_ids.index(eval_cell_id)
        cell = cells[idx]
        cell_image_path = image_paths[idx]
        # change path
        cell_image_path = '/data/kang/vlmloc/' + cell_image_path
        
        center_info = centers_info[idx]
        scene_graph = create_scene_graph(center_info)

        bbox = cell.bbox_w
        objects_in_cell = []
        for obj in objects:
            obj_xyz = obj.xyz
            mask = (
                (obj_xyz[:, 0] >= bbox[0]) & (obj_xyz[:, 0] <= bbox[3]) &
                (obj_xyz[:, 1] >= bbox[1]) & (obj_xyz[:, 1] <= bbox[4]) &
                (obj_xyz[:, 2] >= bbox[2]) & (obj_xyz[:, 2] <= bbox[5])
            )
            if np.sum(mask) > 0:
                objects_in_cell.append((obj, mask))

        object_instance_ids = [obj.instance_id for obj, _ in objects_in_cell]

        pose_ground_pairs = ground_pose_to_image_scene_graph_v2(pose, scene_graph, objects_in_cell, object_instance_ids)

        assert len(pose_ground_pairs) == len(pose.descriptions) == 6
        phrases = []
        assign_pairs = []
        for i, hint in enumerate(poses[pid].descriptions):
            ground_info = pose_ground_pairs[i]
            assign_pairs.append({
                "object_label": hint.object_label,
                "grounded": ground_info["grounded"],
                "matched_node": ground_info["matched_node"]['node_id'] if ground_info["matched_node"] else None
            })
            phrases.append(f"{hint.direction} of a {hint.object_color_text} {hint.object_label}")

        natural_clause = _join_phrases_with_and(phrases)
        query_description = f"The target location is {natural_clause}."

        user_message = " "
        if query_description:
            user_message += query_description

        pose_image_path = cell_image_path

        x_min, y_min = bbox[0], bbox[1]
        x_scale = (IMAGE_SIZE) / BEV_RANGE
        y_scale = (IMAGE_SIZE) / BEV_RANGE

        pose_world_coords = np.asarray(pose.pose_w[0:2], dtype=float).reshape(1, 2)

        x_img, y_img, idx = project_points_to_pixels(
            pose_world_coords,
            x_min, y_min,
            x_scale, y_scale,
            img_size_minus_1=IMAGE_SIZE - 1
        )
        assert 0 <= x_img <= IMAGE_SIZE and 0 <= y_img <= IMAGE_SIZE
        pose_pixel_coor = [int(x_img[0]), int(y_img[0])]

        assistant_message_obj = {
            "assignments": assign_pairs,
            "point_2d": pose_pixel_coor,
        }

        assistant_message = json.dumps(assistant_message_obj, ensure_ascii=False)

        for node in scene_graph['nodes']:
            if "world_center" in node:
                del node["world_center"]
        scene_graph_message = json.dumps(scene_graph, ensure_ascii=False)

        scene_info.append({
            "pid": pid,
            "scene": scene,
            "user_message": user_message,
            "assistant_message": assistant_message, 
            "scene_graph_message": scene_graph_message,
            "image_path": pose_image_path,
        })

    return scene_info


def process_single_scene(scene, args, BEV_RANGE, IMAGE_SIZE, OUTPUT_DIR, DATA_DIR):
    """Process a single scene for generation."""
    print(f"\nProcessing scene: {scene}")
    # path_poses = osp.join(args.path_out, "poses", f"{scene}.pkl")
    if not osp.exists(osp.join(DATA_DIR, "poses", f"{scene}.pkl")):
        print(f"  Skipping scene {scene}: poses file not found.")
        return []
    poses = pickle.load(open(osp.join(DATA_DIR, "poses", f"{scene}.pkl"), "rb"))
    cells = pickle.load(open(osp.join(DATA_DIR, "cells", f"{scene}.pkl"), "rb"))
    objects = pickle.load(open(osp.join(DATA_DIR, "objects", f"{scene}.pkl"), "rb"))
    save_data = pickle.load(open(osp.join(DATA_DIR, f"{scene}_centers_info.pkl"), "rb"))
    print(f"Loaded centers info for scene {scene}")
    centers_info = save_data["centers_info"]
    # image_center = save_data["image_center"]
    image_paths = save_data["image_paths"]
    
    assert len(centers_info) == len(cells) == len(image_paths)
    scene_data = create_text_image_pairs(poses, cells, objects, scene, centers_info, BEV_RANGE, image_paths)
    
    print(f"Scene {scene} completed: {len(scene_data)} samples generated")

    # scene_data = []
    return scene_data
        

if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"))
    print()
    set_global_seed(42)


    SCENE_NAMES_TEST = [
        "birmingham_block_0_inst",
        "birmingham_block_5_inst",
        "birmingham_block_6_inst",
        "birmingham_block_12_inst",
        "cambridge_block_2_inst",
        "cambridge_block_3_inst",
        "cambridge_block_8_inst",
        "cambridge_block_10_inst",
        "cambridge_block_14_inst",
        "cambridge_block_21_inst",
        "cambridge_block_26_inst",
        # "cambridge_block_32_inst",
    ]


    GENERATION_SPLIT = SCENE_NAMES_TEST

    if GENERATION_SPLIT == SCENE_NAMES_TRAIN:
        save_suffix = "training"
    elif GENERATION_SPLIT == SCENE_NAMES_VAL:
        save_suffix = "val"
    elif GENERATION_SPLIT == SCENE_NAMES_TEST:
        save_suffix = "testing"

    BEV_RANGE = 50.0  # meters (50m x 50m around center)
    IMAGE_SIZE = 224
    SPATIAL_RESOLUTION = BEV_RANGE/IMAGE_SIZE
    
    # DATA_DIR = '/data/kang/vlmloc/data/k360_50-10_gridCells_pd10_pc2_shiftPoses_all_nm-6'
    DATA_DIR = '/home/data_sata/vlmloc/CityRefer/data/sensaturban/cityrefer_data_0124'
    # Output parameters
    output_version = "vlmloc_data_cityrefer_0124"
    OUTPUT_DIR = os.path.join("/home/data_sata/vlmloc/vlmloc", output_version)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Configuration:")
    print(f"  BEV Range: {BEV_RANGE}m x {BEV_RANGE}m")
    print(f"  Image Size: {IMAGE_SIZE} x {IMAGE_SIZE} pixels")
    print(f"  Spatial Resolution: {SPATIAL_RESOLUTION}m/pixel")
    print(f"  DATA Directory: {DATA_DIR}")
    print(f"  Output Directory: {OUTPUT_DIR}")

    all_scene_data = []
    for scene in GENERATION_SPLIT:

        scene_data = process_single_scene(
            scene, args, BEV_RANGE, IMAGE_SIZE, OUTPUT_DIR, DATA_DIR
        )
        all_scene_data.extend(scene_data)
        print(f"✓ Scene {scene} data collected: {len(scene_data)} samples")

    
    test_data = []
    detailed_data = []

    for item in all_scene_data:
        train_sample = create_train_info(item)
        test_data.append(train_sample)
    
    training_json_path = osp.join(OUTPUT_DIR, "vlmloc_"+save_suffix+"_data.json")
    with open(training_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(test_data)} training samples to {training_json_path}")





