# Referred from https://github.com/meidachen/STPLS3D/blob/main/HAIS/data/prepare_data_inst_instance_stpls3d.py
import glob
import json
import re
import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tool import DataProcessing

np.random.seed(3824)

def getFiles(files, fileSplit):
    res = []
    for filePath in files:
        name = os.path.basename(filePath)
        if name.strip('.ply') in fileSplit:
            res.append(filePath)
    return res

def _read_json(path):
    with open(path) as f:
        file = json.load(f)
    return file

def get_false_segments(false_seg_dir):
    false_segs = []
    print('Read false segments')
    for false_seg_file in glob.glob(os.path.join(false_seg_dir, '*.csv')):
        print('loading:', false_seg_file)
        false_seg_df = pd.read_csv(false_seg_file)
        false_segs.append(false_seg_df)
    print()

    false_seg_df = pd.concat(false_segs)

    false_object_ids = set()
    for _, row in false_seg_df.iterrows():
        false_object_id = row.area + '_block_' + str(row.block_id) + '_' + str(row.object_id)
        false_object_ids.add(false_object_id)
    return false_object_ids

def preparePthFiles(args, files, split, outPutFolder, false_segments):
    # Keep all semantic classes, all instances, and all points.

    semantic_names = {
        0: 'Ground',
        1: 'High Vegetation',
        2: 'Buildings',
        3: 'Walls',
        4: 'Bridge',
        5: 'Parking',
        6: 'Rail',
        7: 'traffic Roads',
        8: 'Street Furniture',
        9: 'Cars',
        10: 'Footpath',
        11: 'Bikes',
        12: 'Water'
    }

    skipped_by_semantic = {}  # Count skipped instances by semantic class
    false_segments_count = 0  # Count skipped false segments

    print(split)
    for file in tqdm(files, total=len(files)):
        print('loading:', file)
        seg_file = re.sub('.ply', '.segs.json', file)
        scene_id = os.path.basename(seg_file).rstrip('.segs.json')

        # Read the PLY file.
        xyz, rgb, labels = DataProcessing.read_ply_data(file)

        # Add instance labels.
        instance_db = _read_json(seg_file)
        labels = labels[:, np.newaxis]
        empty_instance_label = np.full(labels.shape, -100)
        labels = np.hstack((labels, empty_instance_label))

        # Collect all instances for this scene.
        scene_instances = []

        # Process each instance.
        for instance in instance_db["segGroups"]:
            occupied_indices = np.array(instance['pointIds'])

            # Filter invalid point indices.
            valid_indices = occupied_indices[occupied_indices < len(labels)]
            if len(valid_indices) == 0:
                continue

            occupied_indices = valid_indices

            if scene_id + "_" + str(instance["id"]) in false_segments:
                false_segments_count += 1
                continue

            # Get points for this instance.
            instance_xyz = xyz[occupied_indices]
            instance_rgb = rgb[occupied_indices]
            instance_semantic = labels[occupied_indices, 0]
            instance_id = int(instance["id"])

            # Use majority voting for the semantic label of this instance.
            unique_semantics, counts = np.unique(instance_semantic, return_counts=True)
            main_semantic = unique_semantics[np.argmax(counts)]

            # Filter out Cars (9) and Bikes (11).
            if main_semantic in [9, 11]:
                print(
                    f"Filtering out instance {instance_id} with semantic {main_semantic} "
                    f"({semantic_names.get(main_semantic, 'Unknown')})"
                )
                continue

            if instance_xyz.shape[0] < 500:
                # Count skipped instances by semantic class.
                if main_semantic not in skipped_by_semantic:
                    skipped_by_semantic[main_semantic] = 0
                skipped_by_semantic[main_semantic] += 1
                continue

            # Create the instance label array.
            instance_label = np.full(len(occupied_indices), instance_id)

            # Normalize colors to the [-1, 1] range.
            instance_colors = instance_rgb.astype(np.float32) / 127.5 - 1

            # Convert to the expected data types.
            coords = np.float32(instance_xyz)
            colors = np.float32(instance_colors)
            sem_labels = instance_semantic.astype(np.int32)
            inst_labels = instance_label.astype(np.int32)

            # Store instance data.
            instance_data = {
                'coords': coords,
                'colors': colors,
                'sem_labels': sem_labels,
                'inst_labels': inst_labels,
                'instance_id': instance_id
            }
            scene_instances.append(instance_data)

        # Save all instances of this scene into one file.
        if scene_instances:
            scene_name = f"{scene_id}"
            outFilePath = os.path.join(outPutFolder, scene_name + '_inst.pth')

            # Compute the total number of points.
            total_points = sum(len(instance['coords']) for instance in scene_instances)

            print(f'saving scene: {scene_name}, total instances: {len(scene_instances)}, total points: {total_points}')

            # Save processed data.
            torch.save(scene_instances, outFilePath)
        else:
            print(f'scene: {scene_id} - No valid instances (all filtered out)')

        print()

    print(f'False segments skipped: {false_segments_count}')

    # Report skipped instances by semantic class.
    if skipped_by_semantic:
        print('\n=== Skipped instances by semantic class (points < 500) ===')
        total_skipped_instances = 0
        for semantic_id in sorted(skipped_by_semantic.keys()):
            semantic_name = semantic_names.get(semantic_id, f"Unknown_{semantic_id}")
            count = skipped_by_semantic[semantic_id]
            total_skipped_instances += count
            print(f"  {semantic_id:2d}: {semantic_name:<20} - {count:4d} instances")
        print(f"\nTotal skipped instances: {total_skipped_instances}")
    else:
        print('\nNo instances were skipped due to insufficient points')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="scans")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--split_data", type=str, default="meta_data/balance_split")
    parser.add_argument("--false_seg_dir", type=str, default="meta_data/false_segments")
    parser.add_argument("--sample_type", type=str, default="random")
    parser.add_argument("--grid_size", type=float, default=0.2)
    parser.add_argument("--random_sample_ratio", type=int, default=10)
    parser.add_argument("--train_crop_size", type=int, default=50)
    parser.add_argument("--val_crop_size", type=int, default=250)
    parser.add_argument("--aug_times", type=int, default=6)
    args = parser.parse_args()

    filesOri = sorted(glob.glob(args.data_dir + '/*/*.ply'))
    split_dir = os.path.basename(args.split_data)

    out_dir = os.path.join(args.out_dir, split_dir)

    false_segments = get_false_segments(args.false_seg_dir)

    # Validation split with validation crop.
    testSplit = [line.strip() for line in open(os.path.join(args.split_data, 'sensaturban_val_test.txt')).readlines()]
    split = 'test'
    valFiles = getFiles(filesOri, testSplit)
    valOutDir = os.path.join(out_dir, split)
    print(valOutDir)
    os.makedirs(valOutDir, exist_ok=True)
    preparePthFiles(args, valFiles, split, valOutDir, false_segments)

