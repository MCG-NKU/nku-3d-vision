import sys 
sys.path.append('/home/data_sata/vlmloc/Text2Loc')
from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL
import pickle
import os
import os.path as osp
import json
city_refer_seq = sorted(os.listdir("/home/data_sata/vlmloc/CityRefer/data/sensaturban/cityrefer_data_0124/cells"))

# base_path = "/home/data_sata/vlmloc/Text2Loc/data/k360_50-10_gridCells_pd10_pc2_shiftPoses_all_nm-6_v5_random"
base_path = '/home/data_sata/vlmloc/CityRefer/data/sensaturban/cityrefer_data_0124'
output_dir = osp.join(base_path, "direction")
if not osp.exists(output_dir):
    os.mkdir(output_dir)

def check_neighbor(cell_x, cell_y, neigh_x, neigh_y):
    return abs(cell_x - neigh_x) <= 10 and abs(cell_y - neigh_y) <= 10

def get_direction(cell_x, cell_y, neigh_x, neigh_y):
    diff_x = neigh_x - cell_x
    diff_y = neigh_y - cell_y 
    assert diff_x != 0 or diff_y != 0
    if diff_x == 10 and diff_y == 0:
        return "east"
    elif diff_x == -10 and diff_y == 0:
        return "west"
    elif diff_x == 0 and diff_y == 10:
        return "north"
    elif diff_x == 0 and diff_y == -10:
        return "south"
    elif diff_x == 10 and diff_y == 10:
        return "northeast"
    elif diff_x == 10 and diff_y == -10:
        return "southeast"
    elif diff_x == -10 and diff_y == 10:
        return "northwest"
    elif diff_x == -10 and diff_y == -10:
        return "southwest"
    

for scene_name in city_refer_seq:
    output_path = osp.join(output_dir, f"{scene_name}.json")
    print("Processing the scene: " + scene_name)

    # cells = pickle.load(
    #     open(osp.join(base_path, "cells", f"{scene_name}.pkl"), "rb")
    # )  # Also use objects from here for classification
    # for cityrefer
    cells = pickle.load(
        open(osp.join(base_path, "cells", scene_name), "rb")
    )  # Also use objects from here for classification
    cells_dict = {cell.id: {
        "east": None,
        "west": None,
        "north": None,
        "south": None,
        "northeast": None,
        "northwest": None,
        "southeast": None,
        "southwest": None
    } for cell in cells}

    for cell in cells:
        cell_id = cell.id
        cell_x = cell.bbox_w[0]
        cell_y = cell.bbox_w[1]
        for neighbor in cells:
            if neighbor.id == cell_id:
                continue
            neighbor_id = neighbor.id
            neighbor_x = neighbor.bbox_w[0]
            neighbor_y = neighbor.bbox_w[1]
            if check_neighbor(cell_x, cell_y, neighbor_x, neighbor_y):
                direction = get_direction(cell_x, cell_y, neighbor_x, neighbor_y)
                cells_dict[cell_id][direction] = neighbor_id

    out_file = open(output_path, "w")
    # json.dump(output_path, cells_dict, indent = 4)
    with open(output_path, "w") as f:
        json.dump(cells_dict, f, indent=4)

    out_file.close()