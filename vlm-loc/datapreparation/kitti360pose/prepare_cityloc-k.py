from typing import List, Tuple
import cv2
import os
import os.path as osp
import numpy as np
import pickle
import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "..", ".."))
import time
import open3d
from tqdm import tqdm
from plyfile import PlyData

from datapreparation.kitti360pose.utils import CLASS_TO_LABEL
from datapreparation.kitti360pose.utils import CLASS_TO_MINPOINTS, CLASS_TO_VOXELSIZE, STUFF_CLASSES
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.descriptions import (
    create_cell,
    describe_pose_in_pose_cell,
    ground_pose_to_best_cell,
)
from datapreparation.args import parse_arguments

def project_points_to_pixels(xy: np.ndarray, x_min, y_min, x_scale, y_scale, img_size_minus_1):
    """
    World -> pixel projection using left-edge binning.

    Returns:
        x_img (int32), y_img_flipped (int32), idx (int64)

    Notes:
        - W = img_size_minus_1 + 1
        - x_scale and y_scale are only used to derive the BEV range
    """
    W = int(img_size_minus_1) + 1
    H = W  # Square BEV

    # Use left-edge binning: i = floor((x - x_min) * W / range)
    # where range = W / x_scale
    bev_range_x = W / float(x_scale)
    bev_range_y = H / float(y_scale)

    x = xy[:, 0]
    y = xy[:, 1]

    # Avoid floating-point issues on the right boundary
    eps = 1e-9

    # Compute the unflipped image row first, then flip vertically
    i = np.floor(((x - x_min) * W / bev_range_x) - eps).astype(np.int64, copy=False)
    j0 = np.floor(((y - y_min) * H / bev_range_y) - eps).astype(np.int64, copy=False)

    # Clamp to valid indices
    np.clip(i, 0, W - 1, out=i)
    np.clip(j0, 0, H - 1, out=j0)

    # Flip y: image rows go top-to-bottom, while BEV y goes bottom-to-top
    j = (H - 1) - j0

    # Cast back to int32 to keep the original return types
    x_img = i.astype(np.int32, copy=False)
    y_img_flipped = j.astype(np.int32, copy=False)

    idx = (j * W + i).astype(np.int64, copy=False)
    return x_img, y_img_flipped, idx

def save_cells_without_raw(cells, path_cells, scene_name):
    """
    Remove xyz_raw and rgb_raw from each cell before saving.
    """
    def safe_delattr(obj, name):
        if hasattr(obj, name):
            delattr(obj, name)

    for cell in cells:
        # Remove top-level attributes
        safe_delattr(cell, 'xyz_raw')
        safe_delattr(cell, 'rgb_raw')

        # Clean nested objects as well if present
        if hasattr(cell, 'objects'):
            for obj in getattr(cell, 'objects'):
                safe_delattr(obj, 'xyz_raw')
                safe_delattr(obj, 'rgb_raw')

    # Save the compact cells
    with open(path_cells, "wb") as f:
        pickle.dump(cells, f)

    print(f"[{scene_name}] Saved {len(cells)} cleaned cells to {path_cells}")



def pixels_to_world_from_center(
    x_pix, y_pix,
    center_world,   # (cx_w, cy_w)
    bev_range,      # covered side length in meters; consistent with bev_range = W / x_scale
    image_size,     # H = W
    use_center=True # always back-project using the pixel center
):
    """
    Pixel -> world back-projection anchored at the pixel center.

    This is strictly dual to the binning used in project_points_to_pixels:
    given (x_pix, y_pix), back-project to the world-space center of that pixel.
    Projecting it forward again returns the same pixel.
    """
    W = int(image_size)
    H = W
    cx_w, cy_w = float(center_world[0]), float(center_world[1])
    half = float(bev_range) / 2.0

    x_min = cx_w - half
    y_min = cy_w - half

    # Pixel center offset
    off = 0.5 if use_center else 0.0

    # Convert flipped row index back to the original row index
    x_pix = np.asarray(x_pix, dtype=np.float64)
    y_pix = np.asarray(y_pix, dtype=np.float64)

    j = y_pix
    j0 = (H - 1) - j

    # World coordinates of the pixel center
    xw = x_min + ((x_pix + off) * bev_range) / W
    yw = y_min + ((j0    + off) * bev_range) / H
    return xw, yw


def load_points(filepath):
    plydata = PlyData.read(filepath)

    xyz = np.stack((plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"])).T
    rgb = np.stack(
        (plydata["vertex"]["red"], plydata["vertex"]["green"], plydata["vertex"]["blue"])
    ).T

    lbl = plydata["vertex"]["semantic"]
    iid = plydata["vertex"]["instance"]

    return xyz, rgb, lbl, iid


def downsample_points(points, voxel_size):
    # voxel_size = 0.25
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points.copy())
    _, _, indices_list = point_cloud.voxel_down_sample_and_trace(
        voxel_size, point_cloud.get_min_bound(), point_cloud.get_max_bound()
    )
    # print(f'Downsampled from {len(points)} to {len(indices_list)} points')

    indices = np.array(
        [vec[0] for vec in indices_list]
    )  # Not vectorized but seems fast enough, CARE: first-index color sampling (not averaging)

    return indices


def extract_objects(xyz, rgb, lbl, iid):
    """
    1. Classify points into objects based on their semantic labels.
    2. For each object, create an Object3d instance.
    """
    objects = []

    for label_name, label_idx in CLASS_TO_LABEL.items():
        mask = lbl == label_idx
        label_xyz, label_rgb, label_iid = xyz[mask], rgb[mask], iid[mask]

        for obj_iid in np.unique(label_iid):
            mask = label_iid == obj_iid
            obj_xyz, obj_rgb = label_xyz[mask], label_rgb[mask]
            obj_xyz_raw, obj_rgb_raw = label_xyz[mask], label_rgb[mask]

            obj_rgb = obj_rgb.astype(np.float32) / 255.0  # Scale colors [0,1]

            # objects.append(Object3d(obj_xyz, obj_rgb, label_name, obj_iid))
            objects.append(
                Object3d(obj_iid, obj_iid, obj_xyz, obj_rgb, label_name, obj_xyz_raw, obj_rgb_raw)
            )  # Initially also set id instance-id for later mergin. Re-set in create_cell()

    return objects

def gather_objects_both(path_input, folder_name):
    """
    Gather all objects in a KITTI-360 scene.

    This version keeps both raw and downsampled point clouds:
      - obj.xyz_raw: raw point cloud
      - obj.xyz: downsampled point cloud (based on CLASS_TO_VOXELSIZE)

    Filtering uses the downsampled point count, but the returned Object3d
    instances still keep the raw point cloud.
    """
    print(f"Loading objects for {folder_name}")

    path = osp.join(path_input, "data_3d_semantics", folder_name, "static")
    assert osp.isdir(path)
    file_names = [f for f in os.listdir(path) if not f.startswith("._")]

    scene_objects = {}

    for file_name in tqdm(file_names, desc="loading objects"):
        xyz, rgb, lbl, iid = load_points(osp.join(path, file_name))
        file_objects = extract_objects(xyz, rgb, lbl, iid)

        # Merge objects with the same id
        for obj in file_objects:
            if obj.id in scene_objects:
                scene_objects[obj.id] = Object3d.merge(scene_objects[obj.id], obj)
            else:
                scene_objects[obj.id] = obj

    # Secondary processing stage
    objects = list(scene_objects.values())
    thresh_counts = {}
    objects_threshed = []

    for obj in objects:
        # Downsample using the class-specific voxel size
        voxel_size = CLASS_TO_VOXELSIZE.get(obj.label, None)
        if voxel_size is not None:
            ds_idx = downsample_points(obj.xyz, voxel_size)
            obj.xyz = obj.xyz[ds_idx]
            if hasattr(obj, "rgb"):
                obj.rgb = obj.rgb[ds_idx]

        # Count points after downsampling
        ds_count = len(obj.xyz)
        min_required = CLASS_TO_MINPOINTS[obj.label]

        # Filter by downsampled point count
        if ds_count < min_required:
            thresh_counts[obj.label] = thresh_counts.get(obj.label, 0) + 1
        else:
            objects_threshed.append(obj)
        assert obj.xyz_raw is not None
    print(thresh_counts)
    return objects_threshed



def get_close_locations(
    locations: List[np.ndarray], scene_objects: List[Object3d], cell_size, location_objects=None
):
    """Retains all locations that are at most cell_size / 2 distant from an instance-object.

    Args:
        locations (List[np.ndarray]): Pose locations
        scene_objects (List[Object3d]): All objects in the scene
        cell_size ([type]): Size of a cell
        location_objects (optional): Location objects to plot for debugging. Defaults to None.
    """
    instance_objects = [obj for obj in scene_objects if obj.label not in STUFF_CLASSES]
    close_locations, close_location_objects = [], []
    for i_location, location in enumerate(locations):
        for obj in instance_objects:
            closest_point = obj.get_closest_point(location)
            dist = np.linalg.norm(location - closest_point)
            obj.closest_point = None
            if dist < cell_size / 2:
                close_locations.append(location)
                close_location_objects.append(location_objects[i_location])
                break

    assert (
        len(close_locations) > len(locations) * 2 / 5
    ), f"Too few locations retained ({len(close_locations)} of {len(locations)}), are all objects loaded?"
    print(f"close locations: {len(close_locations)} of {len(locations)}")

    if location_objects:
        return close_locations, close_location_objects
    else:
        return close_locations


def create_locations(path_input, folder_name, location_distance, return_location_objects=False):
    """Sample locations along the original trajectories with at least location_distance between any two locations."""
    path = osp.join(path_input, "data_poses", folder_name, "poses.txt")
    poses = np.loadtxt(path)
    poses = poses[:, 1:].reshape((-1, 3, 4))  # Convert to 3x4 matrices
    poses = poses[:, :, -1]  # Take last column (translation), [N,3]

    sampled_poses = [
        poses[0],
    ] # valid sample locations
    for pose in poses:
        dists = np.linalg.norm(pose - sampled_poses, axis=1)
        if np.min(dists) >= location_distance:
            sampled_poses.append(pose)

    if return_location_objects:
        pose_objects = []
        for pose in sampled_poses:
            pose_objects.append(
                Object3d(-1, -1, np.random.rand(50, 3) * 3 + pose, np.ones((50, 3)), "_pose")
            )
        print(f"{folder_name} sampled {len(sampled_poses)} locations")
        return sampled_poses, pose_objects
    else:
        return sampled_poses




def generate_sem_bev_image(
    cell,
    image_size,
    bev_range,
    save_path=None,
):
    bbox = cell.bbox_w  # shape: (6,)
    center_location = 0.5 * (bbox[0:3] + bbox[3:6])

    x_min, y_min = bbox[0], bbox[1]
    x_max, y_max = bbox[3], bbox[4]

    H = W = int(image_size)
    x_scale = image_size / (x_max - x_min)
    y_scale = image_size / (y_max - y_min)
    img_size_minus_1 = image_size - 1

    # 1) Initialize a white-background BEV image
    bev_image = np.full((H, W, 3), 255, dtype=np.uint8)
    flat_img = bev_image.reshape(-1, 3)

    centers = []

    # 2) Precompute the projection for each object and process object by object
    stuff_entries = []     # [(uniq_idx, mean_col, label), ...]
    nonstuff_entries = []  # [(uniq_idx, mean_col, label), ...]

    for obj in cell.objects:
        # Render with raw points
        xyz = getattr(obj, "xyz_raw", None)
        rgb = getattr(obj, "rgb_raw", None)

        if not isinstance(xyz, np.ndarray) or xyz.ndim != 2 or xyz.shape[1] < 2:
            continue

        # Project 3D points to pixel indices
        _, _, idx = project_points_to_pixels(
            xyz, x_min, y_min, x_scale, y_scale, img_size_minus_1
        )
        if idx.size == 0:
            continue

        uniq_idx = np.unique(idx)
        # Mean color for the object
        mean_col = np.round(rgb.astype(np.float32).mean(axis=0)).astype(np.uint8)
        entry = (uniq_idx, mean_col, obj.label)

        if obj.label in STUFF_CLASSES:
            stuff_entries.append(entry)
        else:
            nonstuff_entries.append(entry)

    # 3) Draw STUFF first, then non-STUFF, both object by object
    def _paint_and_collect(entries):
        nonlocal centers, flat_img
        for uniq_idx, mean_col, lab in entries:
            if uniq_idx.size == 0:
                continue

            # Paint pixels
            flat_img[uniq_idx] = mean_col

            # Compute the pixel-space centroid
            ys = (uniq_idx // W).astype(np.float32)
            xs = (uniq_idx %  W).astype(np.float32)
            x_mean, y_mean = float(xs.mean()), float(ys.mean())

            # Convert the centroid back to world coordinates
            xw, yw = pixels_to_world_from_center(
                x_mean, y_mean,
                center_world=center_location[:2],
                bev_range=bev_range,
                image_size=H,
                use_center=True
            )

            centers.append({
                "label": lab,
                "pixel_center": (int(round(x_mean)), int(round(y_mean))),
                "world_center": (float(xw), float(yw)),
                "color_rgb": (int(mean_col[0]), int(mean_col[1]), int(mean_col[2])),
            })

    # Draw STUFF object by object
    _paint_and_collect(stuff_entries)
    # Draw non-STUFF object by object
    _paint_and_collect(nonstuff_entries)

    # 4) Save the image
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR))

    return bev_image, centers

def create_cells(objects, locations, scene_name, cell_size, args) -> Tuple[bool, List["Cell"]]:
    """
    Create cells only from the given locations, each with an axis-aligned bbox
    of size args.cell_size. No grid generation, no shifting, and one cell per input location.
    """
    print("Creating cells (locations-only)...")
    cells = []
    none_indices = []

    locations = np.asarray(locations, dtype=float)

    assert len(scene_name.split("_")) == 6
    scene_name_short = scene_name.split("_")[-2]


    scene_path = os.path.join(args.path_out, "bev_image", scene_name)
    os.makedirs(scene_path, exist_ok=True)
    centers_info = []
    image_paths = []
    for i_location, location in enumerate(tqdm(locations, desc='Generating Cells')):
        # bbox: [x0, y0, z0, x1, y1, z1]
        bbox = np.hstack((location - cell_size / 2, location + cell_size / 2))

        cell = create_cell(
            i_location,
            scene_name_short,
            bbox,
            objects,
            num_mentioned=args.num_mentioned,
            stuff_classes=STUFF_CLASSES
        )

        if cell is not None:
            assert len(cell.objects) >= args.num_mentioned
            cells.append(cell)
            image_path = osp.join(scene_path, f"bev_{i_location:06d}.png")
            bev_image, centers = generate_sem_bev_image(cell, image_size=224, bev_range=50, save_path=image_path)
            centers_info.append(centers)
            image_paths.append(image_path)
            save_data = {
                "centers_info": centers_info,
                "image_paths": image_paths
            }
            pickle.dump(save_data, open(osp.join(args.path_out, f"{scene_name}_centers_info.pkl"), "wb"))
        else:
            none_indices.append(i_location)

    print(f"None cells: {len(none_indices)} / {len(locations)}")
    return (False, none_indices) if len(none_indices) > len(locations) else (True, cells)


def create_poses(objects: List[Object3d], locations, cells: List[Cell], args) -> List[Pose]:
    """Create the poses of a scene.
    Create cells -> sample pose location -> describe with pose-cell -> convert description to best-cell for training

    Args:
        objects (List[Object3d]): Objects of the scene, needed to verify enough objects a close to a given pose.
        locations: Locations of the original trajectory around which to sample the poses.
        cells (List[Cell]): List of cells
        scene_name: Name of the scene
    """
    print("Creating poses...")
    poses = []
    none_indices = []

    cell_centers = np.array([cell.bbox_w for cell in cells])
    cell_centers = 1 / 2 * (cell_centers[:, 0:3] + cell_centers[:, 3:6])

    if args.pose_count > 1:
        locations = np.repeat(
            locations, args.pose_count, axis=0
        )  # Repeat the locations to increase the number of poses. (Poses are randomly shifted below.)
        assert (
            args.shift_poses == True
        ), "Pose-count greater than 1 but pose shifting is deactivated!"

    unmatched_counts = []
    num_duplicates = 0
    num_rejected = 0  # Pose that were rejected because the cell was None
    for i_location, location in enumerate(tqdm(locations, desc='Generating Poses')):
        # Shift the poses randomly to de-correlate database-side cells and query-side poses.
        if args.shift_poses:
            location[0:2] += np.intp(
                np.random.rand(2) * 30 / 2.1
            )  # Shift less than 1 / 2 cell-size so that the pose has a corresponding cell # TODO: shift more?

        # Find closest cell. Discard poses too far from a database-cell so that all poses are retrievable.
        dists = np.linalg.norm(location - cell_centers, axis=1)
        best_cell = cells[np.argmin(dists)]

        eval_half = 50.0 / 2.0
        lower = cell_centers - eval_half
        upper = cell_centers + eval_half
        
        contains = np.all((location >= lower) & (location <= upper), axis=1)
        candidate_indices = np.where(contains)[0]
        if len(candidate_indices) == 0:
            none_indices.append(i_location)
            continue

        rand_idx = np.random.randint(0, len(candidate_indices))
        eval_cell = cells[candidate_indices[rand_idx]]

        if np.min(dists) > args.cell_size / 2:
            none_indices.append(i_location)
            continue

        # Create an extra cell on top of the pose to create the query-side description decoupled from the database-side cells.
        pose_cell_bbox = np.hstack(
            (location - args.cell_size / 2, location + args.cell_size / 2)
        )  # [x0, y0, z0, x1, y1, z1]
        pose_cell = create_cell(
            -1, "pose", pose_cell_bbox, objects, num_mentioned=args.num_mentioned, stuff_classes=STUFF_CLASSES
        )

       
        if pose_cell is None:  # Pose can be too far from objects to describe it
            none_indices.append(i_location)
            num_rejected += 1
            continue

        # Select description strategy / strategies
        if args.describe_by == "all":
            description_methods = ("closest", "class", "direction")
        else:
            description_methods = (args.describe_by,)

        mentioned_object_ids = []
        do_break = False
        for description_method in description_methods:
            if do_break:
                break

            # if args.describe_best_cell:  # Ablation: use ground-truth best cell to describe the pose
            #     descriptions = describe_pose_in_pose_cell(
            #         location, best_cell, description_method, args.num_mentioned
            #     )
            # else:  # Obtain the descriptions based on the pose-cell
            descriptions = describe_pose_in_pose_cell(
                location, pose_cell, description_method, args.num_mentioned
            )

            if descriptions is None or len(descriptions) < args.num_mentioned:
                none_indices.append(i_location)
                do_break = True  # Don't try again with other strategy
                continue

            assert len(descriptions) == args.num_mentioned

            # Convert the descriptions to the best-matching database cell for training. Some descriptions might not be matched anymore.
            # if args.all_cells:
            #     descriptions, pose_in_cell, num_unmatched = ground_pose_to_best_cell(
            #         location, descriptions, best_cell, all_cells=True
            #     )
            # else:
            
            descriptions, pose_in_cell, num_unmatched = ground_pose_to_best_cell(
                location, descriptions, best_cell
            )
            assert len(descriptions) == args.num_mentioned
            unmatched_counts.append(num_unmatched)

            # if args.describe_best_cell:
            #     assert num_unmatched == 0, "Unmatched descriptors for best cell!"

            # Only append the new description if it actually comes out different than the ones before
            mentioned_ids = sorted([d.object_id for d in descriptions if d.is_matched])
            if mentioned_ids in mentioned_object_ids:
                num_duplicates += 1
            else:
                pose = Pose(
                    pose_in_cell,
                    location,
                    best_cell.id,
                    best_cell.scene_name,descriptions,
                    eval_cell.id,
                    eval_cell.scene_name,
                    
                    described_by=description_method,
                )

                poses.append(pose)
                mentioned_object_ids.append(mentioned_ids)

    print(f"Num duplicates: {num_duplicates} / {len(poses)}")
    print(
        f"None poses: {len(none_indices)} / {len(locations)}, avg. unmatched: {np.mean(unmatched_counts):0.1f}, num_rejected: {num_rejected}"
    )
    if len(none_indices) > len(locations):
        return False, none_indices
    else:
        return True, poses


if __name__ == "__main__":
    np.random.seed(4096)  # Seed for the main process
    args = parse_arguments()
    print(str(args).replace(",", "\n"))
    print()
    
    # for scene_name in SCENE_NAMES_TRAIN:
    # for scene_name in SCENE_NAMES_TEST:
    for scene_name in ['2013_05_28_drive_0003_sync']:
        args.scene_name = scene_name
        cell_locations, cell_location_objects = create_locations(
            args.path_in,
            args.scene_name,
            location_distance=args.cell_dist,
            return_location_objects=True,
        )  # Create sampled locations for database cells
        pose_locations, pose_location_objects = create_locations(
            args.path_in,
            args.scene_name,
            location_distance=args.pose_dist,
            return_location_objects=True,
        )  # Create sampled locations for query poses

        path_objects = osp.join(args.path_out, "objects", f"{args.scene_name}.pkl")
        path_cells = osp.join(args.path_out, "cells", f"{args.scene_name}.pkl")
        path_poses = osp.join(args.path_out, "poses", f"{args.scene_name}.pkl")

        t_start = time.time()

        # Load or gather objects
        # if not osp.isfile(path_objects):  # Build if not cached
        objects = gather_objects_both(args.path_in, args.scene_name)
        pickle.dump(objects, open(path_objects, "wb"))
        print(f"Saved objects to {path_objects}")
        # else:
        #     print(f"Loaded objects from {path_objects}")
        #     objects = pickle.load(open(path_objects, "rb"))

        t_object_loaded = time.time()

        cell_locations, cell_location_objects = get_close_locations(
            cell_locations, objects, args.cell_size, cell_location_objects
        )
        pose_locations, pose_location_objects = get_close_locations(
            pose_locations, objects, args.cell_size, pose_location_objects
        )

        t_close_locations = time.time()

        # if not osp.isfile(path_cells):
            # Create cells for each sampled location
        res, cells = create_cells(objects, cell_locations, args.scene_name, args.cell_size, args)
        # else:
        #     cells = pickle.load(open(path_cells, "rb"))
        #     print(f"Loaded {len(cells)} cells from {path_cells}")
        # t_cells_created = time.time()

        res, poses = create_poses(objects, pose_locations, cells, args)
        assert res is True, "Too many pose nones, quitting."

        t_poses_created = time.time()

        # print(
        #     f"Ela: objects {t_object_loaded - t_start:0.2f} close {t_close_locations - t_object_loaded:0.2f} cells {t_cells_created - t_close_locations:0.2f} poses {t_poses_created - t_cells_created:0.2f}"
        # )
        # print()

        # raw point cloud is used only for generating BEV image. Remove it from the saved cells to save space.
        save_cells_without_raw(cells, path_cells, scene_name)
        print(f"Saved {len(cells)} cells to {path_cells}")
        pickle.dump(poses, open(path_poses, "wb"))
        print(f"Saved {len(poses)} poses to {path_poses}")
        print()
