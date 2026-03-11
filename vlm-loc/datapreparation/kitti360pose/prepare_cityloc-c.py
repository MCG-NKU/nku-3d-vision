from typing import List, Tuple

import cv2
import os
import os.path as osp
import numpy as np
import pickle
import sys

import torch
import open3d
from tqdm import tqdm
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.append("/home/data_sata/vlmloc/vlm-loc/")  # Add project root to sys.path
from datapreparation.kitti360pose.utils import CLASS_TO_LABEL
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.descriptions import (
    create_cell,
    describe_pose_in_pose_cell,
    ground_pose_to_best_cell,
)
from datapreparation.args import parse_arguments

import numpy as np

STUFF_CLASSES = [
    "Ground",
    "High Vegetation",
    # "Building",
    "Wall",
    "Bridge",
    "Parking",
    "Rail",
    "traffic Road",
    # "Street Furniture",
    "Footpath",
    "Water",
]

semantic_names = {
    0: 'Ground',
    1: 'High Vegetation', 
    2: 'Building',
    3: 'Wall',
    4: 'Bridge',
    5: 'Parking',
    6: 'Rail',
    7: 'traffic Road',
    8: 'Street Furniture',
    9: 'Car',
    10: 'Footpath',
    11: 'Bike',
    12: 'Water'
}


def convert_cityrefer_objects(objects, semantic_names):
    """
    Convert CityRefer-style dict objects into Object3d instances.

    Notes:
    - Input colors are in the [-1, 1] range.
    - Output colors are mapped to the [0, 1] range.
    - Semantic names are mapped automatically, and xyz_raw / rgb_raw are preserved.
    """

    converted_objects = []

    for obj in objects:
        coords = obj['coords']
        colors = obj['colors']
        sem_labels = obj['sem_labels']
        inst_labels = obj['inst_labels']
        instance_id = obj['instance_id']

        # Convert to NumPy
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
        if isinstance(sem_labels, torch.Tensor):
            sem_labels = sem_labels.cpu().numpy()
        if isinstance(inst_labels, torch.Tensor):
            inst_labels = inst_labels.cpu().numpy()

        # Map colors from [-1, 1] to [0, 1]
        colors = (colors + 1.0) / 2.0
        colors = np.clip(colors, 0.0, 1.0)

        # Semantic class name
        unique_sem = np.unique(sem_labels)
        sem_id = int(unique_sem[0]) if len(unique_sem) == 1 else int(sem_labels[0])
        sem_name = semantic_names.get(sem_id, f"Unknown({sem_id})")

        # Build Object3d
        obj3d = Object3d(
            id=int(instance_id),
            instance_id=int(instance_id),
            xyz=coords,
            rgb=colors,
            label=sem_name,
            xyz_raw=coords.copy(),
            rgb_raw=colors.copy(),
        )

        converted_objects.append(obj3d)

    # print(f"[INFO] Converted {len(converted_objects)} objects to Object3d instances.")
    return converted_objects


def sample_grid_points(objects, interval=5.0, target_semantics={0, 4, 5, 7, 10},
                       min_distance=None, margin=25.0):
    """
    Sample regular grid points within the object bbox and keep points whose nearest
    semantic label belongs to the target set.

    Args:
        objects: list[dict], each containing 'coords', 'colors', 'sem_labels',
            'inst_labels', and 'instance_id'
        interval: grid interval in meters
        target_semantics: target semantic id set
        min_distance: minimum distance between sampled points; defaults to interval
        margin: inward XY margin from the bbox boundary in meters

    Returns:
        result_points: (N, 3) ndarray of sampled 3D coordinates
        result_sem:    (N,) ndarray of semantic labels
    """
    if min_distance is None:
        min_distance = interval

    # Merge all coordinates and semantic labels
    all_coords, all_sem = [], []
    for obj in objects or []:
        coords = obj['coords']
        sem = obj['sem_labels']
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if isinstance(sem, torch.Tensor):
            sem = sem.cpu().numpy()
        if coords.size == 0:
            continue
        all_coords.append(coords)
        all_sem.append(sem)

    if len(all_coords) == 0:
        # Return empty arrays for empty input
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=int)

    all_coords = np.concatenate(all_coords, axis=0)
    all_sem = np.concatenate(all_sem, axis=0)

    # Compute the global bbox and shrink XY by margin
    min_xyz = all_coords.min(axis=0)
    max_xyz = all_coords.max(axis=0)

    min_xy_margin = min_xyz[:2] + margin
    max_xy_margin = max_xyz[:2] - margin

    # Fall back to the original XY bbox if the shrunk bbox is invalid
    if np.any(min_xy_margin >= max_xy_margin):
        print(f"Warning: bbox too small after applying margin {margin}m. Using original bbox.")
        min_xy_margin = min_xyz[:2]
        max_xy_margin = max_xyz[:2]

    # Generate regular XY grid points
    xs = np.arange(min_xy_margin[0], max_xy_margin[0] + 1e-9, interval)
    ys = np.arange(min_xy_margin[1], max_xy_margin[1] + 1e-9, interval)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=int)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
    grid_points_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    # Nearest-neighbor matching in XY only
    tree = cKDTree(all_coords[:, :2])
    dist, idx = tree.query(grid_points_xy, k=1)
    nearest_sem = all_sem[idx]
    nearest_z = all_coords[idx, 2]

    # Semantic filtering
    mask = np.isin(nearest_sem, list(target_semantics))
    if not np.any(mask):
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=int)

    result_points = np.concatenate([grid_points_xy[mask], nearest_z[mask, None]], axis=1)
    result_sem = nearest_sem[mask]

    # Remove points that are too close
    if len(result_points) > 0:
        result_points, result_sem = _enforce_min_distance(result_points, result_sem, min_distance)

    return result_points, result_sem


def _enforce_min_distance(points, sem, min_dist):
    """
    Enforce a minimum distance between points using greedy selection.
    """
    if len(points) <= 1:
        return points, sem

    tree = cKDTree(points)
    selected = np.zeros(len(points), dtype=bool)
    remaining = np.arange(len(points))

    while len(remaining) > 0:
        idx = remaining[0]
        selected[idx] = True
        # Find all points within min_dist and remove them
        neighbors = tree.query_ball_point(points[idx], r=min_dist)
        remaining = np.setdiff1d(remaining, neighbors, assume_unique=True)

    return points[selected], sem[selected]


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Convert an RGB array to uint8 [0, 255], supporting uint8 and float inputs.
    - If dtype is uint8: return as is
    - If dtype is float: multiply by 255 if max <= 1.0, otherwise assume [0, 255]
    """
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32, copy=False)
    maxv = np.nanmax(arr) if arr.size else 1.0
    if maxv <= 1.0 + 1e-6:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)

def project_points_to_pixels(xy: np.ndarray, x_min, y_min, x_scale, y_scale, img_size_minus_1):
    """
    World -> pixel projection using left-edge binning.

    Returns:
      x_img (int32), y_img_flipped (int32), idx (int64)

    Notes:
      - W = img_size_minus_1 + 1
      - x_scale and y_scale are only used through bev_range = W / x_scale = H / y_scale
    """
    W = int(img_size_minus_1) + 1
    H = W  # Square BEV

    # Left-edge binning: i = floor((x - x_min) * W / range)
    bev_range_x = W / float(x_scale)
    bev_range_y = H / float(y_scale)

    x = xy[:, 0]
    y = xy[:, 1]

    # Avoid floating-point issues on the right boundary
    eps = 1e-9

    # Compute unflipped image row first, then flip vertically
    i = np.floor(((x - x_min) * W / bev_range_x) - eps).astype(np.int64, copy=False)
    j0 = np.floor(((y - y_min) * H / bev_range_y) - eps).astype(np.int64, copy=False)

    # Clamp to valid indices
    np.clip(i, 0, W - 1, out=i)
    np.clip(j0, 0, H - 1, out=j0)

    # Flip y: image rows go top-to-bottom, BEV y goes bottom-to-top
    j = (H - 1) - j0

    # Cast back to int32 to preserve the original return type
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

        # Clean nested objects if present
        if hasattr(cell, 'objects'):
            for obj in getattr(cell, 'objects'):
                safe_delattr(obj, 'xyz_raw')
                safe_delattr(obj, 'rgb_raw')

    # Save compact cells
    with open(path_cells, "wb") as f:
        pickle.dump(cells, f)

    print(f"[{scene_name}] Saved {len(cells)} cleaned cells to {path_cells}")

def pixels_to_world_from_center(
    x_pix, y_pix,
    center_world,
    bev_range,
    image_size,
    use_center=True
):
    """
    Pixel -> world back-projection anchored at the pixel center.

    This is the strict dual of project_points_to_pixels:
    given (x_pix, y_pix), back-project to the world coordinate of that pixel center,
    and projecting it again returns the same pixel.
    """
    W = int(image_size)
    H = W
    cx_w, cy_w = float(center_world[0]), float(center_world[1])
    half = float(bev_range) / 2.0

    x_min = cx_w - half
    y_min = cy_w - half

    # Pixel-center offset
    off = 0.5 if use_center else 0.0

    # Convert back to the unflipped row index
    x_pix = np.asarray(x_pix, dtype=np.float64)
    y_pix = np.asarray(y_pix, dtype=np.float64)

    j = y_pix
    j0 = (H - 1) - j

    # World coordinates of the pixel center
    xw = x_min + ((x_pix + off) * bev_range) / W
    yw = y_min + ((j0 + off) * bev_range) / H
    return xw, yw

def show(img_or_name, img=None):
    if img is not None:
        cv2.imshow(img_or_name, img)
    else:
        cv2.imshow("", img_or_name)
    cv2.waitKey()


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


def gather_objects_cityrefer(objects):
    """
    Process CityRefer objects and convert them into filtered Object3d instances.

    - Keep both raw (xyz_raw) and downsampled (xyz) versions.
    - Apply random 1/10 downsampling.
    - Filter out objects with fewer than 200 points after downsampling.
    """
    print(f"Processing {len(objects)} objects for CityRefer...")
    objects_threshed = []
    thresh_counts = 0

    converted_objects = convert_cityrefer_objects(objects, semantic_names)
    for obj in tqdm(converted_objects, desc="downsampling"):
        xyz = obj.xyz
        n_points = len(xyz)
        if n_points == 0:
            continue

        # Random 1/10 downsampling, keeping at least one point
        num_keep = max(1, int(n_points / 10))
        idx = np.random.choice(n_points, num_keep, replace=False)
        obj.xyz = xyz[idx]
        obj.rgb = obj.rgb[idx]

        assert obj.xyz.shape[0] == obj.rgb.shape[0]
        # Filter out instances with fewer than 200 points
        if len(obj.xyz) < 200:
            thresh_counts += 1
            continue

        objects_threshed.append(obj)

    print(f"[INFO] Filtered out {thresh_counts} instances (<200 pts).")
    print(f"[INFO] Remaining objects: {len(objects_threshed)} / {len(objects)}")

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
    *,
    color_mode: str = "mean",
):
    """
    Generate a semantic BEV image and return object pixel/world centers and colors.

    Args:
        cell: cell object containing .bbox_w and .objects
        image_size: square image side length in pixels
        bev_range: world-to-pixel projection range
        save_path: optional output path
        color_mode: "mean" for per-object mean RGB, "point" for per-point RGB

    Returns:
        bev_image (H, W, 3) uint8, centers: List[dict]
    """
    bbox = cell.bbox_w
    center_location = 0.5 * (bbox[0:3] + bbox[3:6])

    x_min, y_min = bbox[0], bbox[1]
    x_max, y_max = bbox[3], bbox[4]

    H = W = int(image_size)
    x_scale = image_size / (x_max - x_min + 1e-8)
    y_scale = image_size / (y_max - y_min + 1e-8)
    img_size_minus_1 = image_size - 1

    # Initialize a white-background BEV image
    bev_image = np.full((H, W, 3), 255, dtype=np.uint8)
    flat_img = bev_image.reshape(-1, 3)

    centers = []

    # Precompute per-object projections
    stuff_entries = []
    nonstuff_entries = []

    for obj in cell.objects:
        # Render with raw points
        xyz = getattr(obj, "xyz_raw", None)
        rgb = getattr(obj, "rgb_raw", None)

        if not isinstance(xyz, np.ndarray) or xyz.ndim != 2 or xyz.shape[1] < 2:
            continue

        # Convert colors to uint8
        rgb_u8 = to_uint8_rgb(rgb) if rgb is not None else np.full((xyz.shape[0], 3), 200, np.uint8)

        # Project points to flat pixel indices
        _, _, idx = project_points_to_pixels(
            xyz, x_min, y_min, x_scale, y_scale, img_size_minus_1
        )
        if idx.size == 0:
            continue

        mean_col = np.round(rgb_u8.astype(np.float32).mean(axis=0)).astype(np.uint8)
        entry = (idx, rgb_u8, mean_col, obj.label)

        if obj.label in STUFF_CLASSES:
            stuff_entries.append(entry)
        else:
            nonstuff_entries.append(entry)

    def _paint_and_collect(entries):
        nonlocal centers, flat_img
        for idx, rgb_u8, mean_col, lab in entries:
            if idx.size == 0:
                continue

            if color_mode == "point":
                # Per-point RGB; later writes overwrite earlier ones on the same pixel
                flat_img[idx] = rgb_u8[:idx.size]
                indices_for_center = np.unique(idx)
            else:
                # Per-object mean RGB
                uniq_idx = np.unique(idx)
                flat_img[uniq_idx] = mean_col
                indices_for_center = uniq_idx

            if indices_for_center.size == 0:
                continue

            # Compute the pixel centroid of the object
            ys = (indices_for_center // W).astype(np.float32)
            xs = (indices_for_center % W).astype(np.float32)
            x_mean, y_mean = float(xs.mean()), float(ys.mean())

            # Convert back to world coordinates
            xw, yw = pixels_to_world_from_center(
                x_mean, y_mean,
                center_world=center_location[:2],
                bev_range=bev_range,
                image_size=H,
                use_center=True
            )

            # Store display color using the mean color
            centers.append({
                "label": lab,
                "pixel_center": (int(round(x_mean)), int(round(y_mean))),
                "world_center": (float(xw), float(yw)),
                "color_rgb": (int(mean_col[0]), int(mean_col[1]), int(mean_col[2])),
            })

    # Draw STUFF first, then non-STUFF
    _paint_and_collect(stuff_entries)
    _paint_and_collect(nonstuff_entries)

    # Save image
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR))

    return bev_image, centers


def create_cells(objects, locations, scene_name, cell_size, args) -> Tuple[bool, List["Cell"]]:
    """Create cells only from the given locations, with one axis-aligned cell per location."""
    print("Creating cells (locations-only)...")
    cells = []
    none_indices = []

    locations = np.asarray(locations, dtype=float)

    scene_name_short = "_".join(scene_name.split("_")[0:3])

    # Generate one cell per location without shift, grid, or deduplication
    scene_path = os.path.join(args.path_out, "bev_image", scene_name)
    os.makedirs(scene_path, exist_ok=True)
    centers_info = []
    image_paths = []
    for i_location, location in enumerate(tqdm(locations, desc='Generating Cells', ncols=100)):
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


            descriptions = describe_pose_in_pose_cell(
                location, pose_cell, description_method, args.num_mentioned
            )

            if descriptions is None or len(descriptions) < args.num_mentioned:
                none_indices.append(i_location)
                do_break = True  # Don't try again with other strategy
                continue

            assert len(descriptions) == args.num_mentioned
            
            descriptions, pose_in_cell, num_unmatched = ground_pose_to_best_cell(
                location, descriptions, best_cell
            )
            assert len(descriptions) == args.num_mentioned
            unmatched_counts.append(num_unmatched)

            # Only append the new description if it actually comes out different than the ones before
            mentioned_ids = sorted([d.object_id for d in descriptions if d.is_matched])
            if mentioned_ids in mentioned_object_ids:
                num_duplicates += 1
            else:
                pose = Pose(
                    pose_in_cell,
                    location,
                    best_cell.id,
                    best_cell.scene_name,
                    descriptions,
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

import copy

if __name__ == "__main__":
    np.random.seed(4096)  # Main-process seed
    args = parse_arguments()
    print(str(args).replace(",", "\n"))
    print()

    test_dir = '/home/data_sata/vlmloc/CityRefer/data/sensaturban/balance_split/test'

    split = 'test'  # 'val' or 'test'

    scene_dir = test_dir

    # Statistics
    total_cells = 0
    total_poses = 0
    scene_stats = []

    max_bbox_global = None
    for scene_name in sorted(os.listdir(scene_dir)):
        args.scene_name = scene_name.split('.')[0]
        print(f"Processing scene {args.scene_name}...")

        # Load scene data
        objects = torch.load(osp.join(scene_dir, scene_name))

        cell_locations, result_sem = sample_grid_points(objects, interval=20.0)
        pose_locations = copy.deepcopy(cell_locations)

        # Report semantic distribution
        unique, counts = np.unique(result_sem, return_counts=True)

        filtered_objects = gather_objects_cityrefer(objects)

        res, cells = create_cells(filtered_objects, cell_locations, args.scene_name, args.cell_size, args)
        if cells is None or len(cells) == 0:
            print(f"Scene {args.scene_name}: No cells created, skipping.")
            continue
        res, poses = create_poses(filtered_objects, pose_locations, cells, args)

        num_cells = len(cells)
        num_poses = len(poses)

        print(f'Scene {args.scene_name}: cells={num_cells}, queries={num_poses}')

        pickle.dump(filtered_objects, open(osp.join(args.path_out, 'objects', f"{args.scene_name}.pkl"), "wb"))
        pickle.dump(cells, open(osp.join(args.path_out, 'cells', f"{args.scene_name}.pkl"), "wb"))
        pickle.dump(poses, open(osp.join(args.path_out, 'poses', f"{args.scene_name}.pkl"), "wb"))
        # Accumulate statistics
        total_cells += num_cells
        total_poses += num_poses
        scene_stats.append({
            'scene': args.scene_name,
            'cells': num_cells,
            'queries': num_poses
        })

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)

    print("\nPer-scene breakdown:")
    print(f"{'Scene':<30} {'Cells':<10} {'Queries':<10}")
    print("-" * 50)
    for stat in scene_stats:
        print(f"{stat['scene']:<30} {stat['cells']:<10} {stat['queries']:<10}")

    print("-" * 50)
    print(f"{'TOTAL':<30} {total_cells:<10} {total_poses:<10}")

    print(f"\nSummary:")
    print(f"  - Total scenes processed: {len(scene_stats)}")
    print(f"  - Total cells (database): {total_cells}")
    print(f"  - Total queries: {total_poses}")
    print(f"  - Average cells per scene: {total_cells / len(scene_stats):.1f}")
    print(f"  - Average queries per scene: {total_poses / len(scene_stats):.1f}")
    print("=" * 60)

