from __future__ import annotations

import glob
import gzip
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch

try:
    from pytorch3d.ops import knn_points as _knn_points
except Exception:
    _knn_points = None

try:
    from chamferdist.chamfer import knn_points as _knn_points_chamfer
except Exception:
    _knn_points_chamfer = None

from gradslam.structures.pointclouds import Pointclouds

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.eval import compute_confmatrix, compute_pred_gt_associations


def compute_keep_index(
    class_names: List[str],
    gt_class: torch.Tensor,
    base_ignore_index: np.ndarray,
    gt_class_only: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    all_class_index = np.arange(len(class_names))
    ignore_index = np.asarray(base_ignore_index).copy()

    if gt_class_only:
        existing_index = gt_class.unique().cpu().numpy()
        existing_index = existing_index[existing_index >= 0]
        non_existing_index = np.setdiff1d(all_class_index, existing_index)
        ignore_index = np.append(ignore_index, non_existing_index)
        print("Using only the classes that exist in GT of this scene:", len(existing_index))

    keep_index = np.setdiff1d(all_class_index, ignore_index)
    return keep_index, ignore_index


def load_latest_result(scans_root: str, scene_id: str, pred_exp_name: str):
    result_paths = glob.glob(
        os.path.join(
            scans_root,
            scene_id,
            "pcd_saves",
            f"{pred_exp_name}*.pkl.gz",
        )
    )
    if len(result_paths) == 0:
        raise ValueError(f"No result found for {scene_id} with {pred_exp_name}")

    result_paths = sorted(result_paths, key=os.path.getmtime)
    result_path = result_paths[-1]
    print(f"Loading mapping result from {result_path}")

    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    return results, result_path


def load_objects_from_results(results) -> MapObjectList:
    objects = MapObjectList()
    objects.load_serializable(results["objects"])
    return objects


def append_replica_background_objects(
    results,
    objects: MapObjectList,
    object_class: torch.Tensor,
    class_names: List[str],
) -> Tuple[MapObjectList, torch.Tensor]:
    if results.get("bg_objects", None) is None:
        print("Warning: no background objects found. This is expected if only SAM is used, but not the detector.")
        return objects, object_class

    bg_objects = MapObjectList()
    bg_objects.load_serializable(results["bg_objects"])

    for obj in bg_objects:
        class_name = obj["class_name"][0].lower()
        class_id = class_names.index(class_name)
        object_class = torch.cat([object_class, object_class.new_full([1], class_id)])

    objects += bg_objects
    return objects, object_class


def build_pred_tensors(
    objects: MapObjectList,
    object_class: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    pred_xyz_list = []
    pred_class_list = []
    pred_color_list = []

    for i in range(len(objects)):
        obj_pcd = objects[i]["pcd"]
        pts = np.asarray(obj_pcd.points)
        pred_xyz_list.append(pts)
        pred_class_list.append(np.ones(len(pts), dtype=np.int64) * int(object_class[i].item()))

        cols = np.asarray(obj_pcd.colors) if hasattr(obj_pcd, "colors") else None
        if cols is not None and len(cols) == len(pts):
            pred_color_list.append(cols)

    pred_xyz = torch.from_numpy(np.concatenate(pred_xyz_list, axis=0)).float()
    pred_class = torch.from_numpy(np.concatenate(pred_class_list, axis=0)).long()

    pred_color = None
    if len(pred_color_list) == len(pred_xyz_list) and len(pred_color_list) > 0:
        pred_color = torch.from_numpy(np.concatenate(pred_color_list, axis=0)).float()

    return pred_xyz, pred_class, pred_color


def resample_to_reference(
    pred_xyz: torch.Tensor,
    pred_class: torch.Tensor,
    pred_color: Optional[torch.Tensor],
    slam_h5_path: str,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if _knn_points is None and _knn_points_chamfer is None:
        raise ImportError("Neither pytorch3d.ops.knn_points nor chamferdist.knn_points is available")

    slam_pc = Pointclouds.load_pointcloud_from_h5(slam_h5_path)
    slam_xyz = slam_pc.points_padded[0]

    if _knn_points is not None:
        nn = _knn_points(
            slam_xyz.unsqueeze(0).to(device).contiguous().float(),
            pred_xyz.unsqueeze(0).to(device).contiguous().float(),
            K=1,
            return_nn=False,
        )
        idx_slam_to_pred = nn.idx.squeeze(0).squeeze(-1).detach().cpu()
    else:
        nn = _knn_points_chamfer(
            slam_xyz.unsqueeze(0).to(device).contiguous().float(),
            pred_xyz.unsqueeze(0).to(device).contiguous().float(),
            K=1,
            return_nn=True,
        )
        idx_slam_to_pred = nn.idx.squeeze(0).squeeze(-1).detach().cpu()

    pred_xyz = slam_xyz
    pred_class = pred_class[idx_slam_to_pred]
    pred_color = pred_color[idx_slam_to_pred] if pred_color is not None else None
    return pred_xyz, pred_class, pred_color


def compute_scene_confmatrix(
    pred_xyz: torch.Tensor,
    pred_class: torch.Tensor,
    gt_xyz: torch.Tensor,
    gt_class: torch.Tensor,
    keep_index: np.ndarray,
    ignore_index: np.ndarray,
    class_names: List[str],
    device: str,
) -> torch.Tensor:
    idx_pred_to_gt, idx_gt_to_pred = compute_pred_gt_associations(
        pred_xyz.unsqueeze(0).to(device).contiguous().float(),
        gt_xyz.unsqueeze(0).to(device).contiguous().float(),
    )

    label_gt = gt_class[idx_pred_to_gt.detach().cpu()]
    pred_keep = torch.isin(label_gt, torch.from_numpy(keep_index)) & (label_gt >= 0)
    pred_class = pred_class[pred_keep]
    idx_pred_to_gt = idx_pred_to_gt[pred_keep]
    idx_gt_to_pred = None

    confmatrix = compute_confmatrix(
        pred_class.to(device),
        gt_class.to(device),
        idx_pred_to_gt,
        idx_gt_to_pred,
        class_names,
    )

    assert confmatrix.sum(0)[ignore_index].sum().item() == 0
    assert confmatrix.sum(1)[ignore_index].sum().item() == 0
    return confmatrix


def save_labeled_point_cloud(
    xyz: torch.Tensor,
    labels: torch.Tensor,
    class_colors: np.ndarray,
    save_path: str,
) -> None:
    xyz_np = xyz.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(np.int64)

    colors = np.full((xyz_np.shape[0], 3), 0.3, dtype=np.float32)
    valid = (labels_np >= 0) & (labels_np < len(class_colors))
    if valid.any():
        colors[valid] = class_colors[labels_np[valid]].astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)
