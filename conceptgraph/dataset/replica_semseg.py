from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from gradslam.structures.pointclouds import Pointclouds

from conceptgraph.dataset.replica_constants import (
    REPLICA_CLASSES,
    REPLICA_EXISTING_CLASSES,
    REPLICA_SCENE_IDS,
    REPLICA_SCENE_IDS_,
)


def get_class_setup() -> Tuple[torch.Tensor, List[str]]:
    class_all2existing = torch.ones(len(REPLICA_CLASSES), dtype=torch.long) * -1
    for i, class_id in enumerate(REPLICA_EXISTING_CLASSES):
        class_all2existing[class_id] = i
    class_names = [REPLICA_CLASSES[i] for i in REPLICA_EXISTING_CLASSES]
    return class_all2existing, class_names


def resolve_exclude_indices(n_exclude: int, class_names: List[str]) -> np.ndarray:
    if n_exclude == 1:
        exclude_names = ["other"]
    elif n_exclude == 4:
        exclude_names = ["other", "floor", "wall", "ceiling"]
    elif n_exclude == 6:
        exclude_names = ["other", "floor", "wall", "ceiling", "door", "window"]
    else:
        raise ValueError(f"Invalid n_exclude: {n_exclude}")

    return np.array([class_names.index(c) for c in exclude_names], dtype=np.int64)


def resolve_scene_ids(scene_ids: List[str]) -> Tuple[List[str], List[str]]:
    if "all" in scene_ids:
        return REPLICA_SCENE_IDS, REPLICA_SCENE_IDS_

    selected_scene_ids = scene_ids
    selected_scene_ids_ = [
        REPLICA_SCENE_IDS_[REPLICA_SCENE_IDS.index(sid)] for sid in selected_scene_ids
    ]
    return selected_scene_ids, selected_scene_ids_


def load_gt(
    replica_semantic_root: Path,
    scene_id_: str,
    class_all2existing: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gt_pc_path = os.path.join(replica_semantic_root, scene_id_, "Sequence_1", "saved-maps-gt")
    gt_pose_path = os.path.join(replica_semantic_root, scene_id_, "Sequence_1", "traj_w_c.txt")

    gt_map = Pointclouds.load_pointcloud_from_h5(gt_pc_path)
    gt_poses = np.loadtxt(gt_pose_path)
    gt_poses = torch.from_numpy(gt_poses.reshape(-1, 4, 4)).float()

    gt_xyz = gt_map.points_padded[0]
    gt_embedding = gt_map.embeddings_padded[0]
    gt_class = gt_embedding.argmax(dim=1)
    gt_class = class_all2existing[gt_class]

    gt_xyz = gt_xyz @ gt_poses[0, :3, :3].t() + gt_poses[0, :3, 3]

    return gt_xyz, gt_class
