import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from gradslam.structures.pointclouds import Pointclouds

from conceptgraph.dataset.replica_constants import (
    REPLICA_CLASSES,
    REPLICA_EXISTING_CLASSES,
    REPLICA_SCENE_IDS,
    REPLICA_SCENE_IDS_,
)
from conceptgraph.utils.vis import get_random_colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Replica semantic ground-truth point cloud."
    )
    parser.add_argument(
        "--replica_semantic_root",
        type=Path,
        required=True,
        help="Root to Replica-semantic data (unzipped GT).",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        required=True,
        choices=REPLICA_SCENE_IDS,
        help="Scene ID (e.g., room0).",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=None,
        help="Optional path to save the colored GT point cloud.",
    )
    return parser.parse_args()


def load_gt_pointcloud(replica_semantic_root: Path, scene_id: str):
    # map REPLICA_CLASSES to REPLICA_EXISTING_CLASSES
    class_all2existing = torch.ones(len(REPLICA_CLASSES)).long() * -1
    for i, c in enumerate(REPLICA_EXISTING_CLASSES):
        class_all2existing[c] = i

    scene_idx = REPLICA_SCENE_IDS.index(scene_id)
    scene_id_ = REPLICA_SCENE_IDS_[scene_idx]

    gt_pc_path = replica_semantic_root / scene_id_ / "Sequence_1" / "saved-maps-gt"
    gt_pose_path = replica_semantic_root / scene_id_ / "Sequence_1" / "traj_w_c.txt"

    gt_map = Pointclouds.load_pointcloud_from_h5(gt_pc_path)
    gt_poses = np.loadtxt(gt_pose_path).reshape(-1, 4, 4)

    gt_xyz = gt_map.points_padded[0]  # (N,3) torch tensor
    gt_embedding = gt_map.embeddings_padded[0]  # (N,num_classes) torch tensor
    gt_class = gt_embedding.argmax(dim=1)  # (N,)
    gt_class = class_all2existing[gt_class].numpy()  # map to existing classes

    # Transform into world frame using the first GT pose for consistency with eval script.
    first_pose = gt_poses[0]
    gt_xyz = gt_xyz.numpy() @ first_pose[:3, :3].T + first_pose[:3, 3]

    return gt_xyz, gt_class


def main():
    args = parse_args()

    gt_xyz, gt_class = load_gt_pointcloud(args.replica_semantic_root, args.scene_id)

    num_classes = len(REPLICA_EXISTING_CLASSES)
    class_colors = get_random_colors(num_classes)  # (num_classes,3) in [0,1]
    colors = class_colors[gt_class]

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_xyz)
    gt_pcd.colors = o3d.utility.Vector3dVector(colors)

    if args.save_path is not None:
        o3d.io.write_point_cloud(str(args.save_path), gt_pcd)
        print(f"Saved GT point cloud to {args.save_path}")
    else:
        o3d.visualization.draw_geometries([gt_pcd])


if __name__ == "__main__":
    main()
