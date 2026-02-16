"""
eval_scannet_semseg.py

ScanNet 3D semantic segmentation evaluation for ConceptGraphs-style reconstructions
(and optionally ConceptFusion / OpenFusion / OpenMask3D outputs).

This script is a consolidated/cleaned adaptation of the user's provided Replica evaluator,
plus the previously uploaded:
  - eval_script.py (ScanNet evaluation logic / class lists)
  - gt_generation.py (optional GT .npz export from raw ScanNet scene)

Expected GT format (per scene):
  <scannet_root>/<scene_id>/gt.npz  with arrays:
    - points: (N,3) float
    - labels: (N,) int (NYU40 IDs)

Predictions:
  - ConceptGraphs: <scannet_root>/<scene_id>/pcd_saves/<pred_exp_name>*.pkl.gz (a .pkl.gz containing results['objects'])
  - ConceptFusion: <scannet_root>/<scene_id>-map  (gradslam Pointclouds h5)
  - OpenMask3D:    <scannet_root>/<scene_id>/openmask3d.npz (points, features)
  - OpenFusion:    <scannet_root>/<scene_id>/openfusion_vlfusion.npz (points, labels)

Outputs:
  Prints per-scene and aggregated metrics (mIoU, mean precision/recall/F1, freq-weighted IoU).
  Optionally saves CSV and confusion matrices.

Notes:
  - By default excludes "otherfurniture" (NYU40 id 39).
  - Optionally evaluates only classes present in GT for each scene (recommended).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import open_clip
import open3d as o3d

# Optional (for fair resampling to a reference reconstruction)
try:
    # preferred (already used in earlier eval_script)
    from pytorch3d.ops import knn_points as _knn_points
except Exception:
    _knn_points = None

try:
    # some codebases use chamferdist
    from chamferdist.chamfer import knn_points as _knn_points_chamfer
except Exception:
    _knn_points_chamfer = None

from gradslam.structures.pointclouds import Pointclouds

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.eval import compute_confmatrix, compute_pred_gt_associations, compute_metrics
from conceptgraph.utils.vis import get_random_colors


# -------------------------
# ScanNet NYU40 class setup
# -------------------------
SCANNET_EXISTING_CLASSES_NYU40ID: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 14, 16, 24, 28, 33, 34, 36, 39
]
# From ScanNet NYU40 mapping (subset used in many baselines)
NYU40ID_ID_TO_CLASSES: Dict[int, str] = {
    1:  "wall",
    2:  "floor",
    3:  "cabinet",
    4:  "bed",
    5:  "chair",
    6:  "sofa",
    7:  "table",
    8:  "door",
    9:  "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    14: "desk",
    16: "curtain",
    24: "refridgerator",
    28: "shower curtain",
    33: "toilet",
    34: "sink",
    36: "bathtub",
    39: "otherfurniture",
}


# -------------------------
# Optional: GT export helper
# (adapted from gt_generation.py)
# -------------------------
def _represents_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping_tsv(filename: str, label_from: str = "raw_category", label_to: str = "nyu40id") -> Dict:
    assert os.path.isfile(filename), f"Label map file not found: {filename}"
    mapping: Dict = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # convert keys to int if needed
    if mapping and _represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def read_aggregation_json(filename: str) -> Tuple[Dict[int, List[int]], Dict[str, List[int]]]:
    assert os.path.isfile(filename), f"Aggregation file not found: {filename}"
    object_id_to_segs: Dict[int, List[int]] = {}
    label_to_segs: Dict[str, List[int]] = {}
    with open(filename) as f:
        data = json.load(f)
        for g in data["segGroups"]:
            object_id = g["objectId"] + 1  # 1-indexed instances
            label = g["label"]
            segs = g["segments"]
            object_id_to_segs[object_id] = segs
            label_to_segs.setdefault(label, []).extend(segs)
    return object_id_to_segs, label_to_segs


def read_segmentation_json(filename: str) -> Tuple[Dict[int, List[int]], int]:
    assert os.path.isfile(filename), f"Segmentation file not found: {filename}"
    seg_to_verts: Dict[int, List[int]] = {}
    with open(filename) as f:
        data = json.load(f)
        seg_indices = data["segIndices"]
        num_verts = len(seg_indices)
        for vi, seg_id in enumerate(seg_indices):
            seg_to_verts.setdefault(seg_id, []).append(vi)
    return seg_to_verts, num_verts


def read_mesh_vertices_ply(filename: str) -> np.ndarray:
    # Local import to avoid hard dependency unless exporting GT
    try:
        from plyfile import PlyData
    except Exception as e:
        raise ImportError("Please install plyfile to export GT: pip install plyfile") from e

    assert os.path.isfile(filename), f"PLY mesh file not found: {filename}"
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros((num_verts, 3), dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
    return vertices


def export_scannet_gt_npz(scan_path: str, output_file: str, label_map_file: str) -> None:
    """
    Export per-vertex NYU40 semantic labels for a ScanNet scene to a compressed NPZ.
    Output contains:
      - points: (N,3) mesh vertices
      - labels: (N,) nyu40id per vertex (0 = unannotated)
    """
    scan_path = os.path.abspath(scan_path)
    scan_name = os.path.basename(scan_path)

    mesh_file = os.path.join(scan_path, f"{scan_name}_vh_clean_2.ply")
    agg_file = os.path.join(scan_path, f"{scan_name}.aggregation.json")
    seg_file = os.path.join(scan_path, f"{scan_name}_vh_clean_2.0.010000.segs.json")

    label_map = read_label_mapping_tsv(label_map_file, label_from="raw_category", label_to="nyu40id")
    vertices = read_mesh_vertices_ply(mesh_file)
    _, label_to_segs = read_aggregation_json(agg_file)
    seg_to_verts, num_verts = read_segmentation_json(seg_file)

    label_ids = np.zeros((num_verts,), dtype=np.uint32)  # 0: unannotated
    for label, segs in label_to_segs.items():
        if label not in label_map:
            # leave as 0 if unmapped
            continue
        nyu40id = label_map[label]
        for seg in segs:
            verts_idx = seg_to_verts.get(seg, [])
            label_ids[verts_idx] = nyu40id

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    np.savez_compressed(output_file, points=vertices, labels=label_ids)


# -------------------------
# Evaluation
# -------------------------
def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    
    p.add_argument("--scannet_root", type=str, default="/path/to/scannet",)

    p.add_argument("--pred_exp_name", type=str, default="full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_masksub",
                   help="exp name under <scannet_root>/<scene_id>/pcd_saves/<pred_exp_name>*.pkl.gz")

    # Scene selection
    p.add_argument("--scene_id", type=str, default=["all"], nargs="+",
                   help="ScanNet scene ids to evaluate on. Use 'all' to evaluate default list.")
    # Output
    p.add_argument("--save", action="store_true", help="Save CSV and confusion matrices to ./results/")
    p.add_argument("--out_dir", type=str, default="./results/scannet_semseg", help="Output directory for saved results.")
    p.add_argument("--pcd_save_dir", type=str, default="",
                   help="Optional: if set, save per-scene labeled GT/pred point clouds as <scene_id>_*.ply.")

    # Class/model
    p.add_argument("--n_exclude", type=int, default=1, choices=[1, 4, 6],
                   help='''Number of classes to exclude:
                   1: exclude "otherfurniture"
                   4: exclude "otherfurniture", "floor", "wall", "ceiling"
                   6: exclude "otherfurniture", "floor", "wall", "ceiling", "door", "window"
                   ''')
    p.add_argument("--gt_class_only", action="store_true", help="Evaluate only classes present in GT per scene.")
    p.add_argument("--clip_model", type=str, default="default", choices=["default", "eva"],
                   help="Text encoder backbone for zero-shot CLIP classification.")
    p.add_argument("--device", type=str, default="cuda:0")

    # Optional: export GT
    p.add_argument("--export_gt", action="store_true",
                   help="If set, exports gt.npz for each specified scene from raw ScanNet data.")

    return p


def _load_scannet_gt(scannet_root: str, scene_id: str, class_all2existing: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gt_path = os.path.join(scannet_root, scene_id, "gt.npz")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT not found: {gt_path}")
    gt_file = np.load(gt_path)
    gt_xyz = torch.tensor(gt_file["points"], dtype=torch.float32)
    gt_class_nyu40 = torch.tensor(gt_file["labels"].astype(np.int64))
    gt_class = class_all2existing[gt_class_nyu40]  # map to existing subset indices (or -1)
    return gt_xyz, gt_class


def _compute_keep_index(
    class_names: List[str],
    gt_class: torch.Tensor,
    base_ignore_index: np.ndarray,
    gt_class_only: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    all_class_index = np.arange(len(class_names))
    ignore_index = np.asarray(base_ignore_index).copy()

    if gt_class_only:
        existing_index = gt_class.unique().cpu().numpy()
        existing_index = existing_index[existing_index >= 0]  # drop -1
        non_existing_index = np.setdiff1d(all_class_index, existing_index)
        ignore_index = np.append(ignore_index, non_existing_index)

    keep_index = np.setdiff1d(all_class_index, ignore_index)
    return keep_index, ignore_index


def _build_class_text_features(
    class_names: List[str],
    model_name: str,
    device: str,
) -> torch.Tensor:
    """
    Builds CLIP text embeddings for 'an image of <class>'.
    """
    if model_name == "default":
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
    elif model_name == "eva":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k"
        )
        tokenizer = open_clip.get_tokenizer(
            "hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k"
        )
    else:
        raise ValueError(f"Unsupported clip_model: {model_name}")

    clip_model = clip_model.to(device)
    prompts = [f"an image of {c}" for c in class_names]
    text = tokenizer(prompts).to(device)
    with torch.no_grad():
        class_feats = clip_model.encode_text(text)
        class_feats = class_feats / class_feats.norm(dim=-1, keepdim=True)
    return class_feats


def resample_to_reference(
    pred_xyz: torch.Tensor,
    pred_class: torch.Tensor,
    pred_color: Optional[torch.Tensor],
    slam_h5_path: str,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Resample (pred_xyz, pred_class[, pred_color]) onto the points of a reference reconstruction.
    For each ref point, pick nearest neighbor in pred_xyz and copy its label/color.
    """
    if _knn_points is None and _knn_points_chamfer is None:
        raise ImportError("Neither pytorch3d.ops.knn_points nor chamferdist.knn_points is available")

    try:
        slam_pc = Pointclouds.load_pointcloud_from_h5(slam_h5_path)
        slam_xyz = slam_pc.points_padded[0]  # (N,3)
    except Exception as e:
        # fallback and skip resampling
        print(f"Warning: failed to load Pointclouds from {slam_h5_path}: {e}. Skipping resampling.")
        return pred_xyz, pred_class, pred_color
    

    # choose implementation
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

    new_pred_xyz = slam_xyz
    new_pred_class = pred_class[idx_slam_to_pred]
    new_pred_color = pred_color[idx_slam_to_pred] if pred_color is not None else None
    return new_pred_xyz, new_pred_class, new_pred_color


def _save_labeled_point_cloud(
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


def load_predictions(
    args: argparse.Namespace,
    scene_id: str,
    class_names: List[str],
    class_feats: Optional[torch.Tensor],
    ignore_index: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      pred_xyz: (Np,3) float
      pred_class: (Np,) long (class index in [0, num_classes))
      pred_color: (Np,3) float in [0,1] if available else None
    """
    device = args.device

    # ConceptGraphs style: load objects from pkl.gz
    scene_root = os.path.join(args.scannet_root, scene_id)

    # Load the predicted point cloud
    result_paths = glob.glob(
        os.path.join(
            args.scannet_root, scene_id, "pcd_saves", 
            f"{args.pred_exp_name}*.pkl.gz"
        )
    )
    if len(result_paths) == 0:
        raise ValueError(f"No result found for {scene_id} with {args.pred_exp_name}")
        
    # Get the newest result over result_paths
    result_paths = sorted(result_paths, key=os.path.getmtime)
    result_path = result_paths[-1]
    print(f"Loading mapping result from {result_path}")


    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    objects = MapObjectList()
    objects.load_serializable(results["objects"])

    object_feats = objects.get_stacked_values_torch("clip_ft").to(device)
    if object_feats.ndim == 3 and object_feats.shape[1] == 1:
        object_feats = object_feats[:, 0, :]
    object_feats = object_feats / object_feats.norm(dim=-1, keepdim=True)

    object_class_sim = object_feats @ class_feats.T
    object_class_sim[:, ignore_index] = -1e10
    object_class = object_class_sim.argmax(dim=-1).detach().cpu()  # per object class id

    print("Assigned object classes:")
    for i in range(len(objects)):
        class_id = int(object_class[i].item())
        class_name = class_names[class_id]
        orig_label = objects[i]["label"] if "label" in objects[i] else "N/A"
        print(f"Object {i}: (orig: {orig_label}) class {class_id} - {class_name}")

    pred_xyz_list = []
    pred_color_list = []
    pred_class_list = []
    for i in range(len(objects)):
        obj_pcd = objects[i]["pcd"]
        pts = np.asarray(obj_pcd.points)
        pred_xyz_list.append(pts)
        # color may not exist / may be empty
        cols = np.asarray(obj_pcd.colors) if hasattr(obj_pcd, "colors") else None
        if cols is not None and len(cols) == len(pts):
            pred_color_list.append(cols)
        pred_class_list.append(np.ones(len(pts), dtype=np.int64) * int(object_class[i].item()))

    pred_xyz = torch.from_numpy(np.concatenate(pred_xyz_list, axis=0)).float()
    pred_class = torch.from_numpy(np.concatenate(pred_class_list, axis=0)).long()
    pred_color = None
    if len(pred_color_list) == len(pred_xyz_list) and len(pred_color_list) > 0:
        pred_color = torch.from_numpy(np.concatenate(pred_color_list, axis=0)).float()

    return pred_xyz, pred_class, pred_color


def eval_scannet_scene(
    scene_id: str,
    class_names: List[str],
    class_feats: torch.Tensor,
    args: argparse.Namespace,
    class_all2existing: torch.Tensor,
    base_ignore_index: np.ndarray,
    save_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    # Load GT
    gt_xyz, gt_class = _load_scannet_gt(args.scannet_root, scene_id, class_all2existing)

    # Keep/ignore indices
    keep_index, ignore_index = _compute_keep_index(
        class_names=class_names,
        gt_class=gt_class,
        base_ignore_index=base_ignore_index,
        gt_class_only=args.gt_class_only,
    )
    print(f"[{scene_id}] evaluating {len(keep_index)} classes:", [(i, class_names[i]) for i in keep_index])

    # Load predictions
    pred_xyz, pred_class, pred_color = load_predictions(
        args=args,
        scene_id=scene_id,
        class_names=class_names,
        class_feats=class_feats,
        ignore_index=ignore_index,
    )
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        class_colors = get_random_colors(len(class_names), seed=42)
        _save_labeled_point_cloud(
            pred_xyz,
            pred_class,
            class_colors,
            os.path.join(save_dir, f"{scene_id}_pred_labeled_before_resample.ply"),
        )

    # Optional resampling (onto a reference point cloud)
    slam_path = os.path.join(args.scannet_root, scene_id, "rgb_cloud")
    if os.path.exists(slam_path) and os.path.isdir(slam_path):
        pred_xyz, pred_class, pred_color = resample_to_reference(
            pred_xyz=pred_xyz,
            pred_class=pred_class,
            pred_color=pred_color,
            slam_h5_path=slam_path,
            device=args.device,
        )

    # Associate pred points to GT points (NN)
    idx_pred_to_gt, idx_gt_to_pred = compute_pred_gt_associations(
        pred_xyz.unsqueeze(0).to(args.device).contiguous().float(),
        gt_xyz.unsqueeze(0).to(args.device).contiguous().float(),
    )

    # Keep only pred points whose matched GT is in keep_index (and not -1)
    label_gt = gt_class[idx_pred_to_gt.detach().cpu()]
    pred_keep = torch.isin(label_gt, torch.from_numpy(keep_index)) & (label_gt >= 0)
    pred_class = pred_class[pred_keep]
    pred_xyz = pred_xyz[pred_keep]
    idx_pred_to_gt = idx_pred_to_gt[pred_keep]
    idx_gt_to_pred = None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        class_colors = get_random_colors(len(class_names), seed=42)
        _save_labeled_point_cloud(
            gt_xyz,
            gt_class,
            class_colors,
            os.path.join(save_dir, f"{scene_id}_gt_labeled.ply"),
        )
        _save_labeled_point_cloud(
            pred_xyz,
            pred_class,
            class_colors,
            os.path.join(save_dir, f"{scene_id}_pred_labeled.ply"),
        )

    confmatrix = compute_confmatrix(
        pred_class.to(args.device),
        gt_class.to(args.device),
        idx_pred_to_gt,
        idx_gt_to_pred,
        class_names,
    )

    # sanity: ignored rows/cols should be 0
    assert confmatrix.sum(0)[ignore_index].sum().item() == 0
    assert confmatrix.sum(1)[ignore_index].sum().item() == 0

    return confmatrix, keep_index


def main(args: argparse.Namespace) -> None:
    # Optionally export GT first
    if args.export_gt:
        scene_ids = args.scene_id
        if "all" in scene_ids:
            # best-effort: export for all directories matching scene*_*
            scene_ids = sorted([p.name for p in Path(args.scannet_root).glob("scene*_??") if p.is_dir()])
        for sid in scene_ids:
            scan_path = os.path.join(args.scannet_root, sid)
            out_path = os.path.join(args.scannet_root, sid, "gt.npz")
            label_map_path = os.path.join(args.scannet_root, "scannetv2-labels.combined.tsv")
            print(f"Exporting GT for {sid}: {scan_path} -> {out_path}")
            export_scannet_gt_npz(scan_path, out_path, label_map_path)

    # Build class maps
    class_all2existing = torch.ones(41, dtype=torch.long) * -1
    for i, nyu40id in enumerate(SCANNET_EXISTING_CLASSES_NYU40ID):
        class_all2existing[nyu40id] = i
    class_names = [NYU40ID_ID_TO_CLASSES[i] for i in SCANNET_EXISTING_CLASSES_NYU40ID]

    # Exclusions
    if args.n_exclude == 1:
        exclude_names = ["otherfurniture"]
    elif args.n_exclude == 4:
        exclude_names = ["otherfurniture", "floor", "wall", "ceiling"]
    elif args.n_exclude == 6:
        exclude_names = ["otherfurniture", "floor", "wall", "ceiling", "door", "window"]
    else:
        raise ValueError(f"Unsupported n_exclude: {args.n_exclude}")

    missing_exclude_names = [c for c in exclude_names if c not in class_names]
    if missing_exclude_names:
        print(
            "Warning: requested excluded classes not present in current class set:",
            missing_exclude_names,
        )
    exclude_class = np.array([class_names.index(c) for c in exclude_names if c in class_names], dtype=np.int64)
    print("Excluding classes:", [(int(i), class_names[int(i)]) for i in exclude_class])

    # Scene selection
    default_scenes = [
        "scene0011_00", "scene0030_00", "scene0086_00", "scene0389_00",
        "scene0222_00", "scene0378_00", "scene0046_00", "scene0435_00",
    ]
    if "all" in args.scene_id:
        selected_scene_ids = default_scenes
    else:
        selected_scene_ids = args.scene_id
    print("Evaluating scenes:", selected_scene_ids)

    # Class text feats
    class_feats = _build_class_text_features(
        class_names=class_names,
        model_name=args.clip_model,
        device=args.device,
    )

    conf_matrices: Dict[str, Dict] = {}
    conf_all = None

    results_rows = []
    for sid in selected_scene_ids:
        print(f"\n=== Evaluating {sid} ===")
        conf, keep_index = eval_scannet_scene(
            scene_id=sid,
            class_names=class_names,
            class_feats=class_feats,
            args=args,
            class_all2existing=class_all2existing,
            base_ignore_index=exclude_class,
            save_dir=args.pcd_save_dir if args.pcd_save_dir else None,
        )

        conf_cpu = conf.detach().cpu()
        conf_all = conf_cpu if conf_all is None else (conf_all + conf_cpu)

        conf_matrices[sid] = {"conf_matrix": conf_cpu, "keep_index": keep_index}

        # metrics for this scene
        conf_k = conf_cpu[keep_index, :][:, keep_index]
        keep_names = [class_names[i] for i in keep_index]
        mdict = compute_metrics(conf_k, keep_names)
        row = {
            "scene_id": sid,
            "miou": float(mdict["miou"] * 100.0),
            "mrecall": float(np.mean(mdict["recall"]) * 100.0),
            "mprecision": float(np.mean(mdict["precision"]) * 100.0),
            "mf1score": float(np.mean(mdict["f1score"]) * 100.0),
            "fmiou": float(mdict["fmiou"] * 100.0),
        }
        results_rows.append(row)

        print(
            f"[{sid}] mIoU {row['miou']:.2f} | mRecall {row['mrecall']:.2f} | "
            f"mPrec {row['mprecision']:.2f} | mF1 {row['mf1score']:.2f} | fIoU {row['fmiou']:.2f}"
        )

    # Aggregate
    conf_matrices["all"] = {
        "conf_matrix": conf_all,
        "keep_index": conf_all.sum(axis=1).nonzero().reshape(-1).numpy(),
    }
    keep_index = conf_matrices["all"]["keep_index"]
    conf_k = conf_all[keep_index, :][:, keep_index]
    keep_names = [class_names[int(i)] for i in keep_index]
    mdict = compute_metrics(conf_k, keep_names)
    row_all = {
        "scene_id": "all",
        "miou": float(mdict["miou"] * 100.0),
        "mrecall": float(np.mean(mdict["recall"]) * 100.0),
        "mprecision": float(np.mean(mdict["precision"]) * 100.0),
        "mf1score": float(np.mean(mdict["f1score"]) * 100.0),
        "fmiou": float(mdict["fmiou"] * 100.0),
    }
    results_rows.append(row_all)

    print(
        f"\n=== Aggregate ===\n[all] mIoU {row_all['miou']:.2f} | mRecall {row_all['mrecall']:.2f} | "
        f"mPrec {row_all['mprecision']:.2f} | mF1 {row_all['mf1score']:.2f} | fIoU {row_all['fmiou']:.2f}"
    )

    if args.save:
        os.makedirs(args.out_dir, exist_ok=True)
        df = pd.DataFrame(results_rows)
        exp_basename = os.path.basename(args.pred_exp_name)
        csv_path = os.path.join(args.out_dir, f"scannet_semseg_results_{exp_basename}.csv")
        df.to_csv(csv_path, index=False)

        pkl_path = os.path.join(args.out_dir, "scannet_semseg_conf_matrices.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(conf_matrices, f)

        print(f"Saved results to:\n  {csv_path}\n  {pkl_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
