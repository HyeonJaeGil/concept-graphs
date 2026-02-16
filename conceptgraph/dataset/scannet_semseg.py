from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

SCANNET_EXISTING_CLASSES_NYU40ID: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 14, 16, 24, 28, 33, 34, 36, 39,
]

NYU40ID_ID_TO_CLASSES: Dict[int, str] = {
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
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

DEFAULT_SCENES: List[str] = [
    "scene0011_00",
    "scene0030_00",
    "scene0086_00",
    "scene0389_00",
    "scene0222_00",
    "scene0378_00",
    "scene0046_00",
    "scene0435_00",
]


def get_class_setup() -> Tuple[torch.Tensor, List[str]]:
    class_all2existing = torch.ones(41, dtype=torch.long) * -1
    for i, nyu40id in enumerate(SCANNET_EXISTING_CLASSES_NYU40ID):
        class_all2existing[nyu40id] = i
    class_names = [NYU40ID_ID_TO_CLASSES[i] for i in SCANNET_EXISTING_CLASSES_NYU40ID]
    return class_all2existing, class_names


def resolve_exclude_indices(n_exclude: int, class_names: List[str]) -> np.ndarray:
    if n_exclude == 1:
        exclude_names = ["otherfurniture"]
    elif n_exclude == 4:
        exclude_names = ["otherfurniture", "floor", "wall", "ceiling"]
    elif n_exclude == 6:
        exclude_names = ["otherfurniture", "floor", "wall", "ceiling", "door", "window"]
    else:
        raise ValueError(f"Unsupported n_exclude: {n_exclude}")

    missing_exclude_names = [c for c in exclude_names if c not in class_names]
    if missing_exclude_names:
        print(
            "Warning: requested excluded classes not present in current class set:",
            missing_exclude_names,
        )

    return np.array([class_names.index(c) for c in exclude_names if c in class_names], dtype=np.int64)


def resolve_scene_ids(scene_ids: List[str]) -> List[str]:
    if "all" in scene_ids:
        return DEFAULT_SCENES
    return scene_ids


def load_gt(
    scannet_root: str,
    scene_id: str,
    class_all2existing: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gt_path = os.path.join(scannet_root, scene_id, "gt.npz")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT not found: {gt_path}")

    gt_file = np.load(gt_path)
    gt_xyz = torch.tensor(gt_file["points"], dtype=torch.float32)
    gt_class_nyu40 = torch.tensor(gt_file["labels"].astype(np.int64))
    gt_class = class_all2existing[gt_class_nyu40]
    return gt_xyz, gt_class


def _represents_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping_tsv(
    filename: str,
    label_from: str = "raw_category",
    label_to: str = "nyu40id",
) -> Dict:
    assert os.path.isfile(filename), f"Label map file not found: {filename}"
    mapping: Dict = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])

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
            object_id = g["objectId"] + 1
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
    scan_path = os.path.abspath(scan_path)
    scan_name = os.path.basename(scan_path)

    mesh_file = os.path.join(scan_path, f"{scan_name}_vh_clean_2.ply")
    agg_file = os.path.join(scan_path, f"{scan_name}.aggregation.json")
    seg_file = os.path.join(scan_path, f"{scan_name}_vh_clean_2.0.010000.segs.json")

    label_map = read_label_mapping_tsv(label_map_file, label_from="raw_category", label_to="nyu40id")
    vertices = read_mesh_vertices_ply(mesh_file)
    _, label_to_segs = read_aggregation_json(agg_file)
    seg_to_verts, num_verts = read_segmentation_json(seg_file)

    label_ids = np.zeros((num_verts,), dtype=np.uint32)
    for label, segs in label_to_segs.items():
        if label not in label_map:
            continue
        nyu40id = label_map[label]
        for seg in segs:
            verts_idx = seg_to_verts.get(seg, [])
            label_ids[verts_idx] = nyu40id

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    np.savez_compressed(output_file, points=vertices, labels=label_ids)
