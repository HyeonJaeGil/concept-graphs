from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from conceptgraph.dataset import scannet_semseg as scannet_ds
from conceptgraph.eval.semseg_common import (
    build_pred_tensors,
    compute_keep_index,
    compute_scene_confmatrix,
    load_latest_result,
    load_objects_from_results,
    resample_to_reference,
    save_labeled_point_cloud,
)
from conceptgraph.eval.llm_text_label_classifier import LLMTextLabelClassifier
from conceptgraph.eval.clip_text_label_classifier import ClipTextLabelClassifier
from conceptgraph.utils.eval import compute_metrics
from conceptgraph.utils.vis import get_random_colors


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--scannet_root", type=str, default="/path/to/scannet")
    p.add_argument(
        "--pred_exp_name",
        type=str,
        default="full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_masksub",
        help="exp name under <scannet_root>/<scene_id>/pcd_saves/<pred_exp_name>*.pkl.gz",
    )
    p.add_argument(
        "--scene_id",
        type=str,
        default=["all"],
        nargs="+",
        help="ScanNet scene ids to evaluate on. Use 'all' to evaluate default list.",
    )

    p.add_argument("--save", action="store_true", help="Save CSV and confusion matrices to ./results/")
    p.add_argument(
        "--out_dir",
        type=str,
        default="./results/scannet_semseg",
        help="Output directory for saved results.",
    )
    p.add_argument(
        "--pcd_save_dir",
        type=str,
        default="",
        help="Optional: if set, save per-scene labeled GT/pred point clouds as <scene_id>_*.ply.",
    )

    p.add_argument(
        "--n_exclude",
        type=int,
        default=1,
        choices=[1, 4, 6],
        help='''Number of classes to exclude:
        1: exclude "otherfurniture"
        4: exclude "otherfurniture", "floor", "wall", "ceiling"
        6: exclude "otherfurniture", "floor", "wall", "ceiling", "door", "window"
        ''',
    )
    p.add_argument("--gt_class_only", action="store_true", help="Evaluate only classes present in GT per scene.")
    p.add_argument(
        "--classify_label_with",
        type=str,
        default="clip",
        choices=["clip", "eva", "llm"],
        help="Method to classify object labels.",
    )
    p.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="OpenAI chat model when classify_label_with=llm.")
    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument(
        "--export_gt",
        action="store_true",
        help="If set, exports gt.npz for each specified scene from raw ScanNet data.",
    )

    return p


def main(args: argparse.Namespace) -> None:
    selected_scene_ids = scannet_ds.resolve_scene_ids(args.scene_id)

    if args.export_gt:
        export_scene_ids = selected_scene_ids
        if "all" in args.scene_id:
            export_scene_ids = sorted(
                [p.name for p in Path(args.scannet_root).glob("scene*_??") if p.is_dir()]
            )
        for sid in export_scene_ids:
            scan_path = os.path.join(args.scannet_root, sid)
            out_path = os.path.join(args.scannet_root, sid, "gt.npz")
            label_map_path = os.path.join(args.scannet_root, "scannetv2-labels.combined.tsv")
            print(f"Exporting GT for {sid}: {scan_path} -> {out_path}")
            scannet_ds.export_scannet_gt_npz(scan_path, out_path, label_map_path)

    class_all2existing, class_names = scannet_ds.get_class_setup()
    exclude_class = scannet_ds.resolve_exclude_indices(args.n_exclude, class_names)
    print("Excluding classes:", [(int(i), class_names[int(i)]) for i in exclude_class])
    print("Evaluating scenes:", selected_scene_ids)

    if args.classify_label_with in ["clip", "eva"]:
        classifier = ClipTextLabelClassifier(
            classify_label_with=args.classify_label_with,
            device=args.device,
        )
    elif args.classify_label_with == "llm":
        classifier = LLMTextLabelClassifier(
            llm_model=args.llm_model,
            device=args.device,
        )
    else:
        raise ValueError(f"Unsupported classify_label_with: {args.classify_label_with}")

    conf_matrices: Dict[str, Dict] = {}
    conf_all = None
    results_rows = []

    for sid in selected_scene_ids:
        print(f"\n=== Evaluating {sid} ===")

        gt_xyz, gt_class = scannet_ds.load_gt(
            scannet_root=args.scannet_root,
            scene_id=sid,
            class_all2existing=class_all2existing,
        )

        keep_index, ignore_index = compute_keep_index(
            class_names=class_names,
            gt_class=gt_class,
            base_ignore_index=exclude_class,
            gt_class_only=args.gt_class_only,
        )
        print(f"[{sid}] evaluating {len(keep_index)} classes:", [(int(i), class_names[int(i)]) for i in keep_index])

        results, _ = load_latest_result(args.scannet_root, sid, args.pred_exp_name)
        objects = load_objects_from_results(results)
        
        # for each object, copy the "description" field into "label" field
        for obj in objects:
            if "description" in obj and obj["description"] not in (None, ""):
                obj["label"] = obj["description"]

        object_class = classifier.classify_object_list(
            objects=objects,
            class_names=class_names,
            ignore_index=ignore_index,
            print_assignments=True,
        )

        pred_xyz, pred_class, pred_color = build_pred_tensors(objects, object_class)

        if args.pcd_save_dir:
            os.makedirs(args.pcd_save_dir, exist_ok=True)
            class_colors = get_random_colors(len(class_names), seed=42)
            save_labeled_point_cloud(
                pred_xyz,
                pred_class,
                class_colors,
                os.path.join(args.pcd_save_dir, f"{sid}_pred_labeled_before_resample.ply"),
            )

        slam_path = os.path.join(args.scannet_root, sid, "rgb_cloud")
        if os.path.exists(slam_path) and os.path.isdir(slam_path):
            pred_xyz, pred_class, pred_color = resample_to_reference(
                pred_xyz=pred_xyz,
                pred_class=pred_class,
                pred_color=pred_color,
                slam_h5_path=slam_path,
                device=args.device,
            )

        conf = compute_scene_confmatrix(
            pred_xyz=pred_xyz,
            pred_class=pred_class,
            gt_xyz=gt_xyz,
            gt_class=gt_class,
            keep_index=keep_index,
            ignore_index=ignore_index,
            class_names=class_names,
            device=args.device,
        )

        if args.pcd_save_dir:
            class_colors = get_random_colors(len(class_names), seed=42)
            save_labeled_point_cloud(
                gt_xyz,
                gt_class,
                class_colors,
                os.path.join(args.pcd_save_dir, f"{sid}_gt_labeled.ply"),
            )
            save_labeled_point_cloud(
                pred_xyz,
                pred_class,
                class_colors,
                os.path.join(args.pcd_save_dir, f"{sid}_pred_labeled.ply"),
            )

        conf_cpu = conf.detach().cpu()
        conf_all = conf_cpu if conf_all is None else (conf_all + conf_cpu)

        conf_matrices[sid] = {"conf_matrix": conf_cpu, "keep_index": keep_index}

        conf_k = conf_cpu[keep_index, :][:, keep_index]
        keep_names = [class_names[int(i)] for i in keep_index]
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
