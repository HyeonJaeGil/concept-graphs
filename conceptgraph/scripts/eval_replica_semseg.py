from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from conceptgraph.dataset import replica_semseg as replica_ds
from conceptgraph.eval.semseg_common import (
    append_replica_background_objects,
    build_pred_tensors,
    compute_keep_index,
    compute_scene_confmatrix,
    load_latest_result,
    load_objects_from_results,
    resample_to_reference,
)
from conceptgraph.eval.llm_text_label_classifier import LLMTextLabelClassifier
from conceptgraph.eval.clip_text_label_classifier import ClipTextLabelClassifier
from conceptgraph.utils.eval import compute_metrics


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replica_root",
        type=Path,
        default=Path("~/rdata/Replica/").expanduser(),
    )
    parser.add_argument(
        "--replica_semantic_root",
        type=Path,
        default=Path("~/rdata/Replica-semantic/").expanduser(),
    )
    parser.add_argument(
        "--pred_exp_name",
        type=str,
        default="ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_masksub",
        help="The name of cfslam experiment. Will be used to load the result.",
    )
    parser.add_argument(
        "--n_exclude",
        type=int,
        default=1,
        choices=[1, 4, 6],
        help='''Number of classes to exclude:
        1: exclude "other"
        4: exclude "other", "floor", "wall", "ceiling"
        6: exclude "other", "floor", "wall", "ceiling", "door", "window"
        ''',
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default="all",
        nargs="+",
        help="The replica scene id to evaluate on. Default to all scenes.",
    )
    parser.add_argument("--save", action="store_true", help="Whether to save the results to a csv file.")
    parser.add_argument(
        "--classify_label_with",
        type=str,
        default="clip",
        choices=["clip", "eva", "llm"],
        help="Method to classify object labels.",
    )
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="OpenAI chat model when classify_label_with=llm.")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def main(args: argparse.Namespace) -> None:
    class_all2existing, class_names = replica_ds.get_class_setup()
    exclude_class = replica_ds.resolve_exclude_indices(args.n_exclude, class_names)
    print("Excluding classes:", [(int(i), class_names[int(i)]) for i in exclude_class])

    selected_scene_ids, selected_scene_ids_ = replica_ds.resolve_scene_ids(args.scene_id)
    if "all" in args.scene_id:
        print("Evaluating on all scenes:", selected_scene_ids)
    else:
        print("Evaluating on specified scenes:", selected_scene_ids)

    if args.classify_label_with in ["clip", "eva"]:
        classifier = ClipTextLabelClassifier(
            classify_label_with=args.classify_label_with,
            device=args.device,
            clip_device="cpu",
        )
    elif args.classify_label_with == "llm":
        classifier = LLMTextLabelClassifier(
            llm_model=args.llm_model,
            device=args.device,
        )
    else:
        raise ValueError(f"Unsupported classify_label_with: {args.classify_label_with}")

    conf_matrices = {}
    conf_matrix_all = 0

    for scene_id, scene_id_ in zip(selected_scene_ids, selected_scene_ids_):
        print("Evaluating on:", scene_id, scene_id_)

        gt_xyz, gt_class = replica_ds.load_gt(
            replica_semantic_root=args.replica_semantic_root,
            scene_id_=scene_id_,
            class_all2existing=class_all2existing,
        )

        keep_index, ignore_index = compute_keep_index(
            class_names=class_names,
            gt_class=gt_class,
            base_ignore_index=exclude_class,
            gt_class_only=True,
        )
        print(
            f"{len(keep_index)} classes remains. They are:",
            [(int(i), class_names[int(i)]) for i in keep_index],
        )

        results, _ = load_latest_result(str(args.replica_root), scene_id, args.pred_exp_name)
        objects = load_objects_from_results(results)
        
        # if "description" field exists for an object and is not empty, copy it into "label" field
        for obj in objects:
            if "description" in obj and obj["description"] not in (None, ""):
                obj["label"] = obj["description"]

        object_class = classifier.classify_object_list(
            objects=objects,
            class_names=class_names,
            ignore_index=ignore_index,
            print_assignments=True,
        )

        if args.n_exclude == 1:
            objects, object_class = append_replica_background_objects(
                results=results,
                objects=objects,
                object_class=object_class,
                class_names=class_names,
            )

        pred_xyz, pred_class, pred_color = build_pred_tensors(objects, object_class)

        slam_path = os.path.join(args.replica_root, scene_id, "rgb_cloud")
        pred_xyz, pred_class, pred_color = resample_to_reference(
            pred_xyz=pred_xyz,
            pred_class=pred_class,
            pred_color=pred_color,
            slam_h5_path=slam_path,
            device=args.device,
        )

        conf_matrix = compute_scene_confmatrix(
            pred_xyz=pred_xyz,
            pred_class=pred_class,
            gt_xyz=gt_xyz,
            gt_class=gt_class,
            keep_index=keep_index,
            ignore_index=ignore_index,
            class_names=class_names,
            device=args.device,
        ).detach().cpu()

        conf_matrix_all += conf_matrix
        conf_matrices[scene_id] = {
            "conf_matrix": conf_matrix,
            "keep_index": keep_index,
        }

    conf_matrices["all"] = {
        "conf_matrix": conf_matrix_all,
        "keep_index": conf_matrix_all.sum(axis=1).nonzero().reshape(-1),
    }

    results = []
    for scene_id, res in conf_matrices.items():
        conf_matrix = res["conf_matrix"]
        keep_index = res["keep_index"]
        conf_matrix = conf_matrix[keep_index, :][:, keep_index]
        keep_class_names = [class_names[int(i)] for i in keep_index]

        mdict = compute_metrics(conf_matrix, keep_class_names)
        results.append(
            {
                "scene_id": scene_id,
                "miou": mdict["miou"] * 100.0,
                "mrecall": np.mean(mdict["recall"]) * 100.0,
                "mprecision": np.mean(mdict["precision"]) * 100.0,
                "mf1score": np.mean(mdict["f1score"]) * 100.0,
                "fmiou": mdict["fmiou"] * 100.0,
            }
        )

    for res in results:
        print(
            f"Scene {res['scene_id']}: mIoU: {res['miou']:.2f}, "
            f"mRecall: {res['mrecall']:.2f}, mPrecision: {res['mprecision']:.2f}, "
            f"mF1-score: {res['mf1score']:.2f}, fmiou: {res['fmiou']:.2f}"
        )

    if args.save:
        df_result = pd.DataFrame(results)

        save_path = "./results/%s/replica_ex%d_results.csv" % (
            args.pred_exp_name,
            args.n_exclude,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_result.to_csv(save_path, index=False)

        save_path = "./results/%s/replica_ex%d_conf_matrices.pkl" % (
            args.pred_exp_name,
            args.n_exclude,
        )
        pickle.dump(conf_matrices, open(save_path, "wb"))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
