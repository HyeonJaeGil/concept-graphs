import os
import sys
import argparse
from pathlib import Path
import torch
torch.set_grad_enabled(False)

# GSA_PATH should be set in advance
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
TAG2TEXT_PATH = os.path.join(GSA_PATH, "")
sys.path.append(GSA_PATH)
sys.path.append(TAG2TEXT_PATH)

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline 1: Tagging with RAM/Tag2Text")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Path to the dataset root")
    parser.add_argument("--dataset_config", type=str, required=True,
                        help="Path to the dataset configuration file")
    parser.add_argument("--scene_id", type=str, required=True, help="Scene identifier")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--desired_height", type=int, default=480, help="Desired height for images")
    parser.add_argument("--desired_width", type=int, default=640, help="Desired width for images")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--class_set", type=str, choices=["ram", "tag2text", "none"], default="ram",
                        help="Choose between using RAM or Tag2Text for tagging")
    parser.add_argument("--add_bg_classes", action="store_true", 
                        help="If set, add background classes (wall, floor, ceiling) to the class set. ")
    parser.add_argument("--accumu_classes", action="store_true",
                        help="if set, the class set will be accumulated over frames")
    parser.add_argument("--exp_suffix", type=str, default=None,
                        help="The suffix of the folder that the results will be saved to. ")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    return parser


def get_save_name(args: argparse.Namespace) -> str:
    save_name = args.class_set
    if args.add_bg_classes:
        save_name += "_withbg"
    if args.accumu_classes:
        save_name += "_allclasses"
    if args.exp_suffix:
        save_name += f"_{args.exp_suffix}"
    return save_name


def get_default_path(args: argparse.Namespace):
    save_name = get_save_name(args)
    return {
        "gsa_classes": args.dataset_root / args.scene_id / f"gsa_classes_{save_name}.json",
        "tagging_results": args.dataset_root / args.scene_id / f"tagging_results_{save_name}",
        "detection_results": args.dataset_root / args.scene_id / f"detection_results_{save_name}",
        "final_results": args.dataset_root / args.scene_id / f"gsa_detections_{save_name}"
    }


def is_previous_pipeline_done(args: argparse.Namespace, current_pipeline: str) -> bool:
    default_path = get_default_path(args)
    if current_pipeline == "tagging":
        return True
    elif current_pipeline == "detection_segmentation":
        return default_path["gsa_classes"].exists()
    elif current_pipeline == "clip":
        return default_path["detection_results"].exists()
    else:
        raise ValueError(f"Invalid pipeline name: {current_pipeline}, "
                         f"Please choose from ['tagging', 'detection_segmentation', 'clip']")
