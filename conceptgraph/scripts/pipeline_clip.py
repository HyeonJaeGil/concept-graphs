import os
import argparse
import pickle
import gzip
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import trange
import open_clip
from collections import namedtuple
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.model_utils import compute_clip_features
from conceptgraph.scripts.pipeline_common import *


def dict_to_namedtuple(d):
    return namedtuple('x', d.keys())(*d.values())
    
def main_pipeline_clip():
    parser = get_parser()
    args = parser.parse_args()
    if not is_previous_pipeline_done(args, "clip"):
        print("Please run detection_segmentation pipeline first.")
        return
    default_path = get_default_path(args)
    
    # Initialize CLIP model and preprocess.
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    clip_model = clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    
    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=args.dataset_config,
        start=args.start,
        end=args.end,
        stride=args.stride,
        basedir=args.dataset_root,
        sequence=args.scene_id,
        desired_height=args.desired_height,
        desired_width=args.desired_width,
        device="cpu",
        dtype=torch.float,
    )
    
    for idx in trange(len(dataset)):
        color_path = Path(dataset.color_paths[idx])

        final_save_path = Path(default_path["final_results"]) / color_path.name
        final_save_path = final_save_path.with_suffix(".pkl.gz")
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)

        # Load detection results.
        detection_result_file = default_path["detection_results"] / f"{color_path.stem}_detection.pkl.gz"
        if not detection_result_file.exists():
            print(f"Detection result not found: {detection_result_file}")
            exit()
        with gzip.open(detection_result_file, "rb") as f:
            detection_result = pickle.load(f)
        
        detection_result_nt = dict_to_namedtuple(detection_result)
        if len(detection_result_nt.class_id) == 0:
            image_crops, image_feats, text_feats = [], [], []
        else:
            image = np.array(Image.open(color_path).convert("RGB"))[:, :, ::-1]
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            classes = detection_result.get("classes", ["item"])
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detection_result_nt, clip_model, clip_preprocess, clip_tokenizer, classes, args.device
            )
        
        # Save the Final features
        clip_results = {
            "classes": classes,
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats
        }
        results = {**clip_results, **detection_result}
        with gzip.open(final_save_path, "wb") as f:
            pickle.dump(results, f)
    
if __name__ == "__main__":
    main_pipeline_clip()
