import os
import argparse
import json
import pickle
import gzip
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
import supervision as sv
from tqdm import trange
from PIL import Image
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption
from conceptgraph.scripts.pipeline_common import *
from groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


def get_sam_predictor(device: str):
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    return SamPredictor(sam)

def get_sam_mask_generator(device: str):
    """Initialize SAM for dense segmentation using automatic mask generation."""
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=12,
        points_per_batch=144,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        min_mask_region_area=100,
    )
    return mask_generator

def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def get_sam_segmentation_dense(mask_generator: SamAutomaticMaskGenerator, image: np.ndarray):
    """
    Generate dense segmentation masks from an image.
    
    Returns:
        masks: np.ndarray of shape (N, H, W)
        xyxy: np.ndarray of shape (N, 4) - bounding boxes in xyxy format
        conf: np.ndarray of shape (N,) - predicted IoU scores
    """
    results = mask_generator.generate(image)
    masks = []
    xyxy = []
    conf = []
    for r in results:
        masks.append(r["segmentation"])
        r_xyxy = r["bbox"].copy()
        # Convert from xywh to xyxy
        r_xyxy[2] += r_xyxy[0]
        r_xyxy[3] += r_xyxy[1]
        xyxy.append(r_xyxy)
        conf.append(r["predicted_iou"])
    return np.array(masks), np.array(xyxy), np.array(conf)

def non_max_suppression(xyxy: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    boxes = torch.from_numpy(xyxy)
    scores = torch.from_numpy(scores)
    keep = torchvision.ops.nms(boxes, scores, threshold)
    return keep.numpy().tolist()

def main_pipeline_detection_segmentation():

    parser = get_parser()
    args = parser.parse_args()
    if not is_previous_pipeline_done(args, "detection_segmentation"):
        print("Tagging results not found. Please run the tagging pipeline first.")
        return
    default_path = get_default_path(args)
    os.makedirs(default_path["detection_results"], exist_ok=True)
    
    # Use SAM automatic mask generator (dense segmentation)
    if args.class_set == "none":
        mask_generator = get_sam_mask_generator(args.device)

    # Use GroundingDINO for detection and SAM for box-based segmentation.
    elif args.class_set in ["ram", "tag2text"]:
        grounding_dino_model = GroundingDINOModel(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=args.device
        )
        
        sam_predictor = get_sam_predictor(args.device)

        # tagging_results_path = default_path["tagging_results"]
        # with open(tagging_results_path, "r") as f:
        #     tagging_results = json.load(f)
    
    else:
        raise ValueError(f"Invalid class set: {args.class_set}")

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=args.dataset_config,
        start=args.start,
        end=args.end,
        stride=args.stride,
        basedir=args.dataset_root,
        desired_height=args.desired_height,
        desired_width=args.desired_width,
        device="cpu",
        dtype=torch.float,
    )
    
    for idx in trange(len(dataset)):
        color_path = Path(dataset.color_paths[idx])
        image = np.array(Image.open(color_path).convert("RGB"))[:, :, ::-1]

        tagging_result_file = default_path["tagging_results"] / f"{color_path.stem}_tagging.pkl.gz"
        if not tagging_result_file.exists():
            print(f"Tagging result not found: {tagging_result_file}")
            exit()
        with gzip.open(tagging_result_file, "rb") as f:
            tagging_results = pickle.load(f)

        if args.class_set == "none":
            classes = tagging_results.get("classes", ["item"])
            mask, xyxy, conf = get_sam_segmentation_dense(mask_generator, image)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
        
        elif args.class_set in ["ram", "tag2text"]:
            # tag_result = tagging_results.get(str(color_path), {})
            classes = tagging_results.get("classes", ["item"])
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=classes,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold
            )
            # print(f"Detected {len(detections.class_id)} objects with classes: {classes}")
            if len(detections.class_id) > 0:
                # Apply non-maximum suppression.
                keep_idx = non_max_suppression(detections.xyxy, detections.confidence, args.nms_threshold)
                detections.xyxy = detections.xyxy[keep_idx]
                detections.confidence = detections.confidence[keep_idx]
                detections.class_id = detections.class_id[keep_idx]
                
                # Remove detections with invalid class_id (-1).
                valid_idx = detections.class_id != -1
                detections.xyxy = detections.xyxy[valid_idx]
                detections.confidence = detections.confidence[valid_idx]
                detections.class_id = detections.class_id[valid_idx]
                detections.mask = get_sam_segmentation_from_xyxy(sam_predictor, image, detections.xyxy)
        
        draw_bbox = False
        draw_label = True
        instance_random_color = True if args.class_set == "none" else False
        annotated_image, labels = vis_result_fast(image, detections, classes, 
                                                  instance_random_color=instance_random_color, 
                                                  draw_bbox=draw_bbox, 
                                                  draw_label=draw_label)
        cv2.imwrite(default_path["detection_results"] / f"{color_path.stem}_vis.png", annotated_image)
        
        # Save the detection and segmentation results.
        out_file = default_path["detection_results"] / f"{color_path.stem}_detection.pkl.gz"
        detection_results = {
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask
        }
        results = {**detection_results, **tagging_results}
        with gzip.open(out_file, "wb") as f:
            pickle.dump(results, f)
    
    print(f"Detection and segmentation results saved to {default_path['detection_results']}")

if __name__ == "__main__":
    main_pipeline_detection_segmentation()
