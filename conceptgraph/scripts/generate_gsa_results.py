'''
The script is used to extract Grounded SAM results on a posed RGB-D dataset. 
The results will be dumped to a folder under the scene folder. 
'''

import os
from pathlib import Path
import re
from typing import Any, List
from PIL import Image
import cv2
import json
import imageio
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import pickle
import gzip
import open_clip
import hydra
import omegaconf
from omegaconf import DictConfig

import torch
import torchvision
from torch.utils.data import Dataset
import supervision as sv
from tqdm import trange

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption
from conceptgraph.utils.model_utils import compute_clip_features
import torch.nn.functional as F


try: 
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e

# Set up some path used in this script
# Assuming all checkpoint files are downloaded as instructed by the original GSA repo
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    
import sys
TAG2TEXT_PATH = os.path.join(GSA_PATH, "")
EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
sys.path.append(GSA_PATH) # This is needed for the following imports in this file
sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
sys.path.append(EFFICIENTSAM_PATH)

import torchvision.transforms as TS
try:
    from ram.models import ram
    from ram.models import tag2text
    from ram import inference_tag2text, inference_ram
except ImportError as e:
    print("RAM sub-package not found. Please check your GSA_PATH. ")
    raise e

# Disable torch gradient computation
torch.set_grad_enabled(False)
    
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

FOREGROUND_GENERIC_CLASSES = [
    "item", "furniture", "object", "electronics", "wall decoration", "door"
]

FOREGROUND_MINIMAL_CLASSES = [
    "item"
]


def _build_clip_components(model_name: str, device: str):
    if model_name == "default":
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    elif model_name == "eva":  # eva02
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k"
        )
        clip_tokenizer = open_clip.get_tokenizer(
            "hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k"
        )
    else:
        raise ValueError(f"Unsupported clip_model: {model_name}")
    clip_model = clip_model.to(device)
    return clip_model, clip_preprocess, clip_tokenizer





# Prompting SAM with detected boxes
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


def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
    
    if variant == "mobilesam":
        from MobileSAM.setup_mobile_sam import setup_model
        MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/mobile_sam.pt")
        checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        
        sam_predictor = SamPredictor(mobile_sam)
        return sam_predictor

    elif variant == "lighthqsam":
        from LightHQSAM.setup_light_hqsam import setup_model
        HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
        checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=device)
        
        sam_predictor = SamPredictor(light_hqsam)
        return sam_predictor
        
    elif variant == "fastsam":
        raise NotImplementedError
    else:
        raise NotImplementedError
    


# The SAM based on automatic mask generation, without bbox prompting
def get_sam_segmentation_dense(
    variant:str, model: Any, image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    The SAM based on automatic mask generation, without bbox prompting
    
    Args:
        model: The mask generator or the YOLO model
        image: )H, W, 3), in RGB color space, in range [0, 255]
        
    Returns:
        mask: (N, H, W)
        xyxy: (N, 4)
        conf: (N,)
    '''
    if variant == "sam":
        results = model.generate(image)
        mask = []
        xyxy = []
        conf = []
        for r in results:
            mask.append(r["segmentation"])
            r_xyxy = r["bbox"].copy()
            # Convert from xyhw format to xyxy format
            r_xyxy[2] += r_xyxy[0]
            r_xyxy[3] += r_xyxy[1]
            xyxy.append(r_xyxy)
            conf.append(r["predicted_iou"])
        mask = np.array(mask)
        xyxy = np.array(xyxy)
        conf = np.array(conf)
        return mask, xyxy, conf
    elif variant == "fastsam":
        # The arguments are directly copied from the GSA repo
        results = model(
            image,
            imgsz=1024,
            device="cuda",
            retina_masks=True,
            iou=0.9,
            conf=0.4,
            max_det=100,
        )
        raise NotImplementedError
    else:
        raise NotImplementedError


def get_sam_mask_generator(variant:str, device: str | int) -> SamAutomaticMaskGenerator:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=12,
            points_per_batch=12,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=100,
        )
        return mask_generator
    elif variant == "fastsam":
        raise NotImplementedError
        # from ultralytics import YOLO
        # from FastSAM.tools import *
        # FASTSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/FastSAM-x.pt")
        # model = YOLO(cfg.model_path)
        # return model
    else:
        raise NotImplementedError


def process_tag_classes(text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]) -> list[str]:
    '''
    Convert a text prompt from Tag2Text to a list of classes. 
    '''
    classes = text_prompt.split(',')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes


def process_ai2thor_classes(classes: List[str], add_classes:List[str]=[], remove_classes:List[str]=[]) -> List[str]:
    '''
    Some pre-processing on AI2Thor objectTypes in a scene
    '''
    classes = list(set(classes))
    
    for c in add_classes:
        classes.append(c)
        
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]

    # Split the element in classes by captical letters
    classes = [obj_class.replace("TV", "Tv") for obj_class in classes]
    classes = [re.findall('[A-Z][^A-Z]*', obj_class) for obj_class in classes]
    # Join the elements in classes by space
    classes = [" ".join(obj_class) for obj_class in classes]
    
    return classes
    
    
def process_cfg(cfg: DictConfig) -> DictConfig:
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)

    if cfg.dataset_config.name != "multiscan.yaml":
        dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
    else:
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg


@hydra.main(version_base=None, config_path="../configs/slam_pipeline", config_name="base")
def main(cfg: DictConfig):
    cfg = process_cfg(cfg)
    # Decouple segmentation/tagging from CLIP feature extraction to reduce peak memory.
    need_seg = cfg.stage in ["all", "seg"]
    need_clip = cfg.stage in ["all", "clip"]

    ### Initialize the Grounding DINO / SAM models only if segmentation is needed ###
    if need_seg:
        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
            device=cfg.device
        )

        if cfg.class_set == "none":
            mask_generator = get_sam_mask_generator(cfg.sam_variant, cfg.device)
            sam_predictor = None
        else:
            sam_predictor = get_sam_predictor(cfg.sam_variant, cfg.device)
            mask_generator = None
    else:
        grounding_dino_model = None
        mask_generator = None
        sam_predictor = None
    
    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )

    global_classes = set()
    
    # Initialize a YOLO-World model (segmentation stage only)
    if need_seg and cfg.detector == "yolo":
        from ultralytics import YOLO
        yolo_model_w_classes = YOLO('yolov8l-world.pt')  # or choose yolov8m/l-world.pt
    else:
        yolo_model_w_classes = None
    
    if need_seg:
        if cfg.class_set == "scene":
            # Load the object meta information
            obj_meta_path = cfg.dataset_root / "obj_meta.json"
            with open(obj_meta_path, "r") as f:
                obj_meta = json.load(f)
            # Get a list of object classes in the scene
            classes = process_ai2thor_classes(
                [obj["objectType"] for obj in obj_meta],
                add_classes=[],
                remove_classes=['wall', 'floor', 'room', 'ceiling']
            )
        elif cfg.class_set == "generic":
            classes = FOREGROUND_GENERIC_CLASSES
        elif cfg.class_set == "minimal":
            classes = FOREGROUND_MINIMAL_CLASSES
        elif cfg.class_set in ["tag2text", "ram"]:
            ### Initialize the Tag2Text or RAM model ###
            
            if cfg.class_set == "tag2text":
                # The class set will be computed by tag2text on each image
                # filter out attributes and action categories which are difficult to grounding
                delete_tag_index = []
                for i in range(3012, 3429):
                    delete_tag_index.append(i)

                specified_tags='None'
                # load model
                tagging_model = tag2text.tag2text_caption(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                                                        image_size=384,
                                                        vit='swin_b',
                                                        delete_tag_index=delete_tag_index)
                # threshold for tagging
                # we reduce the threshold to obtain more tags
                tagging_model.threshold = 0.64 
            elif cfg.class_set == "ram":
                tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,
                                             image_size=384,
                                             vit='swin_l')
                
            tagging_model = tagging_model.eval().to(cfg.device)
            
            # initialize Tag2Text
            tagging_transform = TS.Compose([
                TS.Resize((384, 384)),
                TS.ToTensor(), 
                TS.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])
            
            classes = None
        elif cfg.class_set == "none":
            classes = ['item']
        else:
            raise ValueError("Unknown cfg.class_set: ", cfg.class_set)
    else:
        classes = None

    if need_seg:
        if cfg.class_set not in ["ram", "tag2text"]:
            print("There are total", len(classes), "classes to detect. ")
        elif cfg.class_set == "none":
            print("Skipping tagging and detection models. ")
        else:
            print(f"{cfg.class_set} will be used to detect classes. ")
    
    detections_dir = cfg.dataset_root / cfg.detection_folder_name
    vis_dir = cfg.dataset_root / cfg.det_vis_folder_name
    
    vis_video_enabled = cfg.save_video and need_seg
    if vis_video_enabled:
        video_save_path = cfg.dataset_root / f"{cfg.det_vis_folder_name}.mp4"
        frames = []

    def run_segmentation_pass():
        for idx in trange(len(dataset)):
            ### Relevant paths and load image ###
            color_path = dataset.color_paths[idx]

            color_path = Path(color_path)
            
            vis_save_path = vis_dir / color_path.name
            detections_save_path = detections_dir / color_path.name
            detections_save_path = detections_save_path.with_suffix(".pkl.gz")
            
            os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(detections_save_path), exist_ok=True)
            
            # opencv can't read Path objects... sigh...
            color_path = str(color_path)
            vis_save_path = str(vis_save_path)
            detections_save_path = str(detections_save_path)
            
            image = cv2.imread(color_path) # This will in BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB color space
            image_pil = Image.fromarray(image_rgb)
            
            ### Tag2Text ###
            if cfg.class_set in ["ram", "tag2text"]:
                raw_image = image_pil.resize((384, 384))
                raw_image = tagging_transform(raw_image).unsqueeze(0).to(cfg.device)
                
                if cfg.class_set == "ram":
                    res = inference_ram(raw_image , tagging_model)
                    caption="NA"
                elif cfg.class_set == "tag2text":
                    res = inference_tag2text.inference(raw_image , tagging_model, specified_tags)
                    caption=res[2]

                # Currently ", " is better for detecting single tags
                # while ". " is a little worse in some case
                text_prompt=res[0].replace(' |', ',')
                
                # Add "other item" to capture objects not in the tag2text captions. 
                # Remove "xxx room", otherwise it will simply include the entire image
                # Also hide "wall" and "floor" for now...
                add_classes = ["other item"]
                remove_classes = [
                    "room", "kitchen", "office", "house", "home", "building", "corner",
                    "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
                    "apartment", "image", "city", "blue", "skylight", "hallway", 
                    "bureau", "modern", "salon", "doorway", "wall lamp", "wood floor"
                ]
                bg_classes = ["wall", "floor", "ceiling"]

                if cfg.add_bg_classes:
                    add_classes += bg_classes
                else:
                    remove_classes += bg_classes

                classes_local = process_tag_classes(
                    text_prompt, 
                    add_classes = add_classes,
                    remove_classes = remove_classes,
                )
            else:
                classes_local = classes
            
            # add classes to global classes
            global_classes.update(classes_local)
            
            if cfg.accumu_classes:
                # Use all the classes that have been seen so far
                classes_local = list(global_classes)
                
            ### Detection and segmentation ###
            if cfg.class_set == "none":
                # Directly use SAM in dense sampling mode to get segmentation
                mask, xyxy, conf = get_sam_segmentation_dense(
                    cfg.sam_variant, mask_generator, image_rgb)
                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=conf,
                    class_id=np.zeros_like(conf).astype(int),
                    mask=mask,
                )

                ### Visualize results ###
                annotated_image, labels = vis_result_fast(
                    image, detections, classes_local, instance_random_color=True)
                
                cv2.imwrite(vis_save_path, annotated_image)
            else:
                if cfg.detector == "dino":
                    # Using GroundingDINO to detect and SAM to segment
                    detections = grounding_dino_model.predict_with_classes(
                        image=image, # This function expects a BGR image...
                        classes=classes_local,
                        box_threshold=cfg.box_threshold,
                        text_threshold=cfg.text_threshold,
                    )
                
                    if len(detections.class_id) > 0:
                        ### Non-maximum suppression ###
                        nms_idx = torchvision.ops.nms(
                            torch.from_numpy(detections.xyxy), 
                            torch.from_numpy(detections.confidence), 
                            cfg.nms_threshold
                        ).numpy().tolist()

                        detections.xyxy = detections.xyxy[nms_idx]
                        detections.confidence = detections.confidence[nms_idx]
                        detections.class_id = detections.class_id[nms_idx]
                        
                        # Somehow some detections will have class_id=-1, remove them
                        valid_idx = detections.class_id != -1
                        detections.xyxy = detections.xyxy[valid_idx]
                        detections.confidence = detections.confidence[valid_idx]
                        detections.class_id = detections.class_id[valid_idx]

                elif cfg.detector == "yolo":
                    # YOLO 
                    yolo_model_w_classes.set_classes(classes_local)
                    yolo_results_w_classes = yolo_model_w_classes.predict(color_path)

                    yolo_results_w_classes[0].save(vis_save_path[:-4] + "_yolo_out.jpg")
                    xyxy_tensor = yolo_results_w_classes[0].boxes.xyxy 
                    xyxy_np = xyxy_tensor.cpu().numpy()
                    confidences = yolo_results_w_classes[0].boxes.conf.cpu().numpy()
                    
                    detections = sv.Detections(
                        xyxy=xyxy_np,
                        confidence=confidences,
                        class_id=yolo_results_w_classes[0].boxes.cls.cpu().numpy().astype(int),
                        mask=None,
                    )
                    
                if len(detections.class_id) > 0:
                    
                    ### Segment Anything ###
                    detections.mask = get_sam_segmentation_from_xyxy(
                        sam_predictor=sam_predictor,
                        image=image_rgb,
                        xyxy=detections.xyxy
                    )
                else:
                    # Empty detections holder
                    detections.mask = np.zeros((0, *image_rgb.shape[:2]), dtype=bool)
                
                ### Visualize results ###
                annotated_image, labels = vis_result_fast(image, detections, classes_local)
                
                # save the annotated grounded-sam image
                if cfg.class_set in ["ram", "tag2text"] and cfg.use_slow_vis:
                    annotated_image_caption = vis_result_slow_caption(
                        image_rgb, detections.mask, detections.xyxy, labels, caption, text_prompt)
                    Image.fromarray(annotated_image_caption).save(vis_save_path)
                else:
                    cv2.imwrite(vis_save_path, annotated_image)
            
            if vis_video_enabled:
                frames.append(annotated_image)
            
            # Convert the detections to a dict. The elements are in np.array.
            # CLIP features are intentionally left empty here and will be filled in the clip pass.
            results = {
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "class_id": detections.class_id,
                "mask": detections.mask,
                "classes": classes_local,
                "image_crops": [],
                "image_feats": [],
                "text_feats": [],
            }
            
            if cfg.class_set in ["ram", "tag2text"]:
                results["tagging_caption"] = caption
                results["tagging_text_prompt"] = text_prompt
            
            # save the detections using pickle
            with gzip.open(detections_save_path, "wb") as f:
                pickle.dump(results, f)


    if need_seg:
        run_segmentation_pass()

    # Free segmentation models before heavy CLIP pass if both stages are run in one go.
    if need_seg and need_clip:
        del grounding_dino_model
        del sam_predictor
        del mask_generator
        del yolo_model_w_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ### Initialize the CLIP model only if feature extraction is needed ###
    clip_model = clip_preprocess = clip_tokenizer = None
    if need_clip and cfg.stage == "clip":
        clip_model, clip_preprocess, clip_tokenizer = _build_clip_components(
            model_name=cfg.clip_model,
            device=cfg.device,
        )

    def run_clip_pass():
        nonlocal clip_model, clip_preprocess, clip_tokenizer
        if clip_model is None:
            clip_model, clip_preprocess, clip_tokenizer = _build_clip_components(
                model_name=cfg.clip_model,
                device=cfg.device,
            )

        for idx in trange(len(dataset)):
            color_path = Path(dataset.color_paths[idx])
            detections_save_path = detections_dir / color_path.name
            detections_save_path = detections_save_path.with_suffix(".pkl.gz")
            if not detections_save_path.exists():
                print(f"[clip pass] Missing detection file {detections_save_path}, skipping.")
                continue

            with gzip.open(detections_save_path, "rb") as f:
                results = pickle.load(f)

            image = cv2.imread(str(color_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            detections = sv.Detections(
                xyxy=np.asarray(results["xyxy"]),
                confidence=np.asarray(results["confidence"]),
                class_id=np.asarray(results["class_id"]),
                mask=np.asarray(results["mask"]),
            )

            classes_local = results["classes"]
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes_local, cfg.device)

            results["image_crops"] = image_crops
            results["image_feats"] = image_feats
            results["text_feats"] = text_feats

            with gzip.open(detections_save_path, "wb") as f:
                pickle.dump(results, f)

    if need_clip:
        run_clip_pass()

    # save global classes from segmentation stage
    if need_seg:
        with open(cfg.dataset_root / f"{cfg.color_file_name}.json", "w") as f:
            json.dump(list(global_classes), f)
                
    if vis_video_enabled:
        imageio.mimsave(video_save_path, frames, fps=10)
        print(f"Video saved to {video_save_path}")
        

if __name__ == "__main__":
    main()
