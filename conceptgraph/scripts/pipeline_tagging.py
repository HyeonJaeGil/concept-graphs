import os
import argparse
import json
import pickle
import gzip
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as TS
from tqdm import trange
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.scripts.pipeline_common import *
from ram.models import ram, tag2text
from ram import inference_tag2text, inference_ram


def process_tag_classes(text_prompt: str, add_classes: list, remove_classes: list) -> list:
    classes = [cls.strip() for cls in text_prompt.split(',') if cls.strip() != '']
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    for c in remove_classes:
        classes = [cls for cls in classes if c.lower() not in cls.lower()]
    return classes

def get_tagging_model(model_name: str, device: str) -> torch.nn.Module:
    if model_name == "ram":
        model_instance = ram(pretrained=RAM_CHECKPOINT_PATH, 
                             image_size=384, vit='swin_l')
        inference_fn = inference_ram
    elif model_name == "tag2text":
        model_instance = tag2text(pretrained=TAG2TEXT_CHECKPOINT_PATH, 
                                  image_size=384, vit='swin_b', 
                                  delete_tag_index=list(range(3012, 3429)))
        model_instance.threshold = 0.64
        inference_fn = inference_tag2text
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    model_instance = model_instance.eval().to(device)
    return model_instance, inference_fn


def main_pipeline_tagging():
    parser = get_parser()
    args = parser.parse_args()
    default_path = get_default_path(args)
    os.makedirs(default_path["tagging_results"], exist_ok=True)

    # Load the chosen tagging model.
    if args.class_set == "none":
        print("No model is loaded. Saving class set as 'item' only.")
        model_instance, inference_fn = None, None
    else:
        model_instance, inference_fn = get_tagging_model(args.class_set, args.device)
    
    tagging_transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(),
        TS.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    global_classes = set() if args.class_set in ["ram", "tag2text"] else set(["item"])

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
        out_file = default_path["tagging_results"] / f"{color_path.stem}_tagging.pkl.gz"

        if args.class_set == "none":
            results = {"classes": ["item"]}
            with gzip.open(out_file, "wb") as f:
                pickle.dump(results, f)
            continue

        image = Image.open(color_path).convert("RGB")
        raw_image = image.resize((384, 384))
        tensor_image = tagging_transform(raw_image).unsqueeze(0).to(args.device)
        
        if args.class_set == "ram":
            res = inference_fn(tensor_image, model_instance)
            caption = "NA"
        elif args.class_set == "tag2text":
            res = inference_fn(tensor_image, model_instance, "None")
            caption = res[2]
        else:
            raise ValueError(f"Invalid class set: {args.class_set}")
        
        # Process the text prompt to obtain a list of classes.
        text_prompt = res[0].replace(' |', ',')
        add_classes = ["other item"]
        remove_classes = [
            "room", "kitchen", "office", "house", "home", "building", "corner",
            "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium",
            "apartment", "image", "city", "blue", "skylight", "hallway", "bureau",
            "modern", "salon", "doorway", "wall lamp", "wood floor"
        ]
        bg_classes = ["wall", "floor", "ceiling"]

        if args.add_bg_classes:
            add_classes += bg_classes
        else:
            remove_classes += bg_classes
        
        classes = process_tag_classes(text_prompt, add_classes, remove_classes)

        global_classes.update(classes)
        if args.accumu_classes:
            classes = list(global_classes)

        results = {
            "caption": caption,
            "text_prompt": text_prompt,
            "classes": classes
        }
        with gzip.open(out_file, "wb") as f:
            pickle.dump(results, f)
        
    # save global classes
    with open(default_path["gsa_classes"], "w") as f:
        json.dump(list(global_classes), f)

if __name__ == "__main__":
    main_pipeline_tagging()
