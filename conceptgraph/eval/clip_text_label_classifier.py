from __future__ import annotations

from typing import List

import numpy as np
import open_clip
import torch


class ClipTextLabelClassifier:
    """Classifies mapped objects to text labels using CLIP feature similarity."""

    def __init__(
        self,
        model_name: str = "default",
        device: str = "cuda:0",
        clip_device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.clip_device = clip_device or device

        if model_name == "default":
            self.clip_model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
            self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        elif model_name == "eva":
            model_id = "hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k"
            self.clip_model, _, _ = open_clip.create_model_and_transforms(model_id)
            self.tokenizer = open_clip.get_tokenizer(model_id)
        else:
            raise ValueError(f"Unsupported clip model: {model_name}")

        self.clip_model = self.clip_model.to(self.clip_device)

    @torch.no_grad()
    def build_text_features(self, class_names: List[str]) -> torch.Tensor:
        prompts = [f"an image of {c}" for c in class_names]
        text = self.tokenizer(prompts).to(self.clip_device)
        class_feats = self.clip_model.encode_text(text)
        class_feats = class_feats / class_feats.norm(dim=-1, keepdim=True)
        return class_feats.to(self.device)

    @torch.no_grad()
    def classify_object_list(
        self,
        objects,
        class_feats: torch.Tensor,
        class_names: List[str],
        ignore_index: np.ndarray,
        object_feat_key: str = "clip_ft",
        print_assignments: bool = True,
    ) -> torch.Tensor:
        object_feats = objects.get_stacked_values_torch(object_feat_key).to(self.device)
        if object_feats.ndim == 3 and object_feats.shape[1] == 1:
            object_feats = object_feats[:, 0, :]
        object_feats = object_feats / object_feats.norm(dim=-1, keepdim=True)

        object_class_sim = object_feats @ class_feats.T
        if len(ignore_index) > 0:
            object_class_sim[:, ignore_index] = -1e10

        object_class = object_class_sim.argmax(dim=-1).detach().cpu()

        if print_assignments:
            print("Assigned object classes:")
            for i in range(len(objects)):
                class_id = int(object_class[i].item())
                class_name = class_names[class_id]
                orig_label = objects[i]["label"] if "label" in objects[i] else "N/A"
                print(f"Object {i}: (orig: {orig_label}) class {class_id} - {class_name}")

        return object_class
