
from collections.abc import Iterable
import copy
import matplotlib
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d

from conceptgraph.utils.general_utils import to_numpy, to_tensor

class DetectionList(list):
    def get_values(self, key, idx:int=None):
        if idx is None:
            return [detection[key] for detection in self]
        else:
            return [detection[key][idx] for detection in self]
    
    def get_stacked_values_torch(self, key, idx:int=None):
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or \
                isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0)
    
    def get_stacked_values_numpy(self, key, idx:int=None):
        values = self.get_stacked_values_torch(key, idx)
        return to_numpy(values)
    
    def __add__(self, other):
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list
    
    def __iadd__(self, other):
        self.extend(other)
        return self
    
    def slice_by_indices(self, index: Iterable[int]):
        '''
        Return a sublist of the current list by indexing
        '''
        new_self = type(self)()
        for i in index:
            new_self.append(self[i])
        return new_self
    
    def slice_by_mask(self, mask: Iterable[bool]):
        '''
        Return a sublist of the current list by masking
        '''
        new_self = type(self)()
        for i, m in enumerate(mask):
            if m:
                new_self.append(self[i])
        return new_self
    
    def get_most_common_class(self) -> list[int]:
        classes = []
        for d in self:
            values, counts = np.unique(np.asarray(d['class_id']), return_counts=True)
            most_common_class = values[np.argmax(counts)]
            classes.append(most_common_class)
        return classes
    
    def color_by_most_common_classes(self, colors_dict: dict[str, list[float]], color_bbox: bool=True):
        '''
        Color the point cloud of each detection by the most common class
        '''
        classes = self.get_most_common_class()
        for d, c in zip(self, classes):
            color = colors_dict[str(c)]
            d['pcd'].paint_uniform_color(color)
            if color_bbox:
                d['bbox'].color = color
                
    def color_by_instance(self):
        if len(self) == 0:
            # Do nothing
            return
        
        if "inst_color" in self[0]:
            for d in self:
                d['pcd'].paint_uniform_color(d['inst_color'])
                d['bbox'].color = d['inst_color']
        else:
            cmap = matplotlib.colormaps.get_cmap("turbo")
            instance_colors = cmap(np.linspace(0, 1, len(self)))
            instance_colors = instance_colors[:, :3]
            for i in range(len(self)):
                self[i]['pcd'].paint_uniform_color(instance_colors[i])
                self[i]['bbox'].color = instance_colors[i]
            
    
class MapObjectList(DetectionList):
    def compute_similarities(self, new_clip_ft):
        '''
        The input feature should be of shape (D, ), a one-row vector
        This is mostly for backward compatibility
        '''
        # if it is a numpy array, make it a tensor 
        new_clip_ft = to_tensor(new_clip_ft)
        
        # assuming cosine similarity for features
        clip_fts = self.get_stacked_values_torch('clip_ft')

        similarities = F.cosine_similarity(new_clip_ft.unsqueeze(0), clip_fts)
        # return similarities.squeeze()
        return similarities
    
    def to_serializable(self):
        s_obj_list = []
        for obj in self:
            s_obj_dict = copy.deepcopy(obj)
            
            s_obj_dict['clip_ft'] = to_numpy(s_obj_dict['clip_ft'])
            s_obj_dict['text_ft'] = to_numpy(s_obj_dict['text_ft'])
            
            s_obj_dict['pcd_np'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)
            
            del s_obj_dict['pcd']
            del s_obj_dict['bbox']
            
            s_obj_list.append(s_obj_dict)
            
        return s_obj_list
    
    def load_serializable(self, s_obj_list):
        assert len(self) == 0, 'MapObjectList should be empty when loading'
        for s_obj_dict in s_obj_list:
            new_obj = copy.deepcopy(s_obj_dict)
            
            new_obj['clip_ft'] = to_tensor(new_obj['clip_ft'])
            new_obj['text_ft'] = to_tensor(new_obj['text_ft'])
            
            new_obj['pcd'] = o3d.geometry.PointCloud()
            new_obj['pcd'].points = o3d.utility.Vector3dVector(new_obj['pcd_np'])
            new_obj['bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(new_obj['bbox_np']))
            new_obj['bbox'].color = new_obj['pcd_color_np'][0]
            new_obj['pcd'].colors = o3d.utility.Vector3dVector(new_obj['pcd_color_np'])
            
            del new_obj['pcd_np']
            del new_obj['bbox_np']
            del new_obj['pcd_color_np']
            
            self.append(new_obj)

import os
import gzip
import pickle
from pathlib import Path
from datetime import datetime

def save_detections(save_folder: Path, objects, bg_objects, cfg, class_names, class_colors):
    """
    Save detections as individual pickle files in subfolders.
    
    Args:
        save_folder (Path): The folder to save all detections.
        objects (MapObjectList): The list of detected objects.
        bg_objects (MapObjectList or None): The list of background objects (if any).
        cfg: Configuration object.
        class_names: List of class names.
        class_colors: Dictionary of class colors.
    """
    # Create the main save folder if it doesn't exist.
    save_folder.mkdir(parents=True, exist_ok=True)

    # Save the objects in a subfolder.
    objects_folder = save_folder / "objects"
    objects_folder.mkdir(parents=True, exist_ok=True)
    # Get a serializable list (each object as a dict)
    objects_serialized = objects.to_serializable()
    for idx, obj in enumerate(objects_serialized):
        file_path = objects_folder / f"object_{idx:04d}.pkl.gz"
        with gzip.open(file_path, "wb") as f:
            pickle.dump(obj, f)

    # Save background objects if available.
    if bg_objects is not None:
        bg_folder = save_folder / "bg_objects"
        bg_folder.mkdir(parents=True, exist_ok=True)
        bg_serialized = bg_objects.to_serializable()
        for idx, obj in enumerate(bg_serialized):
            file_path = bg_folder / f"bg_object_{idx:04d}.pkl.gz"
            with gzip.open(file_path, "wb") as f:
                pickle.dump(obj, f)

    # Save metadata (cfg, class names, class colors) in the root of save_folder.
    metadata = {
        'cfg': cfg,
        'class_names': class_names,
        'class_colors': class_colors,
    }
    metadata_path = save_folder / "metadata.pkl.gz"
    with gzip.open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved detections to {save_folder}")


def save_detections_pcd(save_folder: Path, objects, bg_objects, cfg, class_names, class_colors):
    # save the pcds under the subfolder "objects_pcd" and "bg_objects_pcd"

    # Create the main save folder if it doesn't exist.
    save_folder.mkdir(parents=True, exist_ok=True)
    # Save the objects in a subfolder.
    objects_folder = save_folder / "objects_pcd"
    objects_folder.mkdir(parents=True, exist_ok=True)
    # Get a serializable list (each object as a dict)
    objects_serialized = objects.to_serializable()
    for idx, obj in enumerate(objects_serialized):
        file_path = objects_folder / f"object_{idx:04d}.pcd"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj['pcd_np'])
        pcd.colors = o3d.utility.Vector3dVector(obj['pcd_color_np'])
        o3d.io.write_point_cloud(file_path, pcd)
    # Save background objects if available.
    if bg_objects is not None:
        bg_folder = save_folder / "bg_objects_pcd"
        bg_folder.mkdir(parents=True, exist_ok=True)
        bg_serialized = bg_objects.to_serializable()
        for idx, obj in enumerate(bg_serialized):
            file_path = bg_folder / f"bg_object_{idx:04d}.pcd"
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj['pcd_np'])
            pcd.colors = o3d.utility.Vector3dVector(obj['pcd_color_np'])
            o3d.io.write_point_cloud(file_path, pcd)
    # Save metadata (cfg, class names, class colors) in the root of save_folder.
    metadata = {
        'cfg': cfg,
        'class_names': class_names,
        'class_colors': class_colors,
    }
    metadata_path = save_folder / "metadata.pkl.gz"
    with gzip.open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved pcds to {save_folder}")


def load_detections(load_folder: Path):
    """
    Load detections from a folder created by save_detections.
    This function rehydrates each detection (using MapObjectList.load_serializable)
    so that keys like 'pcd' and 'bbox' are properly reconstructed.
    
    Returns:
        objects (MapObjectList), bg_objects (MapObjectList or None), metadata (dict)
    """
    # Load metadata first.
    metadata_path = load_folder / "metadata.pkl.gz"
    with gzip.open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Load serialized objects from the "objects" subfolder.
    objects_folder = load_folder / "objects"
    objects_serialized = []
    for file_path in sorted(objects_folder.glob("object_*.pkl.gz")):
        with gzip.open(file_path, "rb") as f:
            obj = pickle.load(f)
        objects_serialized.append(obj)
    
    # Convert the list of dicts into a MapObjectList using load_serializable.
    objects = MapObjectList()
    objects.load_serializable(objects_serialized)

    # Load background objects if available.
    bg_objects = None
    bg_folder = load_folder / "bg_objects"
    if bg_folder.exists():
        bg_serialized = []
        for file_path in sorted(bg_folder.glob("bg_object_*.pkl.gz")):
            with gzip.open(file_path, "rb") as f:
                obj = pickle.load(f)
            bg_serialized.append(obj)
        bg_objects = MapObjectList()
        bg_objects.load_serializable(bg_serialized)

    print(f"Loaded {len(objects)} objects and {len(bg_objects) if bg_objects is not None else 0} background objects")
    return objects, bg_objects, metadata