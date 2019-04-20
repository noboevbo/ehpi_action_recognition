import collections
import os
from typing import Callable, Dict, List, Type

import cv2
import numpy as np
from nobos_commons.data_structures.human import Human
from nobos_commons.data_structures.skeletons.skeleton_base import SkeletonBase
from nobos_commons.tools.decorators.cache_decorator import cache
from nobos_commons.tools.decorators.timing_decorator import stopwatch
from nobos_commons.utils.file_helper import get_create_path, get_filename_from_path, get_img_paths_from_folder, \
    get_filename_without_extension
from nobos_commons.visualization.pose2d_visualizer import save_humans_img

from ehpi_action_recognition.config import cache_config


class Pose2DNetBase(object):
    __slots__ = ['skeleton_type']

    def __init__(self, skeleton_type: Type[SkeletonBase]):
        self.skeleton_type: Type[SkeletonBase] = skeleton_type

    def get_humans_from_img(self, img: np.ndarray) -> List[Human]:
        raise NotImplementedError

    @cache(cache_config)
    def get_human_data_from_img_path(self, img_path: str, export_path: str = None) -> List[Human]:
        img = cv2.imread(img_path)
        humans = self.get_humans_from_img(img)
        if export_path is not None:
            save_humans_img(img, humans,
                            file_path=os.path.join(get_create_path(export_path), get_filename_from_path(img_path)))
        return humans

    @cache(cache_config)
    def get_human_data_from_img_dir(self, img_dir_path: str, export_dir_path: str = None) \
            -> Dict[str, List[Human]]:
        result_dict = {}
        img_paths = get_img_paths_from_folder(img_dir_path)
        for img_path in img_paths:
            filename = get_filename_from_path(img_path)
            human_data = self.get_human_data_from_img_path(img_path, export_dir_path)
            result_dict[filename] = human_data
        return result_dict

    @cache(cache_config)
    def get_human_data_from_img_dirs(self, img_dir_paths: List[str], export_dir_paths: List[str]) \
            -> Dict[str, Dict[str, List[Human]]]:
        result_dict = collections.OrderedDict()
        for img_dir, export_dir in zip(img_dir_paths, export_dir_paths):
            dir_results = self.get_human_data_from_img_dir(img_dir, export_dir)
            result_dict[img_dir] = dir_results
        return result_dict


@stopwatch
def get_with_filename_idx_changed_to_id_idx(human_data_results: Dict[str, List[Human]],
                                            get_img_id: Callable = lambda filename: filename[-5:]) \
        -> Dict[int, List[Human]]:
    new_dict: Dict[int, List[Human]] = {}
    for filename, pose_result_with_filename_idx in human_data_results.items():
        filename_wo_ext = get_filename_without_extension(filename)
        img_id = int(get_img_id(filename_wo_ext))
        new_dict[img_id] = human_data_results[filename]
    return new_dict


@stopwatch
def get_with_filename_idx_changed_to_id_idx_for_img_dirs(
        human_data_results_for_dirs: Dict[str, Dict[str, List[Human]]],
        get_img_id: Callable = lambda filename: filename[-5:]) -> Dict[str, Dict[int, List[Human]]]:
    new_dict: Dict[str, Dict[int, List[Human]]] = {}
    for img_dir, human_data_results in human_data_results_for_dirs.items():
        new_dict[img_dir] = get_with_filename_idx_changed_to_id_idx(human_data_results, get_img_id)
    return new_dict


