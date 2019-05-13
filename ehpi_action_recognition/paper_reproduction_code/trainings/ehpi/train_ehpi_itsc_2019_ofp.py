import os
import random
from typing import List

import numpy as np
import torch
from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_torch_lib.configs.training_configs.training_config_base import TrainingConfigBase
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset, RemoveJointsOutsideImgEhpi, \
    ScaleEhpi, TranslateEhpi, FlipEhpi, NormalizeEhpi, RemoveJointsEhpi
from nobos_torch_lib.datasets.samplers.imbalanced_dataset_sampler import ImbalancedDatasetSampler
from nobos_torch_lib.learning_rate_schedulers.learning_rate_scheduler_stepwise import \
    LearningRateSchedulerStepwise
from nobos_torch_lib.models.detection_models.shufflenet_v2 import ShuffleNetV2
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import transforms

from ehpi_action_recognition.configs.config import ehpi_dataset_path, models_dir
from ehpi_action_recognition.trainer_ehpi import TrainerEhpi

foot_indexes: List[int] = [11, 14]
knee_indexes: List[int] = [10, 13]


def get_sim_pose_algo_only(dataset_path: str, image_size: ImageSize):
    num_joints = 15
    left_indexes: List[int] = [3, 4, 5, 9, 10, 11]
    right_indexes: List[int] = [6, 7, 8, 12, 13, 14]

    datasets: List[EhpiDataset] = [
        EhpiDataset(os.path.join(dataset_path, "ofp_sim_pose_algo_equal_30fps"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset(os.path.join(dataset_path, "ofp_from_mocap_pose_algo_30fps"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
    ]
    for dataset in datasets:
        dataset.print_label_statistics()

    return ConcatDataset(datasets)


def get_sim_gt_only(dataset_path: str, image_size: ImageSize):
    num_joints = 15
    left_indexes: List[int] = [3, 4, 5, 9, 10, 11]
    right_indexes: List[int] = [6, 7, 8, 12, 13, 14]

    datasets: List[EhpiDataset] = [
        EhpiDataset(os.path.join(dataset_path, "ofp_sim_gt_equal_30fps"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset(os.path.join(dataset_path, "ofp_from_mocap_gt_30fps"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
    ]
    for dataset in datasets:
        dataset.print_label_statistics()

    return ConcatDataset(datasets)


def get_sim(image_size: ImageSize):
    num_joints = 15
    left_indexes: List[int] = [3, 4, 5, 9, 10, 11]
    right_indexes: List[int] = [6, 7, 8, 12, 13, 14]

    datasets: List[EhpiDataset] = [
        EhpiDataset("/media/disks/beta/datasets/ehpi/ofp_sim_pose_algo_equal_30fps",
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset("/media/disks/beta/datasets/ehpi/ofp_from_mocap_pose_algo_30fps",
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset("/media/disks/beta/datasets/ehpi/ofp_sim_gt_equal_30fps",
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset("/media/disks/beta/datasets/ehpi/ofp_from_mocap_gt_30fps",
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints)
    ]
    for dataset in datasets:
        dataset.print_label_statistics()

    return ConcatDataset(datasets)


def get_full(dataset_path: str, image_size: ImageSize):
    num_joints = 15
    left_indexes: List[int] = [3, 4, 5, 9, 10, 11]
    right_indexes: List[int] = [6, 7, 8, 12, 13, 14]

    datasets: List[EhpiDataset] = [
        # Real
        EhpiDataset(os.path.join(dataset_path, "ofp_webcam"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset(os.path.join(dataset_path, "ofp_record_2019_03_11_30FPS"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset(os.path.join(dataset_path, "ofp_record_2019_03_11_HSRT_30FPS"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints, dataset_part=DatasetPart.TEST),
        EhpiDataset(os.path.join(dataset_path, "ofp_record_2019_03_11_HELLA_30FPS"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints, dataset_part=DatasetPart.TRAIN),
        # Freilichtmuseum
        EhpiDataset(os.path.join(dataset_path, "2019_03_13_Freilichtmuseum_30FPS"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints, dataset_part=DatasetPart.TRAIN),
        # Simulated
        EhpiDataset(os.path.join(dataset_path, "ofp_from_mocap_30fps/"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset(os.path.join(dataset_path, "ofp_sim_pose_algo_equal_30fps"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset(os.path.join(dataset_path, "ofp_sim_gt_equal_30fps"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset(os.path.join(dataset_path, "ofp_from_mocap_gt_30fps"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                         probability=0.25),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
    ]
    for dataset in datasets:
        dataset.print_label_statistics()

    return ConcatDataset(datasets)


def get_set_wo_sim(dataset_path: str, image_size: ImageSize):
    num_joints = 15
    left_indexes: List[int] = [3, 4, 5, 9, 10, 11]
    right_indexes: List[int] = [6, 7, 8, 12, 13, 14]

    datasets: List[EhpiDataset] = [
        EhpiDataset(os.path.join(dataset_path, "ofp_webcam"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset(os.path.join(dataset_path, "ofp_record_2019_03_11_30FPS"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints),
        EhpiDataset(os.path.join(dataset_path, "ofp_record_2019_03_11_HSRT_30FPS"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints, dataset_part=DatasetPart.TEST),
        EhpiDataset(os.path.join(dataset_path, "ofp_record_2019_03_11_HELLA_30FPS"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints, dataset_part=DatasetPart.TRAIN),
        # Freilichtmuseum
        EhpiDataset(os.path.join(dataset_path, "2019_03_13_Freilichtmuseum_30FPS"),
                    transform=transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        ScaleEhpi(image_size),
                        TranslateEhpi(image_size),
                        FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                        NormalizeEhpi(image_size)
                    ]), num_joints=num_joints, dataset_part=DatasetPart.TRAIN),
    ]
    for dataset in datasets:
        dataset.print_label_statistics()

    return ConcatDataset(datasets)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(0)


if __name__ == '__main__':
    batch_size = 128
    seeds = [0, 104, 123, 142, 200]

    datasets = {
        "sim_pose_algo_only": get_sim_pose_algo_only,
        "sim_gt_only": get_sim_gt_only,
        "wo_sim": get_set_wo_sim,
        "sim": get_sim,
        "full": get_full

    }
    for seed in seeds:
        use_case_dataset_path = os.path.join(ehpi_dataset_path, "use_case")
        for dataset_name, get_dataset in datasets.items():
            # Train set
            set_seed(seed)
            train_set = get_dataset(use_case_dataset_path, image_size=ImageSize(1280, 720))
            sampler = ImbalancedDatasetSampler(train_set, dataset_type=EhpiDataset)
            train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=8)

            # config
            train_config = TrainingConfigBase("itsc2019_{}_seed_{}".format(dataset_name, seed),
                                              os.path.join(models_dir, "train_use_case"))
            train_config.learning_rate_scheduler = LearningRateSchedulerStepwise(lr_decay=0.1, lr_decay_epoch=50)
            train_config.learning_rate = 0.05
            train_config.weight_decay = 5e-4
            train_config.num_epochs = 140

            trainer = TrainerEhpi()

            trainer.train(train_loader, train_config, model=ShuffleNetV2(3))
