import os
import random
from typing import List

import numpy as np
import torch
from nobos_commons.data_structures.dimension import ImageSize
from nobos_torch_lib.configs.training_configs.training_config_base import TrainingConfigBase
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import ScaleEhpi, TranslateEhpi, \
    FlipEhpi, NormalizeEhpi, \
    RemoveJointsOutsideImgEhpi
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import transforms

from ehpi_action_recognition.paper_reproduction_code.datasets.ehpi_lstm_dataset import EhpiLSTMDataset
from ehpi_action_recognition.paper_reproduction_code.models.ehpi_lstm import EhpiLSTM
from ehpi_action_recognition.trainer_ehpi import TrainerEhpi


def get_training_set_gt(dataset_path: str, image_size: ImageSize):
    num_joints = 15
    left_indexes: List[int] = [3, 4, 5, 9, 10, 11]
    right_indexes: List[int] = [6, 7, 8, 12, 13, 14]

    datasets: List[EhpiLSTMDataset] = [
        EhpiLSTMDataset(os.path.join(dataset_path, "JOURNAL_2019_03_GT_30fps"),
                        transform=transforms.Compose([
                            RemoveJointsOutsideImgEhpi(image_size),
                            ScaleEhpi(image_size),
                            TranslateEhpi(image_size),
                            FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                            NormalizeEhpi(image_size)
                        ]), num_joints=num_joints),
    ]
    for dataset in datasets:
        dataset.print_label_statistics()

    return ConcatDataset(datasets)


def get_training_posealgo(dataset_path: str, image_size: ImageSize):
    num_joints = 15
    left_indexes: List[int] = [3, 4, 5, 9, 10, 11]
    right_indexes: List[int] = [6, 7, 8, 12, 13, 14]

    datasets: List[EhpiLSTMDataset] = [
        EhpiLSTMDataset(os.path.join(dataset_path, "JOURNAL_2019_03_POSEALGO_30fps"),
                        transform=transforms.Compose([
                            RemoveJointsOutsideImgEhpi(image_size),
                            ScaleEhpi(image_size),
                            TranslateEhpi(image_size),
                            FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                            NormalizeEhpi(image_size)
                        ]), num_joints=num_joints),
    ]
    for dataset in datasets:
        dataset.print_label_statistics()

    return ConcatDataset(datasets)


def get_training_set_both(dataset_path: str, image_size: ImageSize):
    num_joints = 15
    left_indexes: List[int] = [3, 4, 5, 9, 10, 11]
    right_indexes: List[int] = [6, 7, 8, 12, 13, 14]

    datasets: List[EhpiLSTMDataset] = [
        EhpiLSTMDataset(os.path.join(dataset_path, "JOURNAL_2019_03_POSEALGO_30fps"),
                        transform=transforms.Compose([
                            RemoveJointsOutsideImgEhpi(image_size),
                            ScaleEhpi(image_size),
                            TranslateEhpi(image_size),
                            FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                            NormalizeEhpi(image_size)
                        ]), num_joints=num_joints),
        EhpiLSTMDataset(os.path.join(dataset_path, "JOURNAL_2019_03_GT_30fps"),
                        transform=transforms.Compose([
                            RemoveJointsOutsideImgEhpi(image_size),
                            ScaleEhpi(image_size),
                            TranslateEhpi(image_size),
                            FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                            NormalizeEhpi(image_size)
                        ]), num_joints=num_joints),
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
    journal_dataset_path = "/media/disks/beta/datasets/ehpi"
    model_dir = "/media/disks/beta/models/ehpi_journal_2019_03_v2"
    batch_size = 256
    seeds = [0, 104, 123, 142, 200]
    datasets = {
        "gt": get_training_set_gt,
        "pose": get_training_posealgo,
        "both": get_training_set_both
    }
    for seed in seeds:
        for dataset_name, get_dataset in datasets.items():
            set_seed(seed)
            train_set = get_dataset(journal_dataset_path, ImageSize(1280, 720))
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)

            # config
            train_config = TrainingConfigBase("ehpi_journal_2019_03_{}_seed_{}".format(dataset_name, seed), model_dir)
            train_config.weight_decay = 0
            train_config.num_epochs = 200

            trainer = TrainerEhpi()
            trainer.train(train_loader, train_config, model=EhpiLSTM(15, 5))
