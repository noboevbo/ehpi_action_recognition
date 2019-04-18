import random
from typing import List

import numpy as np
import torch
from nobos_commons.data_structures.dimension import ImageSize
from nobos_torch_lib.configs.training_configs.training_config_base import TrainingConfigBase
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset, FlipEhpi, ScaleEhpi, \
    TranslateEhpi, NormalizeEhpi, RemoveJointsOutsideImgEhpi, RemoveJointsEhpi
from nobos_torch_lib.datasets.samplers.imbalanced_dataset_sampler import ImbalancedDatasetSampler
from nobos_torch_lib.learning_rate_schedulers.learning_rate_scheduler_expotential import \
    LearningRateSchedulerExpotential
from nobos_torch_lib.models.action_recognition_models.ehpi_small_net import EHPISmallNet
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ehpi_action_recognition.trainings import trainer_ehpi

foot_indexes: List[int] = [11, 14]
knee_indexes: List[int] = [10, 13]


def get_training_set(image_size: ImageSize):
    num_joints = 15
    left_indexes: List[int] = [3, 4, 5, 9, 10, 11]
    right_indexes: List[int] = [6, 7, 8, 12, 13, 14]
    return EhpiDataset("/media/disks/beta/datasets/ehpi/JHMDB_ITSC-1-GT/",
                       transform=transforms.Compose([
                           RemoveJointsOutsideImgEhpi(image_size),
                           RemoveJointsEhpi(indexes_to_remove=foot_indexes, indexes_to_remove_2=knee_indexes,
                                            probability=0.25),
                           ScaleEhpi(image_size),
                           TranslateEhpi(image_size),
                           FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                           NormalizeEhpi(image_size)
                       ]), num_joints=num_joints)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(0)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    image_size = ImageSize(320, 240)
    seeds = [0, 104, 123, 142, 200]
    batch_size = 64
    weight_decay = 5e-4
    lr = 0.05

    # balances = [True, False]
    for seed in seeds:
        print("Seed: {}".format(seed))
        set_seed(seed)

        # Train set
        train_set = get_training_set(image_size)
        train_set.print_label_statistics()
        sampler = ImbalancedDatasetSampler(train_set, dataset_type=EhpiDataset)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)

        # config
        train_config = TrainingConfigBase("ehpi_jhmdb_{}_split_1".format(seed), "/media/disks/beta/models/tests_for_repo")
        train_config.learning_rate = lr
        train_config.learning_rate_scheduler = LearningRateSchedulerExpotential(lr_decay=0.1, lr_decay_epoch=50)
        train_config.weight_decay = weight_decay
        train_config.num_epochs = 140
        train_config.checkpoint_epoch = 140

        trainer = trainer_ehpi()
        trainer.train(train_loader, train_config, model=EHPISmallNet(21))
