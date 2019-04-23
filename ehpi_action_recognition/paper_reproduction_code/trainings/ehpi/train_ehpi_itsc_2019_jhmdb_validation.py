import os
import random
from copy import copy
from typing import List

import numpy as np
import torch
from nobos_commons.data_structures.dimension import ImageSize
from nobos_torch_lib.configs.training_configs.training_config_base import TrainingConfigBase
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset, FlipEhpi, ScaleEhpi, \
    TranslateEhpi, NormalizeEhpi, RemoveJointsOutsideImgEhpi, RemoveJointsEhpi
from nobos_torch_lib.datasets.samplers.imbalanced_dataset_sampler import ImbalancedDatasetSampler
from nobos_torch_lib.learning_rate_schedulers.learning_rate_scheduler_stepwise import \
    LearningRateSchedulerStepwise
from nobos_torch_lib.models.action_recognition_models.ehpi_small_net import EHPISmallNet
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from ehpi_action_recognition.config import models_dir, ehpi_dataset_path
from ehpi_action_recognition.trainer_ehpi import TrainerEhpi

foot_indexes: List[int] = [11, 14]
knee_indexes: List[int] = [10, 13]


def get_training_set(dataset_path: str, image_size: ImageSize):
    num_joints = 15
    left_indexes: List[int] = [3, 4, 5, 9, 10, 11]
    right_indexes: List[int] = [6, 7, 8, 12, 13, 14]
    return EhpiDataset(os.path.join(dataset_path, "JHMDB_ITSC-1/"),
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
    batch_sizes = [64, 128]
    weight_decays = [5e-4, 5e-3]
    lrs = [0.05, 0.01]

    # balances = [True, False]
    for seed in seeds:
        for batch_size in batch_sizes:
            for lr in lrs:
                for weight_decay in weight_decays:
                    print("Seed: {}, Batchsize: {}, LR: {}, Weight Decay: {}".format(seed, batch_size, lr, weight_decay))
                    set_seed(0)  # FIXED SEED FOR DATASET SPLIT!

                    # Load full dataset
                    train_full_set = get_training_set(os.path.join(ehpi_dataset_path, "jhmdb"), image_size)
                    test_full_set = copy(train_full_set)
                    test_full_set.transform = transforms.Compose([
                        RemoveJointsOutsideImgEhpi(image_size),
                        NormalizeEhpi(image_size)
                    ])

                    # Create train and validation splits
                    train_indices, val_indices = train_full_set.get_subsplit_indices(validation_percentage=0.3)

                    # Train set
                    train_set = Subset(train_full_set, train_indices)

                    # Validation set
                    val_set = Subset(test_full_set, val_indices)
                    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

                    set_seed(seed)

                    # Dataset Sampler
                    sampler = ImbalancedDatasetSampler(train_set, dataset_type=EhpiDataset)
                    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)

                    # config
                    train_config = TrainingConfigBase("ehpi_jhmdb_{}".format(seed), os.path.join(models_dir, "val_jhmdb"))
                    train_config.learning_rate = lr
                    train_config.learning_rate_scheduler = LearningRateSchedulerStepwise(lr_decay=0.1, lr_decay_epoch=50)
                    train_config.weight_decay = weight_decay
                    train_config.num_epochs = 350
                    train_config.checkpoint_epoch = 10

                    trainer = TrainerEhpi()
                    losses, accuracies = trainer.train(train_loader, train_config, test_loader=val_loader, model=EHPISmallNet(21))

                    with open("losses_seed_{}.txt".format(seed), 'a') as the_file:
                        for loss in losses:
                            the_file.write("{}\n".format(loss))
                    with open("accuracies_seed_{}.txt".format(seed), 'a') as the_file:
                        for accuracy in accuracies:
                            the_file.write("{}\n".format(accuracy))
