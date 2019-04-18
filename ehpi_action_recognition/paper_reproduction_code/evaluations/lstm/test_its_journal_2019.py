import os

import numpy as np
from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset, NormalizeEhpi, \
    RemoveJointsOutsideImgEhpi
from nobos_torch_lib.models.detection_models.shufflenet_v2 import ShuffleNetV2
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms

from ehpi_action_recognition.evaluations.tester_ehpi import TesterEhpi
from ehpi_action_recognition.paper_reproduction_code.datasets.ehpi_lstm_dataset import EhpiLSTMDataset
from ehpi_action_recognition.paper_reproduction_code.models.ehpi_lstm import EhpiLSTM


def get_test_set_lab(image_size: ImageSize):
    num_joints = 15
    datasets = [
    EhpiLSTMDataset("/media/disks/beta/datasets/ehpi/JOURNAL_2019_03_TEST_VUE01_30FPS",
                             transform=transforms.Compose([
                                 RemoveJointsOutsideImgEhpi(image_size),
                                 # ScaleEhpi(image_size),
                                 # TranslateEhpi(image_size),
                                 # FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                                 NormalizeEhpi(image_size)
                             ]), num_joints=num_joints, dataset_part=DatasetPart.TEST),
    EhpiLSTMDataset("/media/disks/beta/datasets/ehpi/JOURNAL_2019_03_TEST_VUE02_30FPS",
                             transform=transforms.Compose([
                                 RemoveJointsOutsideImgEhpi(image_size),
                                 # ScaleEhpi(image_size),
                                 # TranslateEhpi(image_size),
                                 # FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                                 NormalizeEhpi(image_size)
                             ]), num_joints=num_joints, dataset_part=DatasetPart.TEST),
    ]
    for dataset in datasets:
        dataset.print_label_statistics()
    return ConcatDataset(datasets)

def get_test_set_office(image_size: ImageSize):
    num_joints = 15
    dataset = EhpiLSTMDataset("/media/disks/beta/datasets/ehpi/JOURNAL_2019_04_TEST_EVAL2_30FPS",
                             transform=transforms.Compose([
                                 RemoveJointsOutsideImgEhpi(image_size),
                                 # ScaleEhpi(image_size),
                                 # TranslateEhpi(image_size),
                                 # FlipEhpi(left_indexes=left_indexes, right_indexes=right_indexes),
                                 NormalizeEhpi(image_size)
                             ]), num_joints=num_joints, dataset_part=DatasetPart.TEST)
    dataset.print_label_statistics()
    return dataset

if __name__ == '__main__':
    model_names = [
        "ehpi_journal_2019_03_gt_seed_0_cp0200",
        "ehpi_journal_2019_03_gt_seed_104_cp0200",
        "ehpi_journal_2019_03_gt_seed_123_cp0200",
        "ehpi_journal_2019_03_gt_seed_142_cp0200",
        "ehpi_journal_2019_03_gt_seed_200_cp0200",
        #
        "ehpi_journal_2019_03_pose_seed_0_cp0200",
        "ehpi_journal_2019_03_pose_seed_104_cp0200",
        "ehpi_journal_2019_03_pose_seed_123_cp0200",
        "ehpi_journal_2019_03_pose_seed_142_cp0200",
        "ehpi_journal_2019_03_pose_seed_200_cp0200",
        #
        "ehpi_journal_2019_03_both_seed_0_cp0200",
        "ehpi_journal_2019_03_both_seed_104_cp0200",
        "ehpi_journal_2019_03_both_seed_123_cp0200",
        "ehpi_journal_2019_03_both_seed_142_cp0200",
        "ehpi_journal_2019_03_both_seed_200_cp0200",
    ]
    # Test set
    test_set = get_test_set_lab(ImageSize(1280, 720))
    # test_set = get_test_set_office(ImageSize(1280, 720))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    result_path = "/media/disks/beta/dump/its_journal_experiment_results/lab"

    for model_name in model_names:
        print("Model name: {}".format(model_name))
        weights_path = "/media/disks/beta/models/ehpi_journal_2019_03_v2/{}.pth".format(model_name)

        tester = TesterEhpi()
        ehpi_results, seq_results = tester.test(test_loader, weights_path, model=EhpiLSTM(15, 5))
        ehpi_results_np = np.array(ehpi_results, dtype=np.uint32)
        seq_results_np = np.array(seq_results, dtype=np.uint32)
        np.save(os.path.join(result_path, "{}_ehpis".format(model_name)), ehpi_results_np)
        np.save(os.path.join(result_path, "{}_seqs".format(model_name)), seq_results_np)
