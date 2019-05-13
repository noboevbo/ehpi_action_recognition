import os

from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset, NormalizeEhpi, \
    RemoveJointsOutsideImgEhpi
from nobos_torch_lib.models.action_recognition_models.ehpi_small_net import EHPISmallNet
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ehpi_action_recognition.configs.config import ehpi_dataset_path, models_dir
from ehpi_action_recognition.tester_ehpi import TesterEhpi


def get_test_set(dataset_path: str, image_size: ImageSize):
    num_joints = 15
    return EhpiDataset(dataset_path,
                       transform=transforms.Compose([
                           RemoveJointsOutsideImgEhpi(image_size),
                           NormalizeEhpi(image_size)
                       ]), dataset_part=DatasetPart.TEST, num_joints=num_joints)


def test_gt():
    seeds = [0, 104, 123, 142, 200]
    for seed in seeds:
        print("Test JHMDB GT on seed: {}".format(seed))
        weights_path = os.path.join(models_dir, "jhmdb-1-gt", "ehpi_jhmdb_{}_split_1_cp0140.pth".format(seed))

        # Test set
        test_set = get_test_set(os.path.join(ehpi_dataset_path, "jhmdb", "JHMDB_ITSC-1-GT/"), ImageSize(320, 240))
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        tester = TesterEhpi()
        tester.test(test_loader, weights_path, model=EHPISmallNet(21))


def test_jhmdb():
    seeds = [0, 104, 123, 142, 200]
    for split in range(1, 4):
        for seed in seeds:
            print("Test JHMDB Split {} on seed: {}".format(split, seed))
            weights_path = os.path.join(models_dir, "jhmdb", "ehpi_jhmdb_{}_split_{}_cp0200.pth".format(seed, split))

            # Test set
            test_set = get_test_set(os.path.join(ehpi_dataset_path, "jhmdb", "JHMDB_ITSC-1-POSE/"), ImageSize(320, 240))
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

            tester = TesterEhpi()
            tester.test(test_loader, weights_path, model=EHPISmallNet(21))


if __name__ == '__main__':
    test_gt()
    test_jhmdb()
