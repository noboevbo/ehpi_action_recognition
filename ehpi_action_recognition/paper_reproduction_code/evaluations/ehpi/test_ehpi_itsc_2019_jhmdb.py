from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset, NormalizeEhpi, \
    RemoveJointsOutsideImgEhpi
from nobos_torch_lib.models.action_recognition_models.ehpi_small_net import EHPISmallNet
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ehpi_action_recognition.evaluations import tester_ehpi


def get_test_set(image_size: ImageSize):
    num_joints = 15
    return EhpiDataset("/media/disks/beta/datasets/ehpi/JHMDB_ITSC-1-GT/",
                       transform=transforms.Compose([
                           RemoveJointsOutsideImgEhpi(image_size),
                           NormalizeEhpi(image_size)
                       ]), dataset_part=DatasetPart.TEST, num_joints=num_joints)


if __name__ == '__main__':
    weights_path = "/media/disks/beta/models/tests_for_repo/ehpi_jhmdb_0_split_1_cp0140.pth"

    # Test set
    test_set = get_test_set(ImageSize(320, 240))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    tester = tester_ehpi()
    tester.test(test_loader, weights_path, model=EHPISmallNet(21))
