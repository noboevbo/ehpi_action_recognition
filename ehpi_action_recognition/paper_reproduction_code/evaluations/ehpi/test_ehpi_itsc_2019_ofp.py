from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset, NormalizeEhpi, \
    RemoveJointsOutsideImgEhpi
from nobos_torch_lib.models.detection_models.shufflenet_v2 import ShuffleNetV2
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ehpi_action_recognition.evaluations.tester_ehpi import TesterEhpi


def get_test_set(image_size: ImageSize):
    num_joints = 15
    return EhpiDataset("/media/disks/beta/datasets/ehpi/2019_03_13_Freilichtmuseum_30FPS",
                       transform=transforms.Compose([
                           RemoveJointsOutsideImgEhpi(image_size),
                           NormalizeEhpi(image_size)
                       ]), dataset_part=DatasetPart.TEST, num_joints=num_joints)


if __name__ == '__main__':
    model_names = [
        # "itsc2019_full_seed_0_cp0140",
        # # "itsc2019_sim_seed_0_cp0140",
        # # "itsc2019_sim_gt_only_seed_0_cp0140",
        # # "itsc2019_sim_pose_algo_only_seed_0_cp0140",
        # # "itsc2019_wo_sim_seed_0_cp0140",

        "itsc2019_full_seed_104_cp0140",
        # "itsc2019_sim_seed_104_cp0140",
        # "itsc2019_sim_gt_only_seed_104_cp0140",
        # "itsc2019_sim_pose_algo_only_seed_104_cp0140",
        # "itsc2019_wo_sim_seed_104_cp0140"
    ]
    # Test set
    test_set = get_test_set(ImageSize(1280, 720))
    test_set.print_label_statistics()
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    for model_name in model_names:
        print("Model name: {}".format(model_name))
        weights_path = "/media/disks/beta/models/itsc_2019_v2/{}.pth".format(model_name)

        tester = TesterEhpi()
        tester.test(test_loader, weights_path, model=ShuffleNetV2(3))
