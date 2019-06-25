import torch
import cv2
from typing import List

from nobos_commons.data_structures.bounding_box import BoundingBox
from nobos_commons.data_structures.constants.detection_classes import COCO_CLASSES
from nobos_commons.data_structures.dimension import Coord2D
from nobos_torch_lib.configs.detection_model_configs.yolo_v3_config import YoloV3Config
from nobos_torch_lib.models.detection_models.yolo_v3 import Darknet
from nobos_torch_lib.utils.yolo_helper import write_results
from torch.autograd import Variable

from ehpi_action_recognition.config import yolo_v3_config

import numpy as np


class DetectionNetYoloV3(object):
    __slots__ = ['model', 'num_classes']

    def __init__(self, model: Darknet):
        self.model = model
        self.num_classes = len(COCO_CLASSES)
        self.model.net_info["width"] = self.model.net_info["height"] = yolo_v3_config.resolution

        assert yolo_v3_config.resolution % 32 == 0
        assert yolo_v3_config.resolution > 32

        if yolo_v3_config.use_gpu:
            model.cuda()

        model.eval()

    def get_object_bounding_boxes(self, image: np.ndarray) -> List[BoundingBox]:
        network_input = self._get_network_input(image)
        if yolo_v3_config.use_gpu:
            network_input = network_input.cuda()
        output = self.model(Variable(network_input))
        output = write_results(output, yolo_v3_config.confidence, self.num_classes, nms=True,
                               nms_thresh=yolo_v3_config.nms_thresh)

        if output is 0:
            return []
        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(yolo_v3_config.resolution)) / yolo_v3_config.resolution

        #            im_dim = im_dim.repeat(output.size(0), 1)
        output[:, [1, 3]] *= image.shape[1]
        output[:, [2, 4]] *= image.shape[0]
        bbs: List[BoundingBox] = []
        for x in output:
            top_left = tuple(x[1:3].int())
            bottom_right = tuple(x[3:5].int())
            top_left = Coord2D(x=top_left[0].item(), y=top_left[1].item())
            bottom_right = Coord2D(x=bottom_right[0].item(), y=bottom_right[1].item())
            class_id = int(x[-1].cpu())
            class_label = "{0}".format(COCO_CLASSES[class_id])
            bbs.append(BoundingBox(top_left, bottom_right, label=class_label))
        return bbs

    @staticmethod
    def _get_network_input(image: np.ndarray):
        """
        Prepare image for inputting to the neural network.

        Returns a Variable
        """

        net_input_image = cv2.resize(image, (yolo_v3_config.resolution, yolo_v3_config.resolution))
        net_input_image = net_input_image[:, :, ::-1].transpose((2, 0, 1)).copy()
        net_input_image = torch.from_numpy(net_input_image).float().div(255.0).unsqueeze(0)
        return net_input_image


def get_default_detector(cfg: YoloV3Config) -> DetectionNetYoloV3:
    yolo_model = Darknet(cfg)
    yolo_model.load_weights(cfg.model_state_file)
    return DetectionNetYoloV3(model=yolo_model)

