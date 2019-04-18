import os
import torch.backends.cudnn

import nobos_commons.tools.config as nobos_tools_config

from nobos_commons.data_structures.configs.cache_config import CacheConfig
from nobos_commons.data_structures.configs.pose_visualization_config import PoseVisualizationConfig
from nobos_torch_lib.configs.detection_model_configs.yolo_v3_config import YoloV3Config
from nobos_torch_lib.configs.pose_estimation_2d_model_configs.pose_resnet_model_config import PoseResNetModelConfig

curr_dir = os.path.dirname(os.path.realpath(__file__))
# Cache Config
cache_config = CacheConfig(cache_dir="/media/disks/beta/pydata/pycache/ofp_app",
                           func_names_to_reload=[],
                           reload_all=False)
nobos_tools_config.cfg.cache_config = cache_config

# Torch Backend Settings
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

# Models Configs
# Pose Recognition
pose_resnet_config = PoseResNetModelConfig()
pose_resnet_config.model_state_file = "/media/disks/beta/pydata/ofp_app/models/pose/pose_resnet/pose_resnet_50_256x192.pth.tar"

# Object Detection
yolo_v3_config = YoloV3Config()
yolo_v3_config.network_config_file = os.path.join(curr_dir, 'configs', 'yolo_v3.cfg')
yolo_v3_config.model_state_file = "/media/disks/beta/pydata/ofp_app/models/detection/yolov3.weights"
yolo_v3_config.resolution = 160

# Visualization
pose_visualization_config = PoseVisualizationConfig()
