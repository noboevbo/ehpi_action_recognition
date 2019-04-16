import torch.backends.cudnn

import nobos_commons.tools.config as nobos_tools_config
from nobos_commons.data_structures.configs.cache_config import CacheConfig


from nobos_commons.data_structures.configs.pose_visualization_config import PoseVisualizationConfig
from nobos_torch_lib.configs.detection_model_configs.yolo_v3_config import YoloV3Config
from nobos_torch_lib.configs.pose_estimation_2d_model_configs.pose_resnet_model_config import PoseResNetModelConfig

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
pose_resnet_config = PoseResNetModelConfig()
yolo_v3_config = YoloV3Config()
yolo_v3_config.resolution = 160
pose_visualization_config = PoseVisualizationConfig()
