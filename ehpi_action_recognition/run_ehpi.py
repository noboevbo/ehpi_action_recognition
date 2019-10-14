# Live Test ehpi ofp prod
from collections import deque
from operator import itemgetter
from typing import Dict, List, Tuple

from ehpi_action_recognition.config import pose_resnet_config, pose_visualization_config, ehpi_model_state_file

import cv2
import numpy as np
import torch.utils.data.distributed
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.humans_metadata.action import Action
from nobos_commons.data_structures.image_content import ImageContent
from nobos_commons.data_structures.image_content_buffer import ImageContentBuffer
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_joint_config import \
    get_joints_jhmdb
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_producer_ehpi import \
    FeatureVecProducerEhpi
from nobos_commons.input_providers.camera.webcam_provider import WebcamProvider
from nobos_commons.tools.fps_tracker import FPSTracker
from nobos_commons.tools.log_handler import logger
from nobos_commons.tools.pose_tracker import PoseTracker
from nobos_commons.visualization.detection_visualizer import draw_bb
from nobos_commons.visualization.pose2d_visualizer import get_human_pose_image
from nobos_torch_lib.models.detection_models.shufflenet_v2 import ShuffleNetV2
from nobos_torch_lib.models.pose_estimation_2d_models import pose_resnet
from scipy.special import softmax

from ehpi_action_recognition.configurator import setup_application
from ehpi_action_recognition.networks.action_recognition_nets.action_rec_net_ehpi import ActionRecNetEhpi
from ehpi_action_recognition.networks.pose_estimation_2d_nets.pose2d_net_resnet import Pose2DNetResnet

action_save: Dict[str, List[List[float]]] = {}


def get_stabelized_action_recognition(human_id: str, action_probabilities: np.ndarray):
    queue_size = 20
    if human_id not in action_save:
        action_save[human_id] = []
        action_save[human_id].append(deque([0] * queue_size, maxlen=queue_size))
        action_save[human_id].append(deque([0] * queue_size, maxlen=queue_size))
        action_save[human_id].append(deque([0] * queue_size, maxlen=queue_size))
    argmax = np.argmax(action_probabilities)
    # argmax_val = action_probabilities[argmax]
    # # new_factor = 0 if argmax_val < 0 else argmax_val
    for i in range(0, 3):
        if i == argmax:
            action_save[human_id][i].append(1)
        else:
            action_save[human_id][i].append(0)
    return [sum(action_save[human_id][0]) / queue_size,
            sum(action_save[human_id][1]) / queue_size,
            sum(action_save[human_id][2]) / queue_size]


def argmax(items):
    index, element = max(enumerate(items), key=itemgetter(1))
    return index, element


if __name__ == '__main__':
    setup_application()
    # Settings
    skeleton_type = SkeletonStickman
    image_size = ImageSize(width=640, height=360)
    heatmap_size = ImageSize(width=64, height=114)
    camera_number = 0
    fps = 30
    buffer_size = 20
    action_names = [Action.IDLE.name, Action.WALK.name, Action.WAVE.name]
    use_action_recognition = True
    use_quick_n_dirty = False

    # Input Provider
    input_provider = WebcamProvider(camera_number=0, image_size=image_size, fps=fps)
    # input_provider = ImgDirProvider(
    #     "/media/disks/beta/records/real_cam/2019_03_13_Freilichtmuseum_Dashcam_01/full",
    #     image_size=image_size, fps=fps)
    fps_tracker = FPSTracker(average_over_seconds=1)

    # Pose Network
    pose_model = pose_resnet.get_pose_net(pose_resnet_config)

    logger.info('=> loading model from {}'.format(pose_resnet_config.model_state_file))
    pose_model.load_state_dict(torch.load(pose_resnet_config.model_state_file))
    pose_model = pose_model.cuda()
    pose_model.eval()
    pose_net = Pose2DNetResnet(pose_model, skeleton_type)
    pose_tracker = PoseTracker(image_size=image_size, skeleton_type=skeleton_type)

    # Action Network
    action_model = ShuffleNetV2(input_size=32, n_class=3)
    state_dict = torch.load(ehpi_model_state_file)
    action_model.load_state_dict(state_dict)
    action_model.cuda()
    action_model.eval()
    feature_vec_producer = FeatureVecProducerEhpi(image_size,
                                                  get_joints_func=lambda skeleton: get_joints_jhmdb(skeleton))
    action_net = ActionRecNetEhpi(action_model, feature_vec_producer, image_size)

    # Content Buffer
    image_content_buffer: ImageContentBuffer = ImageContentBuffer(buffer_size=buffer_size)

    counts: Dict[str, int] = {}
    for frame_nr, frame in enumerate(input_provider.get_data()):
        last_humans = image_content_buffer.get_last_humans()
        humans = []
        object_bounding_boxes = []
        if not use_quick_n_dirty or last_humans is None or len(last_humans) == 0:
            object_bounding_boxes = pose_net.detector.get_object_bounding_boxes(frame)
            human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
            humans = pose_net.get_humans_from_bbs(frame, human_bbs)

        humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans,
                                                                        previous_humans=last_humans)

        redetected_humans = pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
        humans.extend(redetected_humans)

        human_bbs = [human.bounding_box for human in humans]
        other_bbs = [bb for bb in object_bounding_boxes if bb.label != "person"]

        img = get_human_pose_image(frame, humans,
                                   min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show)

        # bbs_to_draw = [bb for bb in human_data.bbs if bb.label == "person"]
        image_content = ImageContent(humans=humans, objects=human_bbs)

        image_content_buffer.add(image_content)
        actions: Dict[str, Tuple[Action, float]] = {}
        if use_action_recognition:
            action_results = action_net.get_actions(humans, frame_nr)
            for human_id, action_logits in action_results.items():
                action_probabilities = softmax(action_logits)
                actions[human_id] = []
                predictions = get_stabelized_action_recognition(human_id, action_probabilities)
                print(predictions)
                pred_label, probability = argmax(predictions)
                actions[human_id] = (Action[action_names[pred_label]], probability)

        # --------- Draw objects
        bbs_to_draw = [bb for bb in image_content.objects if bb.label != "person"]
        list(map(lambda bb: draw_bb(img, bb), bbs_to_draw))

        list(map(lambda human: draw_bb(img, human.bounding_box, actions[human.uid][0].name if human.uid in actions else "None"), image_content.humans))

        img = cv2.resize(img, (1280, 720))

        cv2.imshow('webcam', img)
        # cv2.imwrite(os.path.join(get_create_path("/media/disks/beta/records/with_visualizations/2019_03_13_Freilichtmuseum_Demo"), "{}.png".format(str(frame_nr).zfill(5))), img)
        # cv2.imwrite(os.path.join(get_create_path(
        #     "/media/disks/beta/records/with_visualizations/2019_03_13_Freilichtmuseum_Dashcam_Test_FASTMODE"),
        #                          "{}.jpg".format(str(frame_nr).zfill(5))), img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        fps_tracker.print_fps()
