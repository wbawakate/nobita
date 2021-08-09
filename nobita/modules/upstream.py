import os
import sys
from typing import List, Tuple, Optional, Union
import numpy as np

from .base_module import Base
from .utils import frame_norm
import cv2

# from matplotlib import pyplot as plt


class FaceDetection(Base):
    def __init__(
        self,
        blob_path: Optional[str] = None,
        input_size: Tuple[int, int] = (300, 300),
        from_oak=True,
    ):
        if blob_path is None:
            here_path = __file__
            dirname = os.path.abspath(os.path.dirname(here_path))
            blob_path = os.path.join(
                dirname,
                "models",
                "face-detection-retail-0004_openvino_2021.2_6shave.blob",
            )
        super(FaceDetection, self).__init__(
            blob_path, input_size=input_size, from_oak=from_oak
        )
        self.source_name: str = "CAM"

    def _get_ndarray(self, frame: np.ndarray) -> List[np.ndarray]:
        bboxes:np.ndarray = np.array(self.queue_nn.get().getFirstLayerFp16())
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        bboxes = bboxes[bboxes[:, 2] > self.threshold][:, 3:7]
        outlist = []
        for raw_bbox in bboxes:
            bbox = frame_norm(frame, raw_bbox)
            det_frame = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            outlist.append(det_frame)
        return outlist


class FacePosition(Base):
    def __init__(
        self,
        blob_path: Optional[str] = None,
        input_size: Tuple[int, int] = (300, 300),
        from_oak=True,
    ):
        if blob_path is None:
            here_path = __file__
            dirname = os.path.abspath(os.path.dirname(here_path))
            blob_path = os.path.join(
                dirname,
                "models",
                "face-detection-retail-0004_openvino_2021.2_6shave.blob",
            )
        super(FacePosition, self).__init__(
            blob_path, input_size=input_size, from_oak=from_oak
        )
        self.source_name: str = "CAM"


    def _get_ndarray(self, frame: np.ndarray) -> List[np.ndarray]:
        bboxes = np.array(self.queue_nn.get().getFirstLayerFp16())
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        bboxes = bboxes[bboxes[:, 2] > self.threshold][:, 3:7]
        return [raw_bbox for raw_bbox in bboxes]


class PoseEstimation(Base):
    def __init__(
        self,
        blob_path: Optional[str] = None,
        input_size: Tuple[int, int] = (192, 192),
        from_oak=True,
    ):
        if blob_path is None:
            here_path = __file__
            dirname = os.path.abspath(os.path.dirname(here_path))
            blob_path = os.path.join(
                dirname,
                "models",
                "movenet_singlepose_lightning_U8_transpose.blob",
            )
        super(PoseEstimation, self).__init__(
            blob_path, input_size=input_size, from_oak=from_oak
        )
        self.source_name = "CAM"

    def _get_ndarray(self, frame: np.ndarray) -> List[np.ndarray]:
        kps = np.array(self.queue_nn.get().getLayerFp16("Identity")).reshape(
            -1, 3
        )  # 17x3
        return [kps]
