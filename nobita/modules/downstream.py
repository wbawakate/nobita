import os
import sys
from typing import List, Tuple, Optional
import numpy as np

from .base_module import Base
from .utils import frame_norm


class EmotionEstimation(Base):
    def __init__(
        self,
        blob_path: Optional[str] = None,
        input_size: Tuple[int, int] = (64, 64),
    ):
        if blob_path is None:
            here_path = __file__
            dirname = os.path.abspath(os.path.dirname(here_path))
            blob_path = os.path.join(
                dirname, "models", "emotions-recognition-retail-0003.blob"
            )
        super(EmotionEstimation, self).__init__(
            blob_path, input_size=input_size, from_oak=False
        )
        self.from_oak = False
        self.source_name = "FaceDetection"

    def _get_ndarray(self, frame: np.ndarray)-> List[np.ndarray]:
        prob_4d = np.array(self.queue_nn.get().getFirstLayerFp16())
        # 1 x 5 x 1 x1
        #  ('neutral', 'happy', 'sad', 'surprise', 'anger')
        return prob_4d


class FaceLandmark(Base):
    def __init__(
        self,
        blob_path: Optional[str] = None,
        input_size: Tuple[int, int] = (60, 60),
    ):
        if blob_path is None:
            here_path = __file__
            dirname = os.path.abspath(os.path.dirname(here_path))
            blob_path = os.path.join(
                dirname, "models", "facial-landmarks-35-adas-0002-shaves6.blob"
            )
        super(FaceLandmark, self).__init__(
            blob_path, input_size=input_size, from_oak=False
        )
        self.from_oak = False
        self.source_name = "FaceDetection"

    def _get_ndarray(self, frame: np.ndarray)-> List[np.ndarray]:
        pos = np.array(self.queue_nn.get().getFirstLayerFp16())
        # The net outputs a blob with the shape: [1, 70], containing row-vector of 70 floating point values for 35 landmarks' normed coordinates in the form (x0, y0, x1, y1, ..., x34, y34).
        # Output layer name in Inference Engine format:
        #   align_fc3
        # Output layer name in Caffe* format:
        #       align_fc3
        return pos


class HeadPose(Base):
    def __init__(
        self,
        blob_path: Optional[str] = None,
        input_size: Tuple[int, int] = (60, 60),
    ):
        if blob_path is None:
            here_path = __file__
            dirname = os.path.abspath(os.path.dirname(here_path))
            blob_path = os.path.join(
                dirname, "models", "head-pose-estimation-adas-0001-shaves4.blob"
            )
        super(HeadPose, self).__init__(blob_path, input_size=input_size, from_oak=False)
        self.from_oak = False
        self.source_name = "FaceDetection"

    def _get_ndarray(self, frame: np.ndarray)-> List[np.ndarray]:
        # print(self.queue_nn.get().getAllLayerNames())
        # print(self.queue_nn.getOutputs)

        # roll pitch yaw
        # print('angle_y_fc', angle_y_fc)
        nn_data = self.queue_nn.get()
        angle_p_fc = np.array(nn_data.getLayerFp16("angle_p_fc")) / 360 * 2 * np.pi
        angle_r_fc = np.array(nn_data.getLayerFp16("angle_r_fc")) / 360 * 2 * np.pi
        angle_y_fc = np.array(nn_data.getLayerFp16("angle_y_fc")) / 360 * 2 * np.pi
        return np.hstack([angle_r_fc, angle_p_fc, angle_y_fc])


class AgeGender(Base):
    def __init__(
        self,
        blob_path: Optional[str] = None,
        input_size: Tuple[int, int] = (62, 62),
    ):
        if blob_path is None:
            here_path = __file__
            dirname = os.path.abspath(os.path.dirname(here_path))
            blob_path = os.path.join(
                dirname,
                "models",
                "age-gender-recognition-retail-0013_openvino_2021.2_6shave.blob",
            )
        super(AgeGender, self).__init__(
            blob_path, input_size=input_size, from_oak=False
        )
        self.from_oak = False
        self.source_name = "FaceDetection"

    def _get_ndarray(self, frame: np.ndarray)-> List[np.ndarray]:
        nn_data = self.queue_nn.get()
        age = np.array(nn_data.getLayerFp16("age_conv3")) * 100
        gender = np.array(nn_data.getLayerFp16("prob"))
        # name: "age_conv3", shape: [1, 1, 1, 1] - Estimated age divided by 100.
        # name: "prob", shape: [1, 2, 1, 1] - Softmax output across 2 type classes [female, male]
        # sys.exit()
        return np.hstack([age, gender])
