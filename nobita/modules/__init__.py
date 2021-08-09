from typing import Union

from .base_module import Base
from .upstream import FaceDetection, PoseEstimation
from .downstream import EmotionEstimation, AgeGender, FaceLandmark, HeadPose

NobitaModuleType = Union[
    Base,
    FaceDetection,
    PoseEstimation,
    EmotionEstimation,
    AgeGender,
    FaceLandmark,
    HeadPose,
]
