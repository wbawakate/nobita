import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from nobita import VisionPipeline
from nobita.modules import FaceDetection, EmotionEstimation

emotions = ["neutral", "happy", "sad", "surprise", "anger"]
with VisionPipeline(
    modules=[FaceDetection(), EmotionEstimation()], use_oak=True
) as pipeline:
    time_start = time.time()
    while True:
        out_frame = pipeline.get()
        time_now = time.time() - time_start
        if out_frame["EmotionEstimation"]:
            print(
                time_now,
                "[sec], ",
                emotions[np.argmax(out_frame["EmotionEstimation"][0])],
            )
        if time_now > 10:
            break
