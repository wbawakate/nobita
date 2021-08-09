import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from nobita import VisionPipeline
from nobita.modules import FaceDetection, EmotionEstimation

emotions = ["neutral", "happy", "sad", "surprise", "anger"]
# prepare VisionPipeline of nobita
with VisionPipeline(modules=[FaceDetection(), EmotionEstimation()], use_oak=True) as pipeline:
    while True:
        # get result of estimation and camera preview
        out_frame = pipeline.get()
        if out_frame["FaceDetection"] :
            # facial image by cropped by face detection
            face_cv2_frame = cv2.UMat(out_frame["FaceDetection"][0] ) 
            if out_frame["EmotionEstimation"]:
                    id_emo = np.argmax(out_frame["EmotionEstimation"][0])
                    max_prob = np.max(out_frame["FaceDetection"][0])
                    # put estimated emotion on a facial image as text
                    cv2.putText(face_cv2_frame, f"{emotions[id_emo]}",(5, 15),cv2.FONT_HERSHEY_TRIPLEX,0.6,(255,0,0))
            # show the facial image
            cv2.imshow(f"face 0", face_cv2_frame)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break