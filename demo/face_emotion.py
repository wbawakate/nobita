import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nobita import nobita_vision
from nobita.modules import FaceDetection, EmotionEstimation
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="demo")
parser.add_argument("--device", "-d", type=str, default="-1", help="device id: If you use OAK-D, set -1 (default).  If you use other web camera, set its device id.")
args = parser.parse_args()


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


if is_integer(args.device):
    args.device = int(args.device)
    if args.device < 0:
        use_oak = True
        arg_vid_cap = args.device
    else:
        use_oak = False
        arg_vid_cap = args.device
else:
    use_oak = False
    arg_vid_cap = args.device

pipeline = nobita_vision.VisionPipeline(
    modules=[EmotionEstimation()],
    use_oak=use_oak,
    arg_vid_cap=arg_vid_cap,
)
pipeline.set_pipline_to_modules()
pipeline.set_device_to_modules()
h_display = 400
emotions = ["neutral", "happy", "sad", "surprise", "anger"]
while True:
    out_frame = pipeline.get()
    # Dict[List[np.array]]. np.arrayは画像 or ニューラルネットワークの予測確率
    if out_frame["FaceDetection"] and out_frame["EmotionEstimation"]:
        print("n_faces", len(out_frame["FaceDetection"]))
        for i, (cam, face, prob) in enumerate(
            zip(
                out_frame["CAM"],
                out_frame["FaceDetection"],
                out_frame["EmotionEstimation"],
            )
        ):
            if face is not None and prob is not None:
                id_emo = np.argmax(prob)
                # print("prob", prob)
                max_prob = np.max(prob)
                # print(id_emo, max_prob)
                # print(f"{i} {emotions[id_emo]} {max_prob:.1%}")
                h_size = face.shape[0]
                w_size = face.shape[1]
                face_cv2_frame = cv2.UMat(cam)
                cv2.putText(
                    face_cv2_frame,
                    f"{i} {emotions[id_emo]} {max_prob:.1%}",
                    (20, 50),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1.0,
                    (255, 255, 255),
                )
                cv2.imshow(f"face: {i}", face_cv2_frame)
            else:
                pass
                # print(f'no result')
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        break
pipeline.close()
