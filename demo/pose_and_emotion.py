import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nobita import nobita_vision
from nobita.modules import PoseEstimation, EmotionEstimation

emotions = ["neutral", "happy", "sad", "surprise", "anger"]


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
preview_size = (400, 400)

pipeline = nobita_vision.VisionPipeline(
    modules=[PoseEstimation(), EmotionEstimation()],
    use_oak=use_oak,
    arg_vid_cap=arg_vid_cap,
    preview_size=preview_size,
)
pipeline.set_pipline_to_modules()
pipeline.set_device_to_modules()
h_display = 900

while True:
    # Dict[List[np.array]]. np.arrayは画像 or ニューラルネットワークの予測確率
    out_frame = pipeline.get()

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

    if isinstance(out_frame["PoseEstimation"], np.ndarray) and out_frame["CAM"]:
        frame = out_frame["CAM"]
        yx_pose = out_frame["PoseEstimation"]
        # print(out_frame["PoseEstimation"])
        if yx_pose is not None:
            h_size = frame[0].shape[0]
            w_size = frame[0].shape[1]
            w_display = int(w_size * h_display / h_size)
            face_cv2_frame = cv2.resize(
                cv2.UMat(frame[0].astype(np.uint8)), (w_display, h_display)
            )
            # ランドマーク描画
            for y, x, s in zip(yx_pose[:, 0], yx_pose[:, 1], yx_pose[:, 2]):
                if s > 0.2:
                    cv2.circle(
                        face_cv2_frame,
                        (int(x * w_display), int(y * h_display)),
                        3,
                        (10, 255, 10),
                        -1,
                    )
            cv2.imshow(f"pose ", face_cv2_frame)
        else:
            pass
            # print(f'no result')
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        break
pipeline.close()
