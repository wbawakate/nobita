import argparse
import os
import sys

import cv2
from matplotlib.pyplot import yscale
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nobita import nobita_vision
from nobita.modules import PoseEstimation

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
    modules=[PoseEstimation()],
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
    if out_frame["PoseEstimation"] and out_frame["CAM"]:
        frame = out_frame["CAM"]
        yx_pose = out_frame["PoseEstimation"][0]
        if yx_pose is not None:
            h_size = frame[0].shape[0]
            w_size = frame[0].shape[1]
            w_display = int(w_size * h_display / h_size)
            face_cv2_frame = cv2.resize(
                cv2.UMat(frame[0].astype(np.uint8)), (w_display, h_display)
            )
            # ランドマーク描画
            # display landmark
            for y, x, s in zip(yx_pose[:, 0], yx_pose[:, 1], yx_pose[:, 2]):
                if s > 0.1:
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
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        break
pipeline.close()
