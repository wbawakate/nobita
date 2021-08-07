import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nobita import nobita_vision
from nobita.modules import FaceDetection
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
    modules=[FaceDetection()],
    use_oak=use_oak,
    arg_vid_cap=arg_vid_cap,
)
pipeline.set_pipline_to_modules()
pipeline.set_device_to_modules()
h_display = 400

while True:
    out_frame = pipeline.get()
    if out_frame["FaceDetection"]:
        cv2.imshow("BBox of Face Detection", out_frame["FaceDetection"][0])
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        break
pipeline.close()
