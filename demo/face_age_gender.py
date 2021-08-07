import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nobita import nobita_vision
from nobita.modules import AgeGender, FaceDetection
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="demo")
parser.add_argument("--device", "-d", type=str, default="-1")
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
print("use OAK-D", use_oak)
with nobita_vision.VisionPipeline(
    modules=[AgeGender()], use_oak=use_oak, arg_vid_cap=arg_vid_cap
) as pipeline:

    h_display = 400
    print("created pipeline")
    while True:
        out_frame = (
            pipeline.get()
        )  # Dict[List[np.array]]. np.arrayは画像 or ニューラルネットワークの予測確率
        if out_frame["FaceDetection"] and out_frame["AgeGender"]:
            # print('n_faces', len(out_frame["FaceDetectionNN"]))
            for i, (face, agegender) in enumerate(
                zip(out_frame["FaceDetection"], out_frame["AgeGender"])
            ):
                if face is not None and agegender is not None:
                    print("prob", agegender.shape)

                    # print(id_emo, max_prob)
                    # print(f"{i} {emotions[id_emo]} {max_prob:.1%}")
                    h_size = face.shape[0]
                    w_size = face.shape[1]
                    w_display = int(w_size * h_display / h_size)
                    face_cv2_frame = cv2.resize(cv2.UMat(face), (w_display, h_display))
                    # 画面に推論結果を表示
                    # agegender : np.array shape=(3,)
                    age = agegender[0]
                    gender = "female" if agegender[1] > agegender[2] else "male"
                    cv2.putText(
                        face_cv2_frame,
                        f"{i} {int(age)}years old, {gender}",
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
