import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nobita import VisionPipeline
from nobita.modules import FaceDetection, EmotionEstimation, HeadPose, PoseEstimation
import cv2
import time
import numpy as np
import pytest

emotions = ["neutral", "happy", "sad", "surprise", "anger"]


def test_run_pipline():
    pipeline = VisionPipeline(modules=[FaceDetection()], use_oak=True)
    pipeline.set_pipline_to_modules()
    pipeline.set_device_to_modules()
    start_time = time.time()
    while True:
        out_frame = pipeline.get()
        # print("out_frame: ", out_frame)
        if out_frame["FaceDetection"]:
            cv2.imshow("BBox of Face Detection", out_frame["FaceDetection"][0])
        # time.sleep(0.1)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
        now_time = time.time()
        if now_time - start_time > 3:
            cv2.destroyAllWindows()
            break
    pipeline.device.close()


def test_make_pipeline_with():
    with VisionPipeline(modules=[FaceDetection()], use_oak=True) as pipeline:
        start_time = time.time()
        while True:
            out_frame = pipeline.get()
            # print(out_frame)
            if out_frame["FaceDetection"]:
                cv2.imshow("BBox of Face Detection", out_frame["FaceDetection"][0])
            # time.sleep(0.1)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
            now_time = time.time()
            if now_time - start_time > 3:
                cv2.destroyAllWindows()
                break


def test_depth():
    with VisionPipeline(modules=[FaceDetection()], use_oak=True) as pipeline:
        start_time = time.time()
        while True:
            out_frame = pipeline.get()
            # print(out_frame)
            if out_frame["depth"]:
                cv2.imshow("depth", out_frame["depth"][0])
            # time.sleep(0.1)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
            now_time = time.time()
            if now_time - start_time > 3:
                cv2.destroyAllWindows()
                break


def test_make_emotion_estimation():
    with VisionPipeline(modules=[EmotionEstimation()], use_oak=True) as pipeline:
        start_time = time.time()
        while True:
            out_frame = pipeline.get()
            # print(out_frame)
            if out_frame["FaceDetection"]:
                cv2.imshow("BBox of Face Detection", out_frame["FaceDetection"][0])
            if out_frame["EmotionEstimation"]:
                key_emo = np.argmax(out_frame["EmotionEstimation"][0])
                print(
                    f'{key_emo}, {emotions[key_emo]}, {out_frame["EmotionEstimation"][0][key_emo]:.1%}',
                    end=" ",
                )
            # time.sleep(0.1)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
            now_time = time.time()
            print(now_time - start_time, "sec")
            if now_time - start_time > 3:
                cv2.destroyAllWindows()
                break


def test_headpose():
    with VisionPipeline(modules=[HeadPose()], use_oak=True) as pipeline:
        start_time = time.time()
        while True:
            out_frame = pipeline.get()

            if out_frame["FaceDetection"]:
                cv2.imshow("BBox of Face Detection", out_frame["FaceDetection"][0])
            if out_frame["HeadPose"]:
                # print(out_frame['HeadPose'])

                pose_ = out_frame["HeadPose"][0]
                print(f"HeadPose: roll={pose_[0]},  pitch={pose_[1]}")

            # time.sleep(0.1)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
            now_time = time.time()
            # print(now_time - start_time, "sec")
            if now_time - start_time > 3:
                cv2.destroyAllWindows()
                break


def test_headpose_pose_emotion():
    with VisionPipeline(
        modules=[HeadPose(), EmotionEstimation(), PoseEstimation()], use_oak=True
    ) as pipeline:
        start_time = time.time()
        while True:
            out_frame = pipeline.get()

            if out_frame["FaceDetection"]:
                cv2.imshow("BBox of Face Detection", out_frame["FaceDetection"][0])
            if out_frame["HeadPose"]:
                # print(out_frame['HeadPose'])

                pose_ = out_frame["HeadPose"][0]
                print(f"HeadPose: roll={pose_[0]},  pitch={pose_[1]}")

            # time.sleep(0.1)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
            now_time = time.time()
            # print(now_time - start_time, "sec")
            if now_time - start_time > 3:
                cv2.destroyAllWindows()
                break
