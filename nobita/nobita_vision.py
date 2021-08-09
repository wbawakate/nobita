from sys import modules
from typing import List, Set, Tuple, Dict
import depthai
import numpy as np
import cv2
from collections import deque

from .modules import *
from .modules import NobitaModuleType


class VisionPipeline:
    def __init__(
        self,
        modules: List[NobitaModuleType],
        use_oak: bool = True,
        arg_vid_cap=0,
        preview_size: Tuple[int, int] = (300, 300),
    ) -> None:
        """pipeline of nobita

        Args:
            modules (List): list of nobita.modules
            use_oak (bool): [description]
            arg_vid_cap (str or int)
        """
        # self.module_list = modules
        self.use_oak = use_oak
        self.arg_vid_cap = arg_vid_cap
        self.preview_size = preview_size
        # 推論モジュールの名前をキーにする辞書
        self.flow_graph: Dict[str, Set] = dict()
        # Dict of modules
        self.modules = {}
        self.lr_check = False
        self.extended_disparity = False
        self.subpixel = False
        for i in range(len(modules)):
            self.modules[modules[i].name] = modules[i]

        self.prepare_flow_graph()
        print("flow: ", self.flow_graph)
        # set modules
        self.pipeline = depthai.Pipeline()
        if self.use_oak:
            # ColorCamera
            print("Creating Color Camera...")
            # create camera
            self.cam = self.pipeline.createColorCamera()
            self.mono_cam_left = self.pipeline.createMonoCamera()
            self.mono_cam_right = self.pipeline.createMonoCamera()
            self.depth = self.pipeline.createStereoDepth()
            self.cam.setPreviewSize(*self.preview_size)
            self.cam.setResolution(
                depthai.ColorCameraProperties.SensorResolution.THE_1080_P
            )
            self.cam.setInterleaved(False)

            # Properties
            self.cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
            self.mono_cam_left.setResolution(
                depthai.MonoCameraProperties.SensorResolution.THE_400_P
            )
            self.mono_cam_left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
            self.mono_cam_right.setResolution(
                depthai.MonoCameraProperties.SensorResolution.THE_400_P
            )
            self.mono_cam_right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

            # set a stereo camera
            self.depth.setConfidenceThreshold(200)
            # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
            median = depthai.StereoDepthProperties.MedianFilter.KERNEL_7x7
            self.depth.setMedianFilter(median)
            self.depth.setLeftRightCheck(self.lr_check)
            self.depth.setExtendedDisparity(self.extended_disparity)
            self.depth.setSubpixel(self.subpixel)

            self.cam_xout = self.pipeline.createXLinkOut()
            self.depth_xout = self.pipeline.createXLinkOut()
            self.depth_xout.setStreamName("disparity")
            self.cam_xout.setStreamName("CAM")
            # link to xout
            self.cam.preview.link(self.cam_xout.input)
            self.mono_cam_left.out.link(self.depth.left)
            self.mono_cam_right.out.link(self.depth.right)
            self.depth.disparity.link(self.depth_xout.input)
        else:
            print(f"video capture from {self.arg_vid_cap}")
            self.cap = cv2.VideoCapture(self.arg_vid_cap)
            for k in self.modules:
                self.modules[k].from_oak = False

    def set_pipline_to_modules(self):
        for k in self.modules:
            # upstream module
            if self.modules[k].source_name == "CAM":
                self.modules[k].from_oak = self.use_oak
                self.modules[k].set_pipeline(self.pipeline)
                # oakを使う場合はcamをupstream moduleにリンク
                if self.modules[k].from_oak:
                    self.cam.preview.link(self.modules[k].image_manip.inputImage)
            else:
                # downstream
                self.modules[k].set_pipeline(self.pipeline)

    def set_device_to_modules(self):
        self.device = depthai.Device()
        self.device.startPipeline(self.pipeline)
        if self.use_oak:
            self.queue_cam = self.device.getOutputQueue("CAM")
            self.queue_depth = self.device.getOutputQueue(
                name="disparity", maxSize=4, blocking=False
            )
        for k in self.modules:
            self.modules[k].set_device(self.device)

    def get_frame(self):
        if self.use_oak:
            return True, np.array(self.queue_cam.get().getData()).reshape(
                (3, self.preview_size[0], self.preview_size[1])
            ).transpose(1, 2, 0).astype(np.uint8)
        else:
            return self.cap.read()

    def get_frame_depth(self):
        if self.use_oak:
            indepth = self.queue_depth.get()
            frame = indepth.getFrame()
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            return True, frame
        else:
            return False, None

    def close(self):
        self.device.close()
        if hasattr(self, "cap"):
            self.cap.release()

    def __enter__(self):
        self.set_pipline_to_modules()
        self.set_device_to_modules()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # queueとして推論結果を取得
    def get(self):
        outputs = {}
        # upstream modules
        read_correctly, frame = self.get_frame()

        # フレームを読み込めなかった場合
        if not read_correctly:
            outputs["CAM"] = []
            for mod_name in self.modules.keys():
                outputs[mod_name] = []
            return outputs
        else:  # フレームを読み込めた場合
            outputs["CAM"] = [frame]
            queue_modules = deque(self.flow_graph["CAM"])
            read_depth, depth_frame = self.get_frame_depth()
            outputs["depth"] = [depth_frame] if read_depth else []
            while queue_modules:
                module_name = queue_modules.popleft()
                up_name = self.modules[module_name].source_name
                out = self.modules[module_name].pop(outputs[up_name])
                outputs[module_name] = out
                for down_name in self.flow_graph[module_name]:
                    queue_modules.append(down_name)
        return outputs

    def prepare_flow_graph(self):
        # まず辞書に入力されたモジュールを登録
        for k in self.modules.keys():
            self.flow_graph[k] = set()
        # deque
        que = deque(self.flow_graph.keys())
        while len(que):
            mod_name = que.pop()
            source_name = self.modules[mod_name].source_name
            if source_name in self.flow_graph:
                self.flow_graph[source_name].add(mod_name)
            else:
                self.flow_graph[source_name] = set([mod_name])
                if source_name != "CAM":
                    que.append(source_name)
                    # add module
                    self.modules[source_name] = eval(
                        f"{source_name}(from_oak={self.use_oak})"
                    )
        assert "CAM" in self.flow_graph, 'No "CAM" key in flow_graph'
