from typing import List, Tuple, Optional, Union


import depthai
import numpy as np

from .utils import frame_norm, to_planar


class Base:
    def __init__(
        self,
        blob_path: Optional[str] = None,
        input_size: Tuple[int, int] = (300, 300),
        threshold: float = 0.7,
        from_oak: bool = True,
    ):
        self.name = str(self.__class__.__name__)
        self.default_blob_path = ""
        self.blob_path = (
            blob_path if isinstance(blob_path, str) else self.default_blob_path
        )
        self.input_size = input_size
        self.source_name: Optional[str] = None
        self.threshold = threshold
        self.from_oak = from_oak

    def set_source_name(self, name: str):
        self.source_name = name

    def set_pipeline(self, pipeline: depthai.Pipeline):
        self.pipeline = pipeline
        # create neural network
        self.nn = self.pipeline.createNeuralNetwork()
        self.nn.setBlobPath(self.blob_path)
        # set XLink Out from NN
        self.nn_xout = self.pipeline.createXLinkOut()
        self.nn_xout.setStreamName(self.name)  # 名前を設定
        self.nn.out.link(self.nn_xout.input)

        #  入力ノードはcam か xinput
        if not self.from_oak:
            self.nn_xin = self.pipeline.createXLinkIn()
            self.nn_xin.setStreamName(f"{self.name}_input")
            self.nn_xin.out.link(self.nn.input)
        else:
            # make resize image
            self.image_manip = self.pipeline.createImageManip()
            self.image_manip.initialConfig.setResize(*self.input_size)
            self.image_manip.initialConfig.setFrameType(depthai.ImgFrame.Type.BGR888p)
            self.image_manip.out.link(self.nn.input)

    def set_device(self, device: depthai.Device):
        self.device = device
        if not self.from_oak:
            self.queue_nn_xin = self.device.getInputQueue(f"{self.name}_input")
        self.queue_nn = self.device.getOutputQueue(self.name)

    def set_source(self, name: str):
        self.source_name = name

    def set_name(self, name: str):
        self.name = name

    # ここを個別に書き換える
    def _get_ndarray(self, frame: np.ndarray):
        output = np.array(self.queue_nn.get().getFirstLayerFp16())
        return output

    # ここも書き換える
    def _preprocess(self, frame:np.ndarray, input_size: Tuple[int, int]):
        return to_planar(frame, input_size)

    def pop(self, frames):
        """NNノードの出力queueから推論結果をpopする
        Args:
            frames (List[numpy.ndarray]): フレームのNumpy配列を格納したリスト

        Returns:
            List[numpy.ndarray] or List[None]: フレームのNumpy配列を格納したリスト。結果がなければNoneが格納されたリストが返ってくる
        """
        if self.from_oak:
            assert len(frames) == 1, "Length of frames must be 1, if you use OAK-D."
            if self.queue_nn.has():
                out = self._get_ndarray(frames[0])
            else:
                out = []
            return out
        else:
            outputlist = []
            for frame in frames:
                nn_data = depthai.NNData()
                nn_data.setLayer("input", self._preprocess(frame, self.input_size))
                self.queue_nn_xin.send(nn_data)
                if self.queue_nn.has():
                    out = self._get_ndarray(frame)
                    if isinstance(out, list):
                        outputlist = out
                    else:
                        outputlist.append(out)
                else:
                    pass
            return outputlist
