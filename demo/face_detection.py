import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nobita import nobita_vision
from nobita.modules import FaceDetection
import cv2

pipeline = nobita_vision.VisionPipeline(modules=[FaceDetection()], use_oak=True)
pipeline.set_pipline_to_modules()
pipeline.set_device_to_modules()
while True:
    out_frame = pipeline.get()  # Dict[List[np.array]]
    # print(out_frame)
    for i, detected in enumerate(out_frame["FaceDetection"]):
        if len(detected) > 0:
            cv2.imshow("BBox of Face Detection {i}", detected)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        break
pipeline.device.close()
