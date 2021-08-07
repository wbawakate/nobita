# nobota
OAK-Dを使ったロボット開発ための非言語コミュニケーションライブラリ

## 準備
このライブラリには[OAK-D](https://store.opencv.ai/)と[depthai](https://github.com/luxonis/depthai)が動くPython環境が必要です。
### インストール
```
pip install https://github.com/wbawakate/nobita
```

## Quik Start
nobitaはdepthaiの面倒な処理をラップしているので、あなたは少ない行数で顔検出と感情推定のような2段階の推論を行うことができます。   
nobitaには大きく分けて2つの要素があります。`nobita.modules`と`nobita.VisionPipeline`です。`nobita.modules`は、顔検出や感情推論のような非言語コミュニケーションに関する典型的なタスクを扱うニューラルネットワークのモジュール群です。詳しくは、[モジュール](##モジュール)をご覧ください。nobita.VisionPipeline`は`nobita.modules`をパイプラインとしてOAK-Dにデプロイしたり、連続的に推論を行ったりします。  
それでは、nobitaを使って顔画像から感情推定を始めましょう。以下が顔検出と感情推定のコードです。
```
import cv2
import numpy as np
from nobita import VisionPipeline
from nobita.modules import FaceDetection, EmotionEstimation

emotions = ["neutral", "happy", "sad", "surprise", "anger"]
# prepare VisionPipeline of nobita
#    pass `nobita.modules` to modules as a list
#    if you use OAK-D, set `use_oak=True`
with VisionPipeline(modules=[FaceDetection(), EmotionEstimation()], use_oak=True) as pipeline:
    while True:
        # get result of estimation and camera preview as dict
        #     key of the dict is name of `nobita.modules`.
        #     value of the dict is list of numpy.array, which is prediction values of estimation of the `nobita.module`.
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
```
`nobita.VisionPipeline`に、`FaceDetection`や`EmotionEstimation`などの`nobita.modules`を渡すだけで簡単に2段階の推論を行うことができます。推論結果は、`pipeline.get()`で辞書として取得することができます。デモでは、`FaceDetection`で検出した顔の画像に`EmotionEstimation`で推論した感情のテキストを貼り付けてディスプレイに表示させています。

## デモ
`demo/`に各モジュールのデモコードあります。
`demo/`のディレクトリで以下のようにコードを実行してください。
```
python face_emotion.py
```
なお、各デモには`--device`というオプションがあります。このオプションではビデオをキャプチャするデバイスを指定することができます。もし、あなたがOAK-Dを使う場合は-1 (default)を指定してください。それ以外のwebカメラを使うときは、OpenCVでそのデバイスを指定する場合と同じデバイスIDを指定してください。


## モジュール
| module | discription | source | blob file | 
|-------|-------------|--------|----|
|FaceDetection | face detection |[OpenVINO Toolkit](https://docs.openvinotoolkit.org/2020.1/_models_intel_face_detection_retail_0004_description_face_detection_retail_0004.html)  |face-detection-retail-0004_openvino_2021.2_6shave.blob |
|PoseEstimation | human pose estimation| [depthai_movenet](https://github.com/geaxgx/depthai_movenet)|movenet_singlepose_lightning_U8_transpose.blob|
|EmotionEstimation | emotion estimation by facial imases |[OpenVINO Toolkit](https://docs.openvinotoolkit.org/2019_R1/_emotions_recognition_retail_0003_description_emotions_recognition_retail_0003.html)| emotions-recognition-retail-0003.blob|
|AgeGender | age and gender estimation facial imases|[OpenVINO Toolkit](https://docs.openvinotoolkit.org/2019_R1/_age_gender_recognition_retail_0013_description_age_gender_recognition_retail_0013.html) | age-gender-recognition-retail-0013_openvino_2021.2_6shave.blob|
|FaceLandmark | facial landmark detection by facial images |[OpenVINO Toolkit](https://docs.openvinotoolkit.org/2019_R1/_facial_landmarks_35_adas_0002_description_facial_landmarks_35_adas_0002.html) | facial-landmarks-35-adas-0002-shaves6.blob|
|HeadPose | head pose estimation by facial images | [OpenVINE](https://docs.openvinotoolkit.org/2019_R1/_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)| head-pose-estimation-adas-0001-shaves4.blob |



# Credit
- WBA Future Leaders ( https://wbawakate.jp )