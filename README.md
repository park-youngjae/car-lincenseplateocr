# 자동차 번호판 인식

## 실행 방법 (Detect)

* 차량 인식 도커 컨테이너  실행 및 접속
```
sudo docker start car-ocr
sudo docker attach car-ocr 
```

* 이미지 업로드 (  ./input  )

* detect.py 실행
```
python detect.py
```

###### detect.py 파라미터 세부정보
```
--Transformation {Transformation 모듈 선택}
--FeatureExtraction {특징 추출 모듈 선택}
--SequenceModeling {시퀀스 모델링 모듈 선택}
--Prediction {예측 모듈 선택}
--image_folder {입력 이미지 폴더 경로}
--saved_model {학습 완료된 모델 정보 파일 경로}
```

* 결과확인  ( ./output  )
이미지 파일 이름.txt 확인




## 실행 방법 (Test)

* 차량 인식 도커 컨테이너  실행 및 접속
```
sudo docker start car-ocr
sudo docker attach car-ocr 
```

* test.py 실행
```
python test.py
```

###### Test.py 파라미터 세부정보
```
--Transformation {Transformation 모듈 선택}
--FeatureExtraction {특징 추출 모듈 선택}
--SequenceModeling {시퀀스 모델링 모듈 선택}
--Prediction {예측 모듈 선택}
--image_folder {입력 이미지 폴더 경로}
--saved_model {학습 완료된 모델 정보 파일 경로}
```

* 결과확인
Test Set 성능 확인



## dependency
```
Python >= 3.7
Pytorch >= 1.3.1
natsort >= 7.1.0
nltk >= 3.5
pillow >= 7.0.0
lmdb >= 1.0.0
```



## Reference
https://github.com/clovaai/deep-text-recognition-benchmark
