# 2018/8/6

+ Dataprovider
    - tf_writer : tfrecord 파일로 변한합니다.
    - tf_padder : 지정한 크기보다 작으면 padding 을 추가합니다
    - random_crop_shuffled_batch : random 으로 crop 하는 shuffle Queue 을 생성합니다. 이미지는 특정 이미지보다 크기가 커야합니다.

+ preprocessing
    - get_undersize : 특정 크기보다 작은 이미지를 걸러내 paths 로 반홥합니다.



+ Process
    >> Fg , Bg 나눈기 >> 특정 크기보다 작은 이미지 추출 하기 >>
