# convolutional_pose_machines
About Convolutional_pose_machines, one of posh estimation

## 01. CPM model 요약

receptive field(지역적 정보)를 global한 영역으로 확대하여 다른 부위와의 관계를 고려한 모델 (6 stage)

## 02. CPM model의 전체적인 흐름

신체 부위에 대한 belief map 을 반복하여 생성 (b_1 ~ b_T) 하여 다음 input 값으로 넣어주고 개선된 탐지를 가능하게 한다

Gradient vanishing을 해결하기 위해 각 stage마다 loss를 계산한다. (intermediate supervsion)

각 stage에서 multi-class Classifier를 통해 각 part에 해당하는 belief map을 추정, 추정된 belief map 정보는 fine tuning을 위해 다음 stage로 전달된다

CNN 하위 layer에는 local한 영역을 해석하고, 상위로 갈수록 receptive field가 커지면서 global하게 해석한다.(receptive field    상관관계)

해석한 정보는 각 stage를 구성하는 CNN의 feature map에 저장

## 03. Belief Map

stage1에서 detect가 쉬운 parts들이 다음 stage에 어려운 part들을 예측하는데에 강한 정보가 된다.

Shoulder, Neck, Head의 spatial context가 R.Elbow의 belief map에 옳은 예측을 하는데에 기여를 하고, 틀린 예측을 없애주는 역할을 한다.

※ belief map : img 내 각 pixel이 part 위치일 확률 / input img와 동일한 크기의 map
g함수를 통해 총 P+1개의 output이 나온다.
