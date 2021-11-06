# 딥러닝 테스트

### 목차

**1. Experiment design**

​		1. 1 실험 목표

​		1.2 데이터 분석(Train Data)

​		1.3 사용 모형

​		1. 4. 실험 전 예상결과

**2. Evaluation metrics**

​		2.1 Matrix

​		2.2 Graph

**3. Experimental results**

​		3.1 결과정리

​		3.2 특이사항





# 1. Experiment design

### 1.1 실험 목표

​	Source sequence를 Target sequence와 matching 시키는것.







### 1.2 데이터 분석(Train Data)

|        | Min_index | Max_index | Min_sen_len | Max_sen_len | Mean_sen_len | Vocab_size |       Duplicate_row       | Corpus_sen_count |
| :----: | :-------: | :-------: | :---------: | :---------: | :----------: | :--------: | :-----------------------: | :--------------: |
| Source |    21     |    619    |      2      |     81      |  18.985399   |     53     | 1192(전체의 문장의 16.4%) |       7260       |
| Target |     0     |    658    |      1      |     54      |  10.051928   |    595     | 1187(전체의 문장의 16.3%) |       7260       |



**데이터 분석 결과**

- target 데이터의 vocab size가 source 데이터의 vocab size보다 11.22배 큼

- source 데이터는 (적은 단어수, 긴 문장길이)로 한문장을 표현하고 target 데이터는 (많은 단어수, 짧은 문장길이)로 한 문장을 표현함

- source, target vocab에서 동시에 가지고 있는 data가 하나 존재함







### 1.3 사용 모형

**Transformer 기반 모델 설계**

- 선정 이유

  Sequence-to-Sequence과제를 수행하기 위한 모델로서 Machine Translation분야에서 높은 성능을 보이고 있습니다.

  실험 목표인 sequence ordering 문제와 word matching 문제해결에 적합할 것으로 생각됩니다.

  

- 모델 configuation

  - batch_size = 128
  - n_layers = 4, 5, 6
  - d_model = 256, 512
  - ffn_hidden = 512, 1024, 2048
  - n_head = 8
  - drop_out = 0.1
  - eps = 1e-9
  - epoch = 300

  

- 특이사항

  - inference시, Greedy Decoder 사용
  - 기존 int형 Data를 String형으로 치환하고 새로운 index 할당
  - Train dataset을 8:2로 분리하여 Validation dataset 생성
  - Source data와 Target data의 겹치는 int형 data가 하나뿐입니다. 같은 inter data라도 문맥상 source와 target에서 다른 의미를 가질 것의로 추측됩니다.
    - source, target data 각각 단어 embedding matrixa 생성
  - 약 7000개의 데이터를 기반으로한 선행 연구를 찾지 못했습니다. 하이퍼파라미터(n_layers, d_model ,ffn_hidden)를 변경하며, 적은 데이터로도 mapping 가능한 모델을 실험할것입니다.





### 1.4 실험 전 예상결과

 Transformer기반 모델들은 대용량 Corpus를 기반으로 학습됩니다. 하지만 주어진 실험가능한 데이터는 약 7000쌍입니다.

따라서 기존의 transformer논문에서 제시된 BLEU값인 25.8보다 낮은 결과가 예상됩니다.







# 2. Evaluation metrics

### 2.1 Matrix







### 2.2 Graph









# 3. Experimental results

### 3.1 결과정리





### 3.2 특이사항

- Training Loss보다 Validation Loss가 더 낮게 관찰됨
  - 

