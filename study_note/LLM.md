# 개요
## LLM 정의
기존 언어모델의 확장판 -> 방대한 파라미터 수를 가진 언어모델  
Foundation Models의 시대라고도 볼 수 있음.  
창발성: 단일 모델로 여러 Task를 처리함. (가장 중요한 특징)   
Human Alignment를 얼마나 잘해낼수 있을까가 중요함.  

Scaling Law: 데이터의 크기(D), 계산량(C), 모델크기(N)이 주어졌을때 성능은 3가ㅣ 요소 각각과 power-law를 가진다.
- N과 D가 같이 커져야 성능이 향상된다.
- 우리는 데이터-모델의 역량을 충분히 활용하고 있는가?

Instruction tuning은 언어 모델의 성능을 향상시킨다.
## LLM 재료
1. Infra
- 운영환경(하드웨어)
2. Backbone Model
3. Tuning
- 경량화
- 행렬 연산 최적화 등
4. Data (고품질 & 다량의 학습 데이터 필요)
## LLM의 방향성
1. Multimodal
language, vision, sign language, speech 등 여러 data를 학습시킨 Foundation Model  
PaLM-E: Google Research가 보유한 PaLM을 Robot과 멀티모달 학습에 적용  
GPT-4: Open AI가 ChatGPT 릴리즈 후, 반년도 안되어 GPT-4 공개  
Gmeini : Google Deepmind의 새로운 Multimodal model

2. Synthetic Data
- 합성 데이터의 비율이 훨씬 높아질 것임.  
3. Domain specialized
4. Evaluation
5. prompt Engineering
- Automatic Curriculum

## In-Context Learning
### Fine tuning과의 차이점
Fine tuning은 대규모 코퍼스로 사전학습 후, 적은 규모의 specific한 데이터셋에 대해 fine tuning 하는 과정이다.  
따라서 일반화된 task가 아닌, 일부 task에 대해서 능력을 집중적으로 향상시킨다.  
Model의 Gradient를 업데이트 한다. 하지만 In- Context Leartning은 Gradient가 업데이트가 되지 않는다.  

## Data-Centric NLP 연구
### LLM 학습 데이터 종류

#### Pre-train data
LLM의 초기 학습에 사용되는 데이터.  
대량의 텍스트 데이터를 포함하며, 문맥을 학습하고 일반적인 언어 패턴을 익히는 데 사용됨.  
일반적으로 웹 문서, 책, 논문, 코드, 뉴스 기사 등 다양한 출처에서 수집됨.  
이 단계에서 모델은 방대한 범위의 언어적 구조와 지식을 학습하여 다양한 도메인에서 활용될 수 있는 기반을 형성함.  

##### 1. Task 특화 사전 학습
사전 학습된 모델이 특정 목적이나 도메인에 맞춰 별도로 학습된 경우.  
일반적인 언어 모델보다 특정 태스크에서 우수한 성능을 보임.  

- **LaMDA**:  
  - 대화 모델에 특화된 사전 학습 데이터 사용.  
  - 보다 자연스러운 대화 흐름과 문맥 이해 능력을 강화.  

- **BLOOM, PaLM**:  
  - 다국어와 다양한 태스크를 지원하는 범용 LLM.  
  - 방대한 데이터를 학습하여 다양한 언어적 표현을 이해하는 데 강점.  

- **Galactica**:  
  - 과학 및 연구 논문 데이터를 학습하여 학술적인 질문에 최적화된 모델.  
  - 과학 분야에서 정확한 정보 제공이 가능하도록 설계됨.  

- **AlphaCode**:  
  - 프로그래밍 문제 해결과 코드 생성에 특화된 사전 학습 모델.  
  - 알고리즘 문제 풀이에 최적화되어 있으며, 실제 프로그래밍 대회 문제를 해결할 수 있음.  

#### Fine-tuning data (Alignment)
Pre-trained 모델을 특정 작업이나 목적에 맞도록 미세 조정(Fine-tuning)하는 데이터.  
이 단계를 통해 모델이 특정 스타일, 도메인, 사용자 요구에 더 적합한 출력을 생성할 수 있도록 조정됨.  
Pre-train 단계에서 학습된 일반적인 지식을 특정 목적에 맞춰 정제하는 과정.  

##### 1. Alignment Tuning
- LLM이 인간의 기대와 선호도에 맞는 출력을 생성하도록 조정하는 과정.  
- 모델이 유용하고 안전하며 윤리적인 방식으로 작동하도록 만들기 위해 적용됨.  
- Alignment Tuning에는 **SFT(Supervised Fine-Tuning)**와 **RLHF(Reinforcement Learning from Human Feedback)**가 포함됨.  

  - **SFT (지도 학습 기반 미세 조정)**  
    - 사람이 직접 작성한 데이터셋을 사용하여 모델을 학습.  
    - 특정 태스크(예: 질의응답, 요약, 번역 등)에 맞춰 정제된 데이터로 학습.  
    - 일반적으로 라벨링된 데이터를 사용하여 명확한 목표를 학습.  
    - SFT는 특정 도메인이나 서비스에서 최적의 응답을 생성하도록 모델을 맞추는 데 유용함.  
      - 예: 의료 챗봇을 위한 의료 데이터 기반 SFT.  

  - **RLHF (인간 피드백 기반 강화 학습)**  
    - 모델이 특정 대상의 선호도를 반영하도록 학습.  
    - 보상 모델을 학습시켜 바람직한 출력을 구별할 수 있도록 유도.  
    - 사람이 여러 개의 모델 출력을 비교하여 순위를 매기면, 이를 통해 강화 학습을 진행.  
    - RLHF는 사용자의 피드백을 반영하여 보다 자연스럽고 유용한 출력을 생성하는 데 도움을 줌.  
      - 예: ChatGPT가 유해한 답변을 줄이고 유익한 답변을 하도록 개선.  

##### 2. Instruction data
- **Instruction tuning**:  
  - 언어 모델이 자연어 형태의 지시사항을 이해할 수 있도록 하는 미세 조정 방법론.  
  - 비교적 적은 수의 예제만으로도 높은 성능을 발휘하며, 새로운 태스크에 대한 일반화가 가능.  
  - 사용자가 명확한 지시를 내릴 때 원하는 방식으로 응답할 수 있도록 학습됨.  
  - 예: "이 텍스트를 요약해줘"와 같은 명령어를 이해하고 수행하는 모델.  

- **Instruction dataset**:  
  - 지시어와 그에 대응하는 출력으로 구성된 Instruction 형식의 데이터.  
  - 다양한 명령어에 대해 일관된 반응을 보일 수 있도록 학습됨.  
  - 대화형 AI 및 AI 도우미 모델에서 중요한 역할을 함.  

### LLM 데이터 전처리

#### 기존 LLM의 데이터 전처리 (GPT-3)
1. 데이터 필터링 (Data Filtering)
- 주어진 데이터셋에서 유해하거나 품질이 낮은 데이터를 제거하는 과정.  
- 일반적으로 **similarity 기반 filtering** 기법을 사용하여, 특정 기준(예: 문장 유사도, 토픽 일관성 등)에 맞지 않는 데이터를 제외함.  
- 예시:
  - NSFW(비속어 포함) 콘텐츠 제거  
  - 중복된 내용 삭제  
  - 지나치게 짧거나 의미 없는 문장 필터링 

2. 중복 제거 (Deduplication)
- 동일하거나 거의 유사한 문서가 모델 학습에 반복적으로 사용되는 것을 방지하는 작업.  
- GPT-3에서는 **Fuzzy deduplication** 기법을 사용하여 단순한 문자열 비교가 아니라 의미적으로 유사한 문서까지 제거함.  
- **Document level**(문서 단위)에서 수행되어, 문장 단위가 아니라 전체 문서를 기준으로 중복 여부를 판단함.  

3. 다양성 확보 (Diversify)
- 학습 데이터의 다양성을 증가시키기 위해 **기존의 고품질 말뭉치(high-quality corpora)**를 추가함.  
- 다양한 도메인(뉴스, 과학 논문, 웹 문서 등)에서 균형 잡힌 데이터를 확보하여 모델의 **일반화 성능**을 높이는 역할을 함.  
- 편향(bias)을 줄이고, 특정한 주제나 문체에만 편중되지 않도록 조정함.  

---

이와 같은 전처리 과정은 모델이 더 깨끗하고 다양한 데이터를 학습할 수 있도록 도와, 결과적으로 LLM의 성능을 향상시키는 데 중요한 역할을 한다.

## LLM 기반 Model-Centric NLP
### LLM fine tuning
### Parameter Efficient Tuning
LLM 전체가 아닌 일부분만을 튜닝하는 방법론
1. Adapter-based Tuning
2. LoRA
3. QLoRA
- 4-bit NormalFloat(NF4)
- Double Quantization
- Paged Optimizer
4. LLaMA-Adapter
### Domain Specialization
#### Knowledge Augmentation
1. RAG(Retrieval-Augmented Geveration)
2. Verify-and-Edit: A Knowledge Enhanced Chain of Thought Framework

## LLM Evaluation-Centric NLP
### A System Study and Comprehensive Evaluation of ChatGPT on Benchmark Datasets
QA, 요약, Code generation, 상식추론, 수학적 문제 해결, 기계 번역, 등 같은 작업을 다루는 다양한 NLP 데이터셋에 대한 ChatGPT의 성능 평가 및 분석
### Interpretability
1. FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets
2. Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts
- counter-memory가 잘못된 정보이므로, LLM이 이러한 오류에 속아 잘못된 정보 제공을 할 수 있는 문제 존재.
3. Benchmarking Foundation Models with Language-Model-as-an-Examiner
4. ALCUNA: Large Language Models Meet New Knowledge
5. HaluEval: A Large-Scale Hallucination Evaludation Benchmark for Large Language Models

## Multimodal LLMs
다양한 Modality(text, Image, audio, ...)를 통합하여 언어 모델의 기능을 확장
1. CLIP
2. BLIP
3. LLaVA
4. GPT-4V
### Flaminog
모델구조
- 사전 학습된 Image Encoder, Language Model
- Perceiver Resampler: 서로 다른 차원을 가지는 임베딩 간의 Cross Attention