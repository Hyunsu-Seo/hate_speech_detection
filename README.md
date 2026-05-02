# 한국어 혐오 발언 탐지

> 2022년 기준 국립국어원 인공지능(AI) 말평  
> **혐오 발언 탐지 상시 평가 2위 달성 프로젝트**

본 프로젝트는 한국어 문장의 혐오 발언 여부를 분류하는 NLP 프로젝트입니다.  
국립국어원 인공지능(AI) 말평의 **혐오 발언 탐지** 과제를 기반으로, 한국어 온라인 텍스트에 포함된 혐오·비난 표현을 탐지하는 이진 분류 모델을 개발했습니다.

단순 키워드 매칭 방식이 아니라, 문맥에 따라 의미가 달라지는 한국어 표현의 특성을 반영하기 위해 다양한 한국어 사전학습 언어모델과 대규모 언어모델을 fine-tuning하고 성능을 비교했습니다.  
최종적으로 K-Fold Cross Validation과 모델 앙상블을 적용하여 예측 안정성과 성능을 개선했습니다.

---

## 1. Achievement

- 국립국어원 인공지능(AI) 말평 혐오 발언 탐지 상시 평가 **2위 달성**
- 한국어 PLM 및 LLM 기반 혐오 발언 탐지 모델 비교 실험
- K-Fold Cross Validation 기반 모델 학습 및 검증
- 여러 모델의 예측 결과를 결합한 앙상블 적용
- 신조어, 비표준 표현, 온라인 텍스트 노이즈를 고려한 전처리 수행

---

## 2. Task Description

본 과제는 입력 문장이 혐오 발언에 해당하는지 여부를 분류하는 이진 분류 문제입니다.

| Input | Output |
|---|---|
| 한국어 문장 | 혐오 발언 여부 |

분류 라벨은 다음과 같습니다.

| Label | Description |
|---|---|
| 0 | 혐오 발언이 아닌 문장 |
| 1 | 혐오 발언 문장 |

혐오 표현 탐지 태스크는 단어 자체의 존재 여부뿐만 아니라 문장 전체의 맥락을 함께 고려해야 합니다.  
예를 들어 특정 단어가 포함되어 있어도 실제 문맥에서는 혐오 표현이 아닐 수 있고, 반대로 명시적인 비속어가 없어도 문맥상 특정 집단에 대한 비난이나 혐오를 포함할 수 있습니다.

따라서 본 프로젝트에서는 한국어 문맥 이해 능력을 갖춘 다양한 사전학습 언어모델을 활용하여 문장 단위 분류 모델을 구축했습니다.

---

## 3. Main Approach

본 프로젝트는 다음과 같은 흐름으로 진행했습니다.

1. 데이터 전처리
2. 한국어 PLM 기반 baseline 모델 학습
3. 한국어 LLM 기반 fine-tuning 실험
4. K-Fold Cross Validation 적용
5. 모델별 성능 비교
6. 앙상블 기반 최종 예측 결과 생성

---

## 4. Data Preprocessing

한국어 온라인 텍스트는 신조어, 비표준 표기, 반복 문자, 특수문자, 이모지 등 다양한 노이즈를 포함하고 있습니다.  
이를 줄이기 위해 다음과 같은 전처리를 수행했습니다.

- URL 제거
- HTML 태그 및 불필요한 기호 제거
- 특수문자 정규화
- 반복 문자 정규화
- 이모지 제거
- 공백 정리
- 중복 표현 제거
- tokenizer의 `[UNK]` 발생 단어 확인
- 신조어 및 비표준 표현 사전 기반 보완

특히 혐오 발언 탐지에서는 온라인 커뮤니티나 SNS에서 사용되는 변형 표현, 축약어, 신조어가 모델 성능에 영향을 줄 수 있습니다.  
따라서 tokenizer가 제대로 처리하지 못하는 단어를 확인하고, 도메인 특화 단어를 전처리 과정에 반영했습니다.

---

## 5. Models

다양한 한국어 사전학습 언어모델과 대규모 언어모델을 실험했습니다.

| Category | Model |
|---|---|
| Korean BERT-based Models | KcBERT, KR-BERT |
| Korean ELECTRA-based Models | KoELECTRA, KcELECTRA |
| Korean LLMs | Polyglot-Ko 1.3B, Polyglot-Ko 12.8B |
| Instruction-tuned LLMs | KoAlpaca-Polyglot-12.8B, Synatra-7B |
| LLaMA-based Models | LLaMA2-Ko-7B |

모델별로 동일한 분류 태스크를 수행하도록 fine-tuning한 뒤 validation score를 비교했습니다.  
대규모 모델의 경우 GPU 메모리 사용량을 고려하여 LoRA 기반 fine-tuning을 적용했습니다.

---

## 6. Training Strategy

### 6.1 Single Model Fine-tuning

먼저 각 모델을 개별적으로 fine-tuning하여 baseline 성능을 확인했습니다.

주요 학습 설정은 다음과 같습니다.

| Parameter | Value |
|---|---|
| Task | Binary Classification |
| Loss Function | CrossEntropyLoss |
| Evaluation Metric | F1 Score |
| Max Sequence Length | 60 ~ 100 |
| Optimizer | AdamW |
| Framework | PyTorch, Hugging Face Transformers |

각 모델의 tokenizer 특성과 문장 길이를 고려하여 max length와 batch size를 조정했습니다.

---

### 6.2 K-Fold Cross Validation

데이터 분할에 따른 성능 편차를 줄이기 위해 K-Fold Cross Validation을 적용했습니다.

K-Fold 학습 과정은 다음과 같습니다.

1. 학습 데이터를 K개의 fold로 분할
2. 각 fold마다 train/validation set 구성
3. fold별 모델 학습
4. fold별 validation score 확인
5. 각 fold 모델의 test 예측 결과를 평균 또는 투표 방식으로 결합

이를 통해 단일 train/validation split에 과도하게 의존하지 않고, 보다 안정적인 예측 결과를 얻을 수 있었습니다.

---

### 6.3 Ensemble

최종 성능 향상을 위해 여러 모델의 예측 결과를 결합했습니다.

앙상블에 활용한 주요 모델은 다음과 같습니다.

- LLaMA2-Ko-7B
- Polyglot-Ko-12.8B
- KcELECTRA
- KR-BERT
- KoAlpaca-Polyglot-12.8B

모델별 예측 결과를 비교한 뒤, 성능이 우수한 모델 조합을 중심으로 최종 결과를 생성했습니다.  
서로 다른 구조의 모델을 결합함으로써 특정 모델이 놓치는 문장에 대한 예측 안정성을 높이고자 했습니다.

---

## 7. Experiment Results

실험 로그 기준 주요 모델의 validation 성능은 다음과 같습니다.

| Model | Validation F1 |
|---|---:|
| KR-BERT | 0.9354 |
| LLaMA2-Ko-7B K-Fold | 0.9389 |
| KcELECTRA K-Fold | 0.9476 |
| Polyglot-Ko-12.8B | 0.9505 |
| KoAlpaca-Polyglot-12.8B K-Fold | 0.9619 |

최종 제출에서는 단일 모델이 아닌 여러 모델의 결과를 결합한 앙상블 전략을 사용했습니다.  
이를 통해 모델별 예측 편차를 줄이고, 최종 리더보드 기준 2위를 달성했습니다.

---

## 8. Project Structure

```bash
.
├── text_preprocessing.ipynb
├── preprocessing_2_ipynb의_사본.ipynb
├── kcbert.ipynb
├── koelectra.ipynb
├── krbert-base.ipynb
├── polyglot-ko-1.3b.ipynb
├── polyglot-ko-12.8b.ipynb
├── polyglot-ko-12.8b_수정.ipynb
├── llama2-ko-7b.ipynb
├── synatra_7b.ipynb
├── KoAlpaca-Polyglot-12.8B.ipynb
├── kfold_kcelectra.ipynb
├── kfold_krbert.ipynb
├── kfold_polyglot-ko-1.3b.ipynb
├── kfold_polyglot-ko-12.8b.ipynb
├── kfold_llama_2_ko_7b.ipynb
├── kfold_llama2ko7b+polyglotko12_8b+kcelectra.ipynb
├── _kfold_lama2ko7b+polyglotko12_8b+kcelectra+krbert.ipynb
└── 신조어.txt
```

---

## 9. Tech Stack

| Category | Stack |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| NLP Framework | Hugging Face Transformers |
| Fine-tuning | PEFT, LoRA |
| Data Processing | pandas, NumPy |
| Evaluation | scikit-learn |
| Experiment Environment | Google Colab |

---

## 10. My Contribution

본 프로젝트에서 담당한 역할은 다음과 같습니다.

- 한국어 텍스트 전처리 파이프라인 구현
- 신조어 및 비표준 표현 기반 전처리 보완
- KcBERT, KR-BERT, KoELECTRA, KcELECTRA 모델 fine-tuning
- Polyglot-Ko, LLaMA2-Ko, KoAlpaca, Synatra 등 LLM 기반 분류 실험
- LoRA 기반 대규모 모델 fine-tuning 적용
- K-Fold Cross Validation 학습 코드 구현
- 모델별 validation score 비교 및 분석
- 여러 모델의 예측 결과를 활용한 앙상블 구성
- 최종 제출 파일 생성 및 리더보드 성능 개선
