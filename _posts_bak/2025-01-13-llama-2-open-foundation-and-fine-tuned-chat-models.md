---
layout: post
title: "Llama 2: Open Foundation and Fine-Tuned Chat Models"
date: 2023-07-18 14:31:57
author: "GenAI, Meta"
categories: "Foundation-Models"
tags: ["Open-and-Efficient-Foundation-Language-Models", "Reinforcement-Learning-with-Human-Feedback", "Rotary-Positional-Embeddings", "Safety-Alignment", "Scaling-Laws-for-Large-Language-Models", "SwiGLU-Activation-Function"]
use_math: true
---
### TL;DR
#### 이 연구를 시작하게 된 배경과 동기는 무엇입니까?

대규모 언어 모델(LLM)이 인공지능 분야에서 중요한 발전을 이루고 있지만, 대부분의 고성능 모델들이 비공개로 유지되면서 연구 커뮤니티의 발전이 제한되는 문제가 있었습니다. 특히 ChatGPT, BARD, Claude와 같은 상용 대화형 AI 시스템들은 광범위한 미세조정 과정을 거쳐 뛰어난 성능과 안전성을 보여주었지만, 이러한 개선 과정이 투명하게 공개되지 않아 AI 정렬(alignment) 연구의 발전을 저해했습니다. Meta 연구진은 이러한 격차를 해소하고 AI 기술의 민주화를 촉진하기 위해 Llama 2 프로젝트를 시작했습니다.

#### 이 연구에서 제시하는 새로운 해결 방법은 무엇입니까?

연구진은 세 가지 핵심적인 혁신을 제시했습니다. 첫째, 사전학습 단계에서 2조 개의 토큰을 사용하고 컨텍스트 길이를 4,000 토큰으로 확장하여 모델의 기본 성능을 크게 향상시켰습니다. 둘째, 지도 학습 기반 미세조정(SFT)과 인간 피드백 기반 강화학습(RLHF)을 결합한 포괄적인 미세조정 파이프라인을 개발했습니다. 셋째, 고스트 어텐션(Ghost Attention)이라는 새로운 기법을 도입하여 다중 턴 대화에서의 일관성을 크게 개선했습니다.

#### 제안된 방법은 어떻게 구현되었습니까?

구현은 크게 세 단계로 진행되었습니다. 먼저 사전학습 단계에서는 엄격한 데이터 정제 과정을 거친 공개 데이터를 사용했으며, 7B부터 70B까지 다양한 크기의 모델을 학습했습니다. 미세조정 단계에서는 27,540개의 고품질 주석 데이터로 SFT를 수행한 후, 140만 개 이상의 인간 선호도 비교 데이터를 활용한 RLHF를 적용했습니다. 마지막으로 350명 이상의 전문가로 구성된 레드팀이 모델의 안전성을 철저히 평가하고 개선했습니다.

#### 이 연구의 결과가 가지는 의미는 무엇입니까?

이 연구는 고성능 대화형 AI 시스템의 개발 과정을 투명하게 공개함으로써 AI 연구의 민주화에 크게 기여했습니다. Llama 2-Chat은 대부분의 벤치마크에서 기존 오픈소스 모델들을 능가했으며, 특히 도움이 되는 정도와 안전성 측면에서 상용 모델들과 대등한 성능을 보여주었습니다. 또한 RLHF 과정에서 발견된 컨텍스트 기반 온도 조절이나 도구 사용의 자발적 출현과 같은 현상들은 언어 모델의 능력에 대한 새로운 통찰을 제공했습니다. 이러한 성과는 향후 AI 시스템의 책임있는 개발과 평가를 위한 중요한 기준점이 될 것으로 기대됩니다.
- - -
## Llama 2: 오픈 파운데이션 및 미세조정된 대화 모델

Meta의 연구진이 공개한 Llama 2는 70억에서 700억 개의 매개변수를 가진 대규모 언어 모델(Large Language Model, LLM) 시리즈입니다. 이 연구는 특히 대화형 애플리케이션에 최적화된 Llama 2-Chat 모델의 개발과 공개에 초점을 맞추고 있습니다.

Llama 2의 개발은 현대 인공지능 연구에서 중요한 이정표를 제시합니다. 기존의 오픈소스 대화 모델들과 비교했을 때, Llama 2-Chat은 대부분의 벤치마크에서 우수한 성능을 보여주었으며, 특히 도움이 되는 응답 생성과 안전성 측면에서 상용 비공개 모델들을 대체할 수 있는 수준에 도달했습니다.

연구진은 Llama 2-Chat의 개발 과정에서 두 가지 핵심적인 미세조정 단계를 거쳤습니다. 첫째, 지도 학습 기반 미세조정(Supervised Fine-Tuning, SFT)을 통해 모델이 인간이 작성한 대화 데이터로부터 자연스러운 대화 패턴을 학습하도록 했습니다. 둘째, 인간 피드백 기반 강화학습(Reinforcement Learning with Human Feedback, RLHF)을 적용하여 모델의 응답이 더욱 도움이 되고 안전하도록 최적화했습니다.

특히 이 논문에서는 Llama 2-Chat의 안전성 향상을 위한 포괄적인 접근 방식을 상세히 설명합니다. 사전 학습 단계에서의 안전성 고려사항부터 시작하여, 미세조정 과정에서의 안전성 강화, 그리고 레드팀(Red Team) 평가를 통한 잠재적 위험 식별까지 다루고 있습니다. 이러한 다층적인 안전성 접근법은 모델이 실제 응용 환경에서 신뢰성 있게 작동할 수 있도록 보장하는 데 중요한 역할을 합니다.

연구진은 이러한 개발 과정과 방법론을 상세히 공개함으로써, 연구 커뮤니티가 이를 기반으로 대규모 언어 모델의 책임있는 개발을 이어나갈 수 있도록 했습니다. 이는 인공지능 기술의 발전과 안전한 활용을 위한 중요한 기여로 평가됩니다.

### 서론

대규모 언어 모델(Large Language Models, LLMs)은 프로그래밍과 창작 등 전문적인 지식이 필요한 복잡한 추론 작업에서 뛰어난 성능을 보이는 AI 어시스턴트로 주목받고 있습니다. 이러한 모델들은 직관적인 채팅 인터페이스를 통해 사용자와 상호작용할 수 있어 일반 대중들 사이에서도 빠르게 확산되고 있습니다.

LLM의 학습 방법론은 겉보기에 단순해 보입니다. 자기지도 학습(self-supervised) 데이터를 사용하여 자동회귀(auto-regressive) 트랜스포머 모델을 사전학습한 후, 인간 선호도 강화학습(Reinforcement Learning with Human Feedback, RLHF)과 같은 기법을 통해 인간의 선호도에 맞게 조정하는 과정을 거칩니다. 하지만 이러한 단순한 방법론에도 불구하고, 막대한 컴퓨팅 자원이 필요하다는 점 때문에 LLM 개발은 소수의 기업들에 의해서만 이루어져 왔습니다.

최근에는 BLOOM, LLaMa-1, Falcon과 같은 사전학습된 LLM들이 공개되어 GPT-3나 Chinchilla와 같은 비공개 모델들과 비슷한 성능을 보여주고 있습니다. 하지만 이러한 공개 모델들은 ChatGPT, BARD, Claude와 같은 상용화된 "제품" LLM을 대체하기에는 부족한 면이 있습니다. 상용 LLM들은 인간의 선호도에 맞게 광범위한 미세조정 과정을 거쳤으며, 이는 모델의 사용성과 안전성을 크게 향상시켰습니다. 이러한 미세조정 과정에는 상당한 컴퓨팅 자원과 인간 주석자의 노력이 필요하며, 그 과정이 투명하게 공개되지 않아 AI 정렬(alignment) 연구의 발전을 제한하는 요인이 되고 있습니다.

![Helpfulness human evaluation results](https://ar5iv.org//html/2307.09288/assets/x1.png)

위 그래프는 다양한 언어 모델들의 도움이 되는 정도(helpfulness)를 인간 평가자들이 비교한 결과를 보여줍니다. 단일 및 다중 턴 프롬프트에 대해 약 4,000개의 응답을 평가했으며, 95% 신뢰구간은 1-2% 수준입니다. Llama-2-70b-chat 모델이 다른 공개 및 비공개 모델들보다 우수한 성능을 보여주고 있음을 확인할 수 있습니다.

![Win-rate percentages](https://ar5iv.org//html/2307.09288/assets/x2.png)

GPT-4 모델을 활용한 보완적 평가에서도 Llama 2-Chat은 상용 모델들과 비교했을 때 도움이 되는 정도와 안전성 측면에서 더 나은 성능을 보여주었습니다. 평가의 공정성을 위해 모델 응답의 순서를 무작위로 바꾸어가며 진행했으며, 동점을 제거하기 위해 $$ win/(win+loss) $$ 비율을 사용했습니다.
![Safety human evaluation results](https://ar5iv.org//html/2307.09288/assets/img/safety_overall_human_temp.png)

안전성 평가에서도 Llama 2-Chat은 주목할 만한 성과를 보여주었습니다. 약 2,000개의 적대적 프롬프트(adversarial prompts)에 대해 단일 및 다중 턴 대화를 평가한 결과, 다른 공개 및 비공개 모델들과 비교했을 때 더 낮은 안전성 위반율을 기록했습니다. 다만 연구진은 이러한 안전성 평가 결과를 해석할 때 프롬프트 세트의 한계, 평가 가이드라인의 주관성, 그리고 개별 평가자들의 주관적 판단이 영향을 미칠 수 있다는 점을 고려해야 한다고 강조했습니다.

![Training of Llama 2-Chat](https://ar5iv.org//html/2307.09288/assets/x3.jpg)

이러한 성과를 바탕으로 연구진은 최대 700억 개의 매개변수를 가진 사전학습 및 미세조정된 LLM 시리즈인 Llama 2와 Llama 2-Chat을 공개했습니다. Llama 2-Chat의 학습 과정은 크게 세 단계로 구성됩니다. 먼저 공개 데이터를 사용하여 Llama 2 모델을 사전학습하고, 이를 지도 학습으로 미세조정하여 초기 버전의 Llama 2-Chat을 만듭니다. 마지막으로 거부 샘플링(rejection sampling)과 근위 정책 최적화(Proximal Policy Optimization, PPO)와 같은 RLHF 기법을 통해 모델을 반복적으로 개선합니다.

연구진은 7B, 13B, 70B 매개변수를 가진 모델 변형들을 공개했으며, 34B 변형도 개발했지만 충분한 안전성 검증 시간이 필요하여 추후 공개할 예정이라고 밝혔습니다. 모든 LLM과 마찬가지로 Llama 2도 잠재적 위험을 가지고 있으므로, 연구진은 개발자들이 각자의 응용 분야에 맞는 안전성 테스트와 조정을 수행할 것을 권장하고 있습니다. 이를 위해 책임있는 사용 가이드와 코드 예제들을 함께 제공하고 있습니다.

이어지는 논문의 구성은 사전학습 방법론(2장), 미세조정 방법론(3장), 모델 안전성 접근법(4장), 주요 관찰과 통찰(5장), 관련 연구(6장), 그리고 결론(7장)으로 이루어져 있습니다.

### Llama 2 모델의 사전학습 방법론

Llama 2 모델은 기존 Llama 1의 사전학습 방법론을 기반으로 하되, 여러 가지 중요한 개선사항을 도입했습니다. 이러한 개선사항들은 모델의 성능을 전반적으로 향상시키는 데 기여했습니다.

가장 주목할 만한 변화는 학습 데이터의 규모와 품질 면에서 이루어졌습니다. 연구진은 공개적으로 접근 가능한 데이터 소스에서 2조 개의 토큰을 수집했으며, 이는 Llama 1 모델보다 40% 더 많은 양입니다. 특히 개인정보가 많이 포함된 것으로 알려진 웹사이트들의 데이터는 제외하는 등 더욱 엄격한 데이터 정제 과정을 거쳤습니다. 또한 사실에 기반한 정보를 담고 있는 데이터 소스의 비중을 높임으로써 모델의 환각(hallucination) 현상을 줄이고자 했습니다.

아키텍처 측면에서도 중요한 개선이 이루어졌습니다. 컨텍스트 길이를 2,000 토큰에서 4,000 토큰으로 두 배 늘렸으며, 34B와 70B 파라미터 모델에는 그룹 쿼리 어텐션(Grouped-Query Attention, GQA)을 도입했습니다. GQA는 대규모 모델의 추론 확장성을 개선하는 데 도움을 주는 기술입니다.

학습 과정에서는 AdamW 옵티마이저를 사용했으며, 하이퍼파라미터는 다음과 같이 설정되었습니다.

$$ \beta_1 = 0.9, \beta_2 = 0.95, \text{eps} = 10^{-5} $$

코사인 학습률 스케줄을 적용했으며, 2,000 스텝의 웜업 기간을 거친 후 최종 학습률을 피크 값의 10%까지 감소시켰습니다. 가중치 감쇠(weight decay)는 0.1, 그래디언트 클리핑은 1.0으로 설정했습니다.

![Training Loss for Llama 2 models](https://ar5iv.org//html/2307.09288/assets/x4.png)

위 그래프는 Llama 2 모델 패밀리의 학습 손실 곡선을 보여줍니다. 주목할 만한 점은 2조 개의 토큰으로 사전학습을 완료한 후에도 모델이 포화 상태에 도달하지 않았다는 것입니다. 이는 더 많은 데이터로 학습을 진행할 경우 성능이 더욱 향상될 수 있음을 시사합니다.

토크나이저로는 Llama 1과 동일하게 SentencePiece 구현체를 사용한 바이트페어 인코딩(BPE) 알고리즘을 채택했습니다. 모든 숫자는 개별 자릿수로 분할하고, 알 수 없는 UTF-8 문자는 바이트 단위로 분해하는 방식을 사용했습니다. 전체 어휘 크기는 32,000 토큰입니다.
### 학습 인프라 및 탄소 발자국

Llama 2 모델의 사전학습은 Meta의 연구용 슈퍼 클러스터(Research Super Cluster, RSC)와 내부 프로덕션 클러스터에서 진행되었습니다. 두 클러스터 모두 NVIDIA A100 GPU를 사용했지만, 몇 가지 중요한 차이점이 있었습니다.

첫 번째 주요 차이점은 인터커넥트 기술입니다. RSC는 NVIDIA Quantum InfiniBand를 사용한 반면, 프로덕션 클러스터는 일반 이더넷 스위치를 기반으로 한 RoCE(RDMA over Converged Ethernet) 솔루션을 채택했습니다. 두 솔루션 모두 200Gbps의 종단 간 대역폭을 제공합니다. 두 번째 차이점은 GPU당 전력 소비 제한으로, RSC는 400W, 프로덕션 클러스터는 350W로 설정되었습니다.

이러한 이중 클러스터 구성을 통해 연구진은 대규모 학습에서 서로 다른 인터커넥트 기술의 적합성을 비교할 수 있었습니다. 특히 주목할 만한 점은 상대적으로 저렴한 상용 인터커넥트 네트워크인 RoCE가 2,000개의 GPU까지 확장할 때 고가의 InfiniBand에 근접한 성능을 보여주었다는 것입니다. 이는 대규모 언어 모델의 학습을 더욱 민주화할 수 있는 가능성을 시사합니다.

사전학습에 따른 탄소 배출량도 면밀히 분석되었습니다. Llama 2 모델 패밀리의 학습에는 총 331만 GPU 시간이 소요되었으며, 이는 539 tCO₂eq의 탄소 배출량으로 추정됩니다. 이 계산에는 GPU의 열설계 전력(TDP)을 기준으로 한 전력 사용량이 고려되었습니다. 다만, 이 추정치에는 인터커넥트나 비GPU 서버 전력 소비, 데이터센터 냉각 시스템 등의 추가적인 전력 수요는 포함되지 않았습니다.

모델 크기별 탄소 배출량을 살펴보면, 7B 모델이 31.22 tCO₂eq, 13B 모델이 62.44 tCO₂eq, 34B 모델이 153.90 tCO₂eq, 그리고 70B 모델이 291.42 tCO₂eq를 배출했습니다. Meta는 이러한 탄소 배출량을 자사의 지속가능성 프로그램을 통해 100% 상쇄했습니다.

특히 주목할 만한 점은 Llama 2를 오픈소스로 공개함으로써 다른 기업들이 동일한 모델을 다시 학습할 필요가 없게 되었다는 것입니다. 이는 전 세계적인 컴퓨팅 자원과 에너지 절약에 기여할 수 있습니다. 이러한 접근 방식은 AI 개발의 환경적 영향을 최소화하면서도 기술의 발전과 접근성을 촉진하는 균형 잡힌 전략을 보여줍니다.

### 학습 모델 평가 결과

Llama 2 모델의 성능을 평가하기 위해 연구진은 다양한 학문적 벤치마크에서 기존의 오픈소스 기반 모델들과 비교 실험을 진행했습니다. 주요 비교 대상으로는 MosaicML의 MPT(MosaicML Pretrained Transformer) 모델과 Falcon 모델이 포함되었습니다.

평가는 크게 6가지 주요 카테고리로 나누어 진행되었습니다.

1. 코드 생성 능력: HumanEval과 MBPP 벤치마크에서의 pass@1 점수를 평가했습니다. 이는 모델이 생성한 코드가 첫 시도에서 정상적으로 실행되는 비율을 의미합니다.

2. 상식적 추론: PIQA, SIQA, HellaSwag, WinoGrande 등 8개의 벤치마크에서 모델의 상식 추론 능력을 평가했습니다. 대부분의 테스트는 제로샷(zero-shot) 방식으로 진행되었으며, CommonSenseQA만 7-shot으로 평가되었습니다.

3. 세계 지식: NaturalQuestions와 TriviaQA 벤치마크에서 5-shot 성능을 평가했습니다. 이는 모델이 일반적인 사실 기반 질문에 얼마나 정확하게 답변할 수 있는지를 측정합니다.

4. 독해 능력: SQuAD, QuAC, BoolQ 벤치마크에서 제로샷 성능을 평가했습니다. 이는 모델이 주어진 텍스트를 얼마나 잘 이해하고 관련 질문에 답변할 수 있는지를 측정합니다.

5. 수학적 능력: GSM8K(8-shot)와 MATH(4-shot) 벤치마크에서 모델의 수학적 문제 해결 능력을 평가했습니다.

6. 종합적 벤치마크: MMLU(5-shot), BBH(3-shot), AGI Eval(3-5 shot) 등 포괄적인 벤치마크에서 모델의 전반적인 성능을 평가했습니다.

실험 결과, Llama 2 모델은 이전 버전인 Llama 1과 비교해 전반적으로 향상된 성능을 보여주었습니다. 특히 70B 파라미터 모델의 경우, MMLU와 BBH 벤치마크에서 각각 약 5점과 8점의 성능 향상을 달성했습니다. 또한 Llama 2의 7B와 34B 모델은 동일한 크기의 MPT와 Falcon 모델들을 대부분의 카테고리에서 앞섰습니다.

비공개 모델들과의 비교에서도 Llama 2 70B는 주목할 만한 성과를 보였습니다. GPT-3.5와 비교했을 때 MMLU와 GSM8K 벤치마크에서 근접한 성능을 보였으며, PaLM(540B)과 비교해서는 대부분의 벤치마크에서 동등하거나 더 나은 성능을 달성했습니다. 다만 GPT-4와 PaLM-2-L과는 아직 상당한 성능 차이가 있는 것으로 나타났습니다.
### Llama 2의 미세조정 방법론

Llama 2-Chat은 수개월에 걸친 연구와 반복적인 정렬(alignment) 기법의 적용을 통해 개발되었습니다. 이 과정에서 지도 학습 기반 미세조정(Supervised Fine-Tuning, SFT)과 인간 피드백 기반 강화학습(Reinforcement Learning with Human Feedback, RLHF)이 핵심적인 역할을 했습니다.

연구진은 먼저 지도 학습 기반 미세조정을 위해 공개된 지시어 튜닝 데이터를 활용했습니다. 하지만 기존의 데이터셋들이 대화형 지시어에 대한 다양성과 품질이 부족하다는 점을 발견했고, 이를 보완하기 위해 고품질의 SFT 데이터를 직접 수집하기로 결정했습니다.

특히 주목할 만한 점은 연구진이 "양보다 질"이라는 원칙을 채택했다는 것입니다. 수백만 개의 서드파티 데이터셋을 제외하고, 대신 자체적으로 수집한 수만 개의 고품질 데이터를 사용했을 때 더 나은 결과를 얻을 수 있었습니다. 이는 Zhou 등의 연구에서도 확인된 바와 같이, 제한된 양의 깨끗한 지시어 튜닝 데이터만으로도 높은 수준의 성능에 도달할 수 있다는 것을 보여줍니다.

최종적으로 연구진은 27,540개의 고품질 주석 데이터를 수집했으며, 이 과정에서 Meta 사용자 데이터는 전혀 포함하지 않았습니다. 데이터의 품질을 검증하기 위해 180개의 샘플을 면밀히 검토한 결과, SFT 모델이 생성한 출력이 인간 주석자가 작성한 데이터와 비교했을 때 경쟁력 있는 수준이라는 것을 발견했습니다.

미세조정의 기술적 세부사항으로는, 코사인 학습률 스케줄을 사용했으며 초기 학습률은 $2 \times 10^{-5}$, 가중치 감쇠는 0.1, 배치 크기는 64, 시퀀스 길이는 4,096 토큰으로 설정했습니다. 각 샘플은 프롬프트와 응답으로 구성되며, 특수 토큰을 사용해 이들을 구분했습니다. 자기회귀(autoregressive) 목적 함수를 사용했으며, 사용자 프롬프트에 대한 손실은 0으로 설정하여 응답 토큰에 대해서만 역전파가 이루어지도록 했습니다. 전체 미세조정은 2 에포크 동안 진행되었습니다.

![SFT annotation examples](https://ar5iv.org//html/2307.09288/assets/x5.png)

위 그림은 SFT를 위한 주석 예시를 보여줍니다. 도움이 되는 응답(상단)과 안전성(하단)에 대한 주석이 포함되어 있으며, 주석자가 프롬프트와 그에 대한 응답을 모두 작성했습니다. 이러한 고품질 데이터를 통해 모델이 자연스럽고 안전한 대화 방식을 학습할 수 있었습니다.
### 인간 피드백 기반 강화학습(RLHF)

인간 피드백 기반 강화학습(RLHF)은 미세조정된 언어 모델의 행동을 인간의 선호도와 지시사항 준수에 더욱 가깝게 정렬하기 위해 적용되는 학습 절차입니다. 이 과정에서는 먼저 인간 주석자들이 모델이 생성한 두 가지 응답 중 어느 것을 더 선호하는지에 대한 데이터를 수집합니다. 이렇게 수집된 인간의 선호도 데이터는 보상 모델(reward model)을 학습하는 데 사용되며, 이 보상 모델은 이후 인간 주석자들의 선호도 패턴을 학습하여 선호도 결정을 자동화할 수 있게 됩니다.

연구진은 인간 선호도 데이터 수집을 위해 이진 비교 프로토콜을 채택했습니다. 이 방식을 선택한 주된 이유는 수집된 프롬프트의 다양성을 최대화할 수 있기 때문입니다. 구체적인 주석 절차는 다음과 같습니다.

1. 주석자가 먼저 프롬프트를 작성합니다.
2. 주어진 프롬프트에 대해 서로 다른 모델 변형에서 생성된 두 가지 응답 중 하나를 선택합니다.
3. 선택한 응답을 얼마나 더 선호하는지 정도를 표시합니다.
   - 매우 더 좋음(significantly better)
   - 더 좋음(better)
   - 약간 더 좋음(slightly better)
   - 거의 비슷함/확실하지 않음(negligibly better/unsure)

선호도 주석 수집은 도움이 되는 정도(helpfulness)와 안전성(safety)이라는 두 가지 측면에 초점을 맞추었습니다. 도움이 되는 정도는 Llama 2-Chat의 응답이 사용자의 요청을 얼마나 잘 충족시키고 필요한 정보를 제공하는지를 평가합니다. 안전성은 응답이 위험하지 않은지를 평가하는데, 예를 들어 "폭탄 제조법 상세 설명"과 같은 응답은 도움이 될 수는 있지만 안전성 기준에 따르면 부적절한 것으로 간주됩니다.

안전성 주석 단계에서는 추가적인 레이블링도 수행되었습니다. 이는 모델 응답을 다음 세 가지 범주로 분류합니다.

1. 선호된 응답은 안전하고 다른 응답은 안전하지 않음 (18%)
2. 두 응답 모두 안전함 (47%)
3. 두 응답 모두 안전하지 않음 (35%)

연구진은 선호된 응답이 안전하지 않고 다른 응답이 안전한 경우는 데이터셋에 포함시키지 않았습니다. 이는 더 안전한 응답이 인간에게도 더 선호될 것이라는 가정에 기반합니다.
### 인간 선호도 데이터 수집과 보상 모델 학습

인간 선호도 데이터는 매주 단위로 수집되었으며, 이를 통해 보상 모델의 성능이 점진적으로 향상되었습니다. 더 많은 선호도 데이터가 수집됨에 따라 보상 모델이 개선되었고, 이는 Llama 2-Chat의 성능 향상으로 이어졌습니다. 

특히 주목할 만한 점은 Llama 2-Chat이 발전함에 따라 모델의 데이터 분포도 함께 변화했다는 것입니다. 보상 모델의 정확도는 새로운 샘플 분포에 노출되지 않으면 빠르게 저하될 수 있는데, 이는 과도한 전문화(hyper-specialization) 현상 때문입니다. 따라서 새로운 Llama 2-Chat 튜닝 반복을 시작하기 전에 최신 모델 반복에서 새로운 선호도 데이터를 수집하는 것이 중요했습니다. 이 단계는 보상 모델이 최신 데이터 분포에 맞춰져 있도록 하고, 최신 모델에 대한 정확한 보상을 유지하는 데 도움을 주었습니다.

연구진은 오픈소스 선호도 데이터셋과 자체 수집한 데이터를 결합하여 더 큰 학습 데이터셋을 구성했습니다. 초기에는 자체 선호도 데이터를 수집하는 동안 보상 모델을 부트스트랩하기 위해 오픈소스 데이터셋을 활용했습니다. 주목할 만한 점은 RLHF 맥락에서 보상 신호의 역할이 일반적인 모델 출력이 아닌 Llama 2-Chat 출력에 대한 인간의 선호도를 학습하는 것이라는 점입니다.

보상 모델 학습을 위해 수집된 데이터의 통계를 살펴보면, Meta에서 수집한 데이터는 약 140만 개의 이진 비교를 포함하고 있으며, 대화당 평균 3.9턴, 예제당 평균 798.5 토큰을 포함하고 있습니다. 이는 기존 오픈소스 데이터셋들과 비교했을 때 더 긴 대화와 더 풍부한 컨텍스트를 제공합니다.

보상 모델의 학습은 사전학습된 채팅 모델 체크포인트에서 시작됩니다. 이는 보상 모델이 채팅 모델의 지식을 공유하도록 보장하며, 예를 들어 환각(hallucination)을 선호하는 등의 정보 불일치를 방지합니다. 모델 아키텍처와 하이퍼파라미터는 사전학습된 언어 모델과 동일하지만, 다음 토큰 예측을 위한 분류 헤드가 스칼라 보상을 출력하는 회귀 헤드로 대체됩니다.

학습 목적 함수로는 다음과 같은 이진 랭킹 손실을 사용했습니다.

$$ \mathcal{L}_{\text{ranking}}=-\text{log}(\sigma(r_{\theta}(x,y_{c})-r_{\theta}(x,y_{r}))) $$

여기서 $r_{\theta}(x,y)$는 프롬프트 $x$와 완성 $y$에 대한 모델 가중치 $\theta$를 사용한 스칼라 점수 출력입니다. $y_c$는 주석자가 선택한 선호 응답이고 $y_r$은 거부된 응답입니다.
이러한 기본적인 이진 랭킹 손실을 바탕으로, 연구진은 도움이 되는 정도와 안전성에 대한 보상 모델을 각각 더욱 최적화했습니다. 특히 4단계로 구분된 선호도 평가(매우 더 좋음, 더 좋음, 약간 더 좋음, 거의 비슷함)를 활용하여 보상 모델이 응답들 간의 품질 차이를 더 명확하게 학습하도록 했습니다. 이를 위해 손실 함수에 마진 컴포넌트를 추가했습니다.

$$ \mathcal{L}_{\text{ranking}}=-\text{log}(\sigma(r_{\theta}(x,y_{c})-r_{\theta}(x,y_{r})-m(r))) $$

여기서 마진 $m(r)$은 선호도 평가에 따른 이산 함수입니다. 두 응답의 차이가 큰 경우에는 큰 마진을, 비슷한 경우에는 작은 마진을 적용했습니다. 이러한 마진 컴포넌트의 도입은 특히 두 응답이 더 구분 가능한 샘플들에 대해 도움이 되는 정도를 평가하는 보상 모델의 정확도를 향상시켰습니다.

데이터 구성 측면에서는, Meta에서 수집한 새로운 데이터와 기존의 오픈소스 선호도 데이터셋을 결합하여 더 큰 학습 데이터셋을 구성했습니다. 도움이 되는 정도를 평가하는 보상 모델은 모든 Meta Helpfulness 데이터와 함께, Meta Safety 데이터와 오픈소스 데이터셋에서 균등하게 샘플링한 데이터를 동일한 비율로 혼합하여 학습했습니다. 안전성 보상 모델은 모든 Meta Safety 데이터와 Anthropic Harmless 데이터를 기본으로 하고, Meta Helpfulness 데이터와 오픈소스 도움이 되는 정도 데이터를 90:10 비율로 혼합하여 학습했습니다. 특히 10%의 도움이 되는 정도 데이터를 포함하는 것이 두 응답 모두 안전하다고 판단된 샘플에 대한 정확도 향상에 도움이 되었습니다.

학습 과정에서는 데이터를 한 번만 학습하도록 했는데, 이는 초기 실험에서 더 오래 학습하면 과적합이 발생할 수 있다는 것을 발견했기 때문입니다. 70B 파라미터 Llama 2-Chat 모델의 경우 최대 학습률을 $5 \times 10^{-6}$으로, 나머지 모델들은 $1 \times 10^{-5}$로 설정했습니다. 학습률은 코사인 스케줄에 따라 최대값의 10%까지 감소하도록 했으며, 전체 스텝의 3%(최소 5스텝)를 웜업 기간으로 설정했습니다. 효과적인 배치 크기는 512쌍(1024행)으로 고정했습니다.

![Reward model results](https://ar5iv.org//html/2307.09288/assets/x7.png)

보상 모델의 성능 평가 결과, Meta의 보상 모델들은 SteamSHP-XL, Open Assistant, GPT4를 포함한 다른 기준 모델들을 모든 평가 지표에서 앞섰습니다. 특히 도움이 되는 정도를 평가하는 보상 모델은 Meta Helpfulness 테스트 세트에서, 안전성 보상 모델은 Meta Safety 테스트 세트에서 가장 우수한 성능을 보였습니다. 이는 각 모델이 자신의 도메인에서 특화된 성능을 보여준다는 것을 의미합니다.
### RLHF 반복 학습과 고스트 어텐션

RLHF 학습은 더 많은 인간 선호도 데이터와 개선된 보상 모델을 확보함에 따라 반복적으로 진행되었습니다. 이 과정에서 연구진은 RLHF-V1부터 RLHF-V5까지 여러 버전의 모델을 학습했으며, 주로 두 가지 알고리즘을 탐구했습니다.

첫 번째는 근위 정책 최적화(Proximal Policy Optimization, PPO)로, RLHF 분야에서 표준으로 사용되는 방법입니다. 두 번째는 거부 샘플링(Rejection Sampling) 미세조정으로, 모델에서 K개의 출력을 샘플링한 후 보상 모델을 사용해 최적의 후보를 선택하는 방식입니다. 이는 Bai 등의 연구에서도 사용된 방법이며, Deng 등은 이러한 재순위화 전략에서 보상을 에너지 함수로 보는 관점을 제시했습니다.

두 RL 알고리즘의 주요 차이점은 다음과 같습니다.

너비(Breadth) 측면에서, 거부 샘플링은 주어진 프롬프트에 대해 K개의 샘플을 탐색하는 반면, PPO는 한 번에 하나의 생성만을 수행합니다. 깊이(Depth) 측면에서, PPO는 학습 단계 t에서 t-1 단계의 정책 업데이트 결과를 반영한 모델로 샘플을 생성하는 반면, 거부 샘플링 미세조정은 모델의 초기 정책을 사용해 모든 출력을 샘플링한 후 새로운 데이터셋을 구성하여 SFT와 유사한 방식으로 미세조정을 진행합니다.

![Max and median rewards among N samples](https://ar5iv.org//html/2307.09288/assets/x6.png)

위 그래프는 샘플 수(N)에 따른 최대 및 중간값 보상의 관계를 보여줍니다. 최대값과 중간값 사이의 차이는 거부 샘플링을 통해 얻을 수 있는 잠재적 이득으로 해석할 수 있습니다. 샘플 수가 증가할수록 최대값은 증가하는 반면 중간값은 일정하게 유지되는 것을 확인할 수 있습니다.

연구진은 RLHF V4까지는 거부 샘플링 미세조정만을 사용했으며, 이후에는 두 방법을 순차적으로 결합하여 거부 샘플링 체크포인트에 PPO를 적용한 후 다시 샘플링하는 방식을 채택했습니다. 특히 거부 샘플링은 70B Llama 2-Chat 모델에서만 수행되었으며, 작은 모델들은 큰 모델의 거부 샘플링 데이터로 미세조정되어 큰 모델의 능력을 증류(distillation)받는 방식으로 학습되었습니다.

![RLHF impact of temperature](https://ar5iv.org//html/2307.09288/assets/x7.png)
![Temperature effects](https://ar5iv.org//html/2307.09288/assets/x8.png)

위 그래프들은 N개의 출력을 샘플링하고 보상 모델로 점수를 매길 때 온도(temperature) 파라미터가 미치는 영향을 보여줍니다. RLHF는 온도 스케일링에 직접적인 영향을 미치며, Llama 2-Chat-RLHF의 경우 10-100개의 출력을 샘플링할 때 최적의 온도는 $T \in [1.2, 1.3]$ 범위인 것으로 나타났습니다. 이는 제한된 계산 자원 내에서 온도를 점진적으로 재조정할 필요가 있음을 시사합니다.
### 다중 턴 일관성을 위한 고스트 어텐션

연구진은 다중 턴 대화에서 발생하는 중요한 문제를 해결하기 위해 고스트 어텐션(Ghost Attention, GAtt)이라는 새로운 기법을 제안했습니다. 이는 대화가 여러 턴 진행되는 동안 특정 지시사항이나 제약조건을 일관되게 유지해야 하는 상황에서 특히 유용합니다. 예를 들어, "간단히 답변하라" 또는 "특정 인물처럼 행동하라"와 같은 지시사항은 전체 대화 과정에서 지속적으로 준수되어야 합니다.

초기 RLHF 모델들은 대화가 몇 턴 진행된 후에는 초기 지시사항을 "잊어버리는" 경향이 있었습니다. 이러한 한계를 극복하기 위해 제안된 GAtt는 Context Distillation에서 영감을 받아 미세조정 데이터를 수정하여 어텐션이 다단계 과정에서 더 효과적으로 작동하도록 돕습니다.

![Multi-turn memory issues and improvements](https://ar5iv.org//html/2307.09288/assets/x9.png)
![GAtt improvements](https://ar5iv.org//html/2307.09288/assets/x10.png)

위 그림에서 볼 수 있듯이, GAtt를 적용하기 전(왼쪽)에는 모델이 대화가 진행됨에 따라 초기 지시사항을 잊어버리는 문제가 있었지만, GAtt 적용 후(오른쪽)에는 다중 턴 대화에서도 지시사항을 일관되게 유지할 수 있게 되었습니다.

GAtt의 구체적인 작동 방식은 다음과 같습니다. 두 사람(사용자와 어시스턴트) 간의 다중 턴 대화 데이터셋 $[u_1,a_1,\ldots,u_n,a_n]$이 있다고 가정할 때, 전체 대화에 걸쳐 준수되어야 할 지시사항 $inst$를 정의합니다. 이 지시사항을 대화의 모든 사용자 메시지에 합성적으로 연결한 후, 최신 RLHF 모델을 사용해 이 수정된 데이터에서 샘플링을 수행합니다.

![Attention visualization with GAtt](https://ar5iv.org//html/2307.09288/assets/x11.png)

위 그림은 GAtt 적용 전후의 어텐션 활성화를 시각화한 것입니다. 각 그림의 왼쪽은 시스템 메시지("Act as Oscar Wilde")에 해당합니다. GAtt가 적용된 모델(오른쪽)이 기존 모델(왼쪽)에 비해 대화의 더 많은 부분에서 시스템 메시지에 대한 높은 어텐션 활성화를 유지하는 것을 확인할 수 있습니다.

연구진은 GAtt를 RLHF V3 이후에 적용했으며, 정량적 분석을 통해 GAtt가 최대 컨텍스트 길이에 도달할 때까지 20턴 이상 일관성을 유지할 수 있음을 확인했습니다. 특히 주목할 만한 점은, "항상 하이쿠로 답변하라"와 같이 GAtt 학습에 포함되지 않았던 제약조건에 대해서도 모델이 일관성을 유지할 수 있었다는 것입니다.

현재 GAtt의 구현은 기본적인 수준이지만, 이 기법의 추가 개발과 반복을 통해 모델의 성능을 더욱 향상시킬 수 있을 것으로 기대됩니다. 예를 들어, 대화 중에 시스템 메시지를 변경하는 것을 학습하도록 미세조정 과정에 관련 데이터를 통합하는 등의 발전 가능성이 있습니다.
### RLHF 결과 분석

RLHF 모델의 성능을 평가하기 위해 연구진은 모델 기반 평가와 인간 평가를 모두 수행했습니다. 모델 기반 평가는 반복적인 실험과 모델 개선을 빠르게 진행하기 위해 사용되었으며, 주요 모델 버전에 대해서는 인간 평가를 통해 검증을 진행했습니다.

보상 모델의 신뢰성을 검증하기 위해 연구진은 도움이 되는 정도와 안전성에 대한 테스트 세트를 구성하고, 세 명의 주석자에게 7점 리커트 척도로 응답의 품질을 평가하도록 했습니다. 분석 결과, 보상 모델은 쌍별 랭킹 손실(Pairwise Ranking Loss)로 학습되었음에도 불구하고 인간의 선호도 주석과 전반적으로 잘 일치하는 것으로 나타났습니다.

하지만 굿하트의 법칙(Goodhart's Law)이 지적하듯이, 측정 지표가 목표가 되면 더 이상 좋은 측정 지표가 되지 못할 수 있습니다. 이러한 문제를 방지하기 위해 연구진은 다양한 오픈소스 보상 모델링 데이터셋에서 학습된 더 일반적인 보상 모델도 추가로 사용했습니다. 현재까지는 이러한 발산 현상이 관찰되지 않았으며, 이는 반복적인 모델 업데이트가 이를 방지하는 데 도움이 되었을 것으로 추정됩니다.

모델의 발전 과정을 살펴보면, 안전성과 도움이 되는 정도 모두에서 RLHF-V3 이후 ChatGPT를 능가하는 성능(50% 이상의 승률)을 달성했습니다. 하지만 Meta의 보상 모델이 Llama 2-Chat에 유리하게 편향되었을 수 있다는 점을 고려하여, 연구진은 GPT-4를 사용한 공정한 비교 평가도 수행했습니다. ChatGPT와 Llama 2-Chat의 출력이 GPT-4 프롬프트에 무작위로 배치되어 편향을 방지했으며, 예상대로 Llama 2-Chat의 승률은 다소 감소했지만 여전히 60% 이상의 승률을 유지했습니다. 이 평가는 안전성에 대해 1,586개, 도움이 되는 정도에 대해 584개의 검증 세트 프롬프트를 사용하여 진행되었습니다.

인간 평가에서는 4,000개 이상의 단일 및 다중 턴 프롬프트에 대해 Llama 2-Chat 모델을 오픈소스 모델(Falcon, MPT, Vicuna)과 비공개 모델(ChatGPT, PaLM)과 비교했습니다. 각 프롬프트에 대해 세 명의 평가자가 독립적으로 평가를 수행했으며, 평가자 간 신뢰도(Inter-Rater Reliability, IRR)는 Gwet's AC1/2 통계를 사용하여 측정되었습니다. 7점 리커트 척도를 사용한 도움이 되는 정도 평가에서 Gwet's AC2 점수는 모델 비교에 따라 0.37에서 0.55 사이의 값을 보였습니다. 특히 Llama 2-Chat-70B와 ChatGPT 비교와 같이 승률이 비슷한 모델 간 비교에서는 낮은 점수를, Llama 2-Chat-34B와 Falcon-40B-instruct 비교와 같이 명확한 우위가 있는 경우에는 높은 점수를 보였습니다.
### 안전성 평가 방법론

Llama 2 모델의 안전성 평가는 매우 체계적이고 포괄적인 방식으로 진행되었습니다. 연구진은 모델의 안전성을 확보하기 위해 사전학습부터 미세조정, 그리고 최종 평가에 이르기까지 다층적인 접근 방식을 채택했습니다.

안전성 평가의 핵심은 두 가지 주요 측면에 초점을 맞추었습니다. 첫째는 모델이 유해하거나 위험한 내용을 생성하지 않도록 하는 것이고, 둘째는 모델이 악의적인 프롬프트나 조작 시도에 대해 적절히 대응할 수 있도록 하는 것입니다. 이를 위해 연구진은 다양한 안전성 벤치마크와 평가 방법을 개발하고 적용했습니다.

특히 주목할 만한 점은 안전성 평가를 위한 보상 모델의 설계입니다. 안전성 보상 모델은 도움이 되는 정도를 평가하는 보상 모델과는 별도로 학습되었는데, 이는 두 목표가 때로는 상충될 수 있기 때문입니다. 예를 들어, 위험한 내용에 대한 상세한 설명은 정보 제공 측면에서는 도움이 될 수 있지만 안전성 측면에서는 부적절할 수 있습니다.

안전성 평가를 위한 데이터 수집 과정에서는 특별히 적대적 프롬프트(adversarial prompts)에 중점을 두었습니다. 이는 모델이 악의적인 의도를 가진 사용자의 요청에 대해서도 안전하게 대응할 수 있는지를 확인하기 위함입니다. 연구진은 이러한 적대적 프롬프트를 다음과 같은 카테고리로 분류했습니다.

- 유해한 내용 생성 요청
- 편향되거나 차별적인 내용 유도
- 개인정보 수집 시도
- 불법적인 활동 지원 요청
- 윤리적으로 문제가 있는 조언 요청

이러한 안전성 평가는 단순히 모델의 응답을 분석하는 것을 넘어, 모델이 이러한 요청들을 어떻게 인식하고 거부하는지, 그리고 대안적인 안전한 응답을 제시할 수 있는지까지 포함하여 종합적으로 이루어졌습니다. 이는 모델이 단순히 유해한 내용을 피하는 것을 넘어, 적극적으로 안전하고 건설적인 대화를 이끌어갈 수 있도록 하는 것을 목표로 합니다.
### 인간 평가의 한계점과 RLHF의 실제 적용

Llama 2-Chat이 인간 평가에서 ChatGPT와 대등한 성능을 보여주었지만, 연구진은 인간 평가 방식이 가지는 여러 한계점을 상세히 분석했습니다. 먼저, 학술 연구 기준으로는 4,000개의 프롬프트가 큰 규모이지만, 이는 실제 세계에서 이러한 모델들이 마주하게 될 다양한 사용 사례들을 완전히 포괄하기에는 부족합니다.

프롬프트의 다양성 측면에서도 한계가 있었습니다. 예를 들어, 평가에 사용된 프롬프트 세트에는 코딩이나 복잡한 추론과 관련된 프롬프트가 포함되지 않았습니다. 이는 모델의 특정 능력들을 충분히 평가하지 못했을 수 있다는 것을 의미합니다. 또한 다중 턴 대화의 경우, 최종 생성 결과만을 평가했다는 한계가 있습니다. 더 의미 있는 평가를 위해서는 모델에게 특정 작업을 완료하도록 하고 여러 턴에 걸친 전체적인 대화 경험을 평가하는 것이 필요할 것입니다.

생성 모델에 대한 인간 평가는 본질적으로 주관적이고 노이즈가 있을 수밖에 없습니다. 이는 다른 프롬프트 세트나 다른 평가 지침을 사용할 경우 다른 결과가 나올 수 있다는 것을 의미합니다. 이러한 한계를 고려할 때, 단일 평가 방법이나 지표에 전적으로 의존하기보다는 다양한 평가 방법을 종합적으로 활용하는 것이 중요합니다.

연구진은 이러한 한계점들을 인식하고, RLHF 과정에서 다양한 보완책을 도입했습니다. 예를 들어, 보상 모델의 정확도가 새로운 데이터 분포에서 급격히 저하되는 것을 방지하기 위해 정기적으로 새로운 선호도 데이터를 수집했으며, 이를 통해 모델이 실제 사용 환경에서도 안정적인 성능을 유지할 수 있도록 했습니다. 또한 도움이 되는 정도와 안전성이라는 두 가지 목표 사이의 균형을 맞추기 위해 별도의 보상 모델을 사용하는 등의 세심한 설계가 이루어졌습니다.

이러한 노력들은 Llama 2-Chat이 단순히 벤치마크 성능을 넘어, 실제 응용 환경에서도 신뢰성 있고 안전한 대화형 AI 시스템으로 기능할 수 있도록 하는 데 초점을 맞추고 있습니다. 연구진의 이러한 포괄적이고 신중한 접근 방식은 향후 대화형 AI 시스템의 개발과 평가에 있어 중요한 참고 사례가 될 것으로 기대됩니다.

### Llama 2의 안전성 평가와 개선 방법론

Llama 2 모델의 안전성을 확보하기 위해 Meta 연구진은 포괄적이고 체계적인 접근 방식을 채택했습니다. 이 섹션에서는 사전학습 데이터와 모델의 안전성 조사부터 시작하여, 안전성 정렬(alignment) 과정, 레드팀(red teaming) 평가, 그리고 Llama 2-Chat의 정량적 안전성 평가에 이르기까지의 전체 과정을 상세히 설명합니다.

먼저 사전학습 데이터의 안전성 측면에서, 연구진은 데이터의 언어 분포, 인구통계학적 대표성, 그리고 유해성(toxicity)을 분석했습니다. 특히 개인정보가 많이 포함된 사이트의 데이터는 제외했으며, 학습 과정의 탄소 발자국을 최소화하기 위해 효율적인 학습 방법을 채택했습니다. 

인구통계학적 대표성 분석을 위해 영어 말뭉치에서 가장 흔한 대명사의 빈도를 조사했습니다. 분석 결과, 'He' 대명사가 'She' 대명사보다 더 자주 등장하는 것으로 나타났습니다. 구체적으로, 'She' 대명사는 문서의 28.45%에서만 발견된 반면, 'He' 대명사는 50.73%의 문서에서 발견되었습니다. 이러한 불균형은 모델이 'She' 대명사와 관련된 맥락에 대해 상대적으로 덜 학습할 수 있음을 시사합니다.

또한 HolisticBias 데이터셋을 활용하여 종교, 성별과 성, 국적, 인종과 민족, 성적 지향과 같은 5가지 축에서 인구통계학적 용어의 분포를 분석했습니다. 분석 결과, 서구 중심적인 편향이 발견되었습니다. 예를 들어, "American"이라는 용어는 문서의 69.4%에서 발견되었으며, "European"이 다른 인종/민족 관련 용어보다 더 자주 등장했고, 종교 측면에서는 "Christian"이 가장 높은 빈도를 보였습니다.

데이터의 유해성 평가를 위해 ToxiGen 데이터셋으로 미세조정된 HateBERT 분류기를 사용했습니다. 각 문서의 각 줄을 개별적으로 평가하고 평균을 내어 문서 점수를 산출했습니다. 전체 말뭉치의 약 0.2%만이 0.5 이상의 유해성 점수를 받았는데, 이는 사전학습 데이터에 유해한 내용이 상대적으로 적게 포함되어 있음을 보여줍니다.

![Data toxicity distribution](https://ar5iv.org//html/2307.09288/assets/img/data_toxicity.png)

위 그래프는 대화 시스템의 어텐션 시각화를 보여주며, Gated Attention (GAtt) 메커니즘의 유무에 따른 어텐션 패턴을 비교합니다. x축은 유해성 점수를, y축은 문서의 비율을 나타냅니다. GAtt 모델이 유해한 출력의 비율을 효과적으로 감소시키는 것을 확인할 수 있습니다.
언어 분포 측면에서는, fastText 언어 식별 도구를 사용하여 사전학습 데이터의 언어 구성을 분석했습니다. 분석 결과, 영어가 89.70%로 가장 큰 비중을 차지했으며, 알 수 없는 언어가 8.38%로 그 뒤를 이었습니다. 알 수 없는 언어의 상당 부분은 프로그래밍 코드로 추정됩니다. 나머지는 독일어(0.17%), 프랑스어(0.16%), 스웨덴어(0.15%) 등이 차지했습니다. 이러한 영어 중심의 데이터 구성은 Llama 2가 영어 이외의 언어에서는 제한된 성능을 보일 수 있음을 시사합니다.

사전학습된 모델의 안전성을 평가하기 위해 연구진은 세 가지 주요 차원에서 자동화된 벤치마크를 실시했습니다.

첫째, 진실성(Truthfulness) 평가를 위해 TruthfulQA를 사용했습니다. 이는 모델이 사실과 상식에 부합하는 신뢰할 수 있는 출력을 생성할 수 있는지를 측정합니다. 

둘째, 유해성(Toxicity) 평가를 위해 ToxiGen을 활용했습니다. 이는 모델이 유해하거나, 무례하거나, 적대적이거나, 암묵적으로 혐오적인 내용을 생성하는 경향을 측정합니다.

셋째, 편향성(Bias) 평가를 위해 BOLD를 사용했습니다. 이는 모델이 생성하는 내용에서 기존의 고정관념적인 사회적 편향이 얼마나 재생산되는지를 연구합니다.

평가를 위한 디코딩 설정으로는 온도(temperature) 파라미터를 0.1로, 핵 샘플링(nucleus sampling)의 top-p 값을 0.9로 설정했습니다. TruthfulQA에서는 진실하면서도 유익한 생성의 비율을, ToxiGen에서는 유해하다고 판단되는 생성의 비율을 측정했습니다.

Llama 1-7B와 비교했을 때, Llama 2-7B는 진실성과 유익성이 21.37% 증가했고 유해성은 7.61% 감소했습니다. 그러나 13B와 70B 모델에서는 유해성이 다소 증가했는데, 이는 더 큰 사전학습 데이터나 다른 데이터셋 구성의 영향일 수 있습니다. 일부 연구자들은 사전학습 데이터셋의 크기와 모델의 유해성 또는 편향성 사이에 관계가 있을 수 있다고 제안했지만, 이를 검증하기 위한 실증적 연구는 아직 진행 중입니다.
안전성 미세조정(Safety Fine-Tuning) 과정에서 연구진은 안전성 카테고리, 주석 가이드라인, 그리고 안전성 위험을 완화하기 위한 기술적 접근 방식을 체계적으로 정립했습니다. 이 과정은 일반적인 미세조정 방법을 기반으로 하되, 안전성과 관련된 몇 가지 중요한 차이점을 포함합니다.

구체적으로 다음과 같은 세 가지 기술이 안전성 미세조정에 적용되었습니다.

첫째, 지도 학습 기반 안전성 미세조정(Supervised Safety Fine-Tuning)입니다. 이 단계에서는 적대적 프롬프트와 안전한 시연 데이터를 수집하여 일반적인 지도 학습 미세조정 과정에 포함시킵니다. 이를 통해 모델은 RLHF 단계 이전에 안전성 가이드라인에 맞춰 학습되며, 이는 고품질의 인간 선호도 데이터 주석을 위한 기반을 마련합니다.

둘째, 안전성 RLHF(Safety RLHF)입니다. 이 단계에서는 안전성에 특화된 보상 모델을 학습시키고, 더 도전적인 적대적 프롬프트를 수집하여 거부 샘플링 스타일의 미세조정과 PPO 최적화에 활용합니다. 이는 일반적인 RLHF 파이프라인에 안전성 요소를 통합하는 과정입니다.

셋째, 안전성 컨텍스트 증류(Safety Context Distillation)입니다. 이 기법은 RLHF 파이프라인을 더욱 정교화합니다. 구체적으로, "당신은 안전하고 책임감 있는 어시스턴트입니다"와 같은 안전성 프리프롬프트를 사용하여 더 안전한 모델 응답을 생성한 후, 이 프리프롬프트 없이도 동일한 수준의 안전한 응답을 생성하도록 모델을 미세조정합니다. 특히 안전성 보상 모델이 각 샘플에 대해 컨텍스트 증류 적용 여부를 선택할 수 있도록 하는 타겟팅 접근 방식을 사용합니다.

안전성 카테고리와 주석 가이드라인 측면에서는, 기존 연구에서 알려진 LLM의 한계점들을 바탕으로 주석 팀에게 두 가지 차원의 지침을 제공했습니다. 하나는 위험 카테고리로, LLM이 안전하지 않은 내용을 생성할 수 있는 잠재적 주제를 다룹니다. 다른 하나는 공격 벡터로, 모델의 나쁜 행동을 유도할 수 있는 다양한 프롬프트 스타일을 포함합니다.
연구진이 고려한 위험 카테고리는 크게 세 가지로 분류됩니다. 첫째는 불법 및 범죄 활동으로, 테러리즘, 절도, 인신매매 등이 포함됩니다. 둘째는 혐오 및 유해 활동으로, 명예훼손, 자해, 섭식장애, 차별 등을 다룹니다. 셋째는 비전문가적 조언으로, 의료, 재무, 법률 등 전문적 자격이 필요한 분야의 조언이 해당됩니다.

공격 벡터는 다양한 형태로 구성되었습니다. 심리적 조작(예: 권위 조작), 논리적 조작(예: 거짓 전제), 구문적 조작(예: 오타), 의미적 조작(예: 은유), 관점 조작(예: 역할극), 비영어 언어 등이 포함됩니다. 이러한 다각적인 접근을 통해 모델의 안전성을 종합적으로 평가하고 개선할 수 있었습니다.

안전하고 도움이 되는 모델 응답을 위한 모범 사례도 정립했습니다. 모델은 먼저 즉각적인 안전성 우려사항이 있다면 이를 먼저 다루고, 그 다음 사용자에게 잠재적 위험을 설명하며, 마지막으로 가능한 경우 추가 정보를 제공하도록 설계되었습니다. 또한 부정적인 사용자 경험을 야기할 수 있는 카테고리들을 피하도록 주석자들에게 지시했습니다.

안전성 지도 학습 미세조정 단계에서는, 정립된 가이드라인에 따라 훈련된 주석자들로부터 프롬프트와 안전한 모델 응답의 시연 데이터를 수집했습니다. 주석자들은 먼저 모델이 안전하지 않은 행동을 보일 수 있는 프롬프트를 생성하는 레드팀 작업을 수행한 후, 그에 대한 안전하고 도움이 되는 응답을 작성했습니다.

![Safety RLHF impact](https://ar5iv.org//html/2307.09288/assets/x14.png)

위 그래프는 Llama 2-Chat과 ChatGPT의 승률 비교를 보여줍니다. 왼쪽은 보상 모델이, 오른쪽은 GPT-4가 판정자 역할을 한 결과입니다. "안전성 개선" 영역이 나타나는 것을 볼 수 있으며, 이는 미세조정 과정을 통해 모델의 안전성과 유용성이 향상되었음을 시사합니다.

연구진은 Llama 2-Chat이 지도 학습의 안전한 시연으로부터 일반화하는 능력을 조기에 보여주었다는 점에 주목했습니다. 모델은 안전성 우려사항을 다루고, 주제의 민감성을 설명하며, 추가적인 유용한 정보를 제공하는 상세한 안전 응답을 생성할 수 있게 되었습니다. 특히 안전한 응답을 생성할 때는 평균적인 주석자보다 더 상세한 내용을 제공하는 경향을 보였습니다.
이러한 초기 성과를 바탕으로, 연구진은 수천 개의 지도 학습 시연 데이터를 수집한 후 RLHF로 전환하여 모델이 더 미묘한 응답을 작성하는 방법을 학습하도록 했습니다. RLHF를 통한 포괄적인 튜닝은 모델의 잠재적인 취약점 공격(jailbreak) 시도에 대한 견고성을 향상시키는 부가적인 이점도 제공했습니다.

안전성 RLHF는 일반적인 RLHF 파이프라인과 유사하게 진행되었습니다. 주석자들이 모델의 안전하지 않은 행동을 유도할 수 있다고 생각하는 프롬프트를 작성하고, 해당 프롬프트에 대한 여러 모델 응답을 비교하여 가이드라인에 따라 가장 안전한 응답을 선택했습니다. 이 인간 선호도 데이터는 안전성 보상 모델을 학습하는 데 사용되었으며, 적대적 프롬프트들은 RLHF 단계에서 모델 샘플링에 재활용되었습니다.

![RLHF impact on safety](https://ar5iv.org//html/2307.09288/assets/x15.png)

위 그래프는 안전성 데이터 비율에 따른 평균 안전성 및 도움이 되는 정도 점수의 변화를 보여줍니다. 안전성 데이터의 비율이 증가함에 따라 안전성 점수가 크게 향상되는 반면, 도움이 되는 정도 점수는 비교적 안정적으로 유지되는 것을 확인할 수 있습니다.

연구진은 안전성 RLHF의 영향을 Meta Safety 테스트 세트에서 생성된 출력물에 대한 보상 모델 점수 분포를 통해 분석했습니다. 결과는 두 가지 중요한 발견을 보여줍니다. 첫째, 안전성 RLHF 이후 안전성 보상 모델 점수의 분포가 더 높은 점수 영역으로 이동했습니다. 둘째, 점수 분포의 긴 꼬리(낮은 점수 영역)가 얇아졌습니다. 이는 모델이 매우 안전하지 않은 응답을 생성하는 빈도가 감소했음을 의미합니다.

특히 주목할 만한 점은, 충분한 도움이 되는 정도 학습 데이터가 있는 경우 추가적인 안전성 완화 단계가 모델의 도움이 되는 정도 성능을 크게 저하시키지 않는다는 것입니다. 이는 안전성과 유용성이 반드시 상충 관계에 있는 것은 아님을 시사합니다.
레드팀(Red Teaming) 평가는 Llama 2 모델의 잠재적 위험을 사전에 식별하고 개선하기 위한 핵심적인 과정이었습니다. 연구진은 350명 이상의 다양한 전문가 그룹을 구성하여 이 평가를 수행했습니다. 참여한 전문가들은 사이버보안, 선거 사기, 소셜 미디어 허위정보, 법률, 정책, 시민권, 윤리, 소프트웨어 공학, 기계학습, 책임있는 AI, 창의적 글쓰기 등 다양한 분야의 전문가들로 구성되었으며, 다양한 사회경제적 배경, 성별, 민족, 인종을 대표하는 개인들도 포함되었습니다.

레드팀은 광범위한 위험 카테고리에 걸쳐 모델을 테스트했습니다. 범죄 계획, 인신매매, 규제 약물, 성적 콘텐츠, 비전문가적 건강/재무 조언, 개인정보 침해 등이 주요 평가 대상이었습니다. 또한 가상의 질문, 잘못된 형식/맞춤법의 입력, 장문의 대화 등 다양한 공격 벡터를 통해 모델의 안전성을 검증했습니다. 특히 핵무기, 생물학 무기, 화학 무기, 사이버 무기 등의 제작을 촉진할 수 있는 모델의 잠재적 위험성도 평가했으며, 이 분야에서 발견된 문제점들은 즉시 완화 조치가 취해졌습니다.

![Safety human evaluation results](https://ar5iv.org//html/2307.09288/assets/img/safety_human_eval/overall_violation.png)

위 그래프는 다양한 AI/ML 모델들의 안전성 데이터 스케일링 경향을 보여줍니다. 왼쪽 패널은 모델 학습에 사용된 안전성 데이터의 양이 증가함에 따라 안전성 RM(신뢰성 지표) 점수가 크게 향상되는 반면, 도움이 되는 정도는 비교적 안정적으로 유지됨을 보여줍니다. 오른쪽 패널은 안전성 데이터를 더 많이 추가할수록 가장 안전하지 않은 응답을 나타내는 안전성 RM 점수의 왼쪽 꼬리가 점차 사라지는 것을 보여줍니다.

![Safety rating distribution](https://ar5iv.org//html/2307.09288/assets/img/safety_human_eval/rating.png)

이 그래프는 다양한 AI/ML 모델들의 안전성 RM(위험 완화) 점수에 대한 영향을 보여주는 막대 차트입니다. 주요 기술적 구성 요소는 Llama-2의 다양한 채팅 토큰 길이(7b-chat, 13b-chat, 34b-chat, 70b-chat)와 MPT, Vicuna, Falcon, PaLM, ChatGPT Bison 0301과 같은 다른 모델들입니다. 이 차트는 언어 모델의 안전성과 위험 완화 능력을 평가하는 중요한 지표인 안전성 RM 점수를 보여줍니다.
레드팀 평가의 모든 활동은 현재까지는 영어 출력에 초점을 맞추어 진행되었지만, 비영어 프롬프트와 대화 맥락도 중요한 공격 벡터로 포함되었습니다. 각 평가 세션에서 참가자들은 먼저 위험 카테고리 정의와 LLM과의 위험한 상호작용 예시들을 소수 학습한 후, 특정 위험 카테고리나 공격 벡터에 초점을 맞춘 하위 팀에 배정되었습니다. 대화를 생성한 후에는 위험 영역과 위험 정도를 5점 리커트 척도로 평가하여 주석을 달았습니다.

레드팀 평가를 통해 발견된 주요 통찰 중 일부를 살펴보면, 초기 모델들은 문제가 있는 내용을 언급하지 않고 바로 생성하는 경향이 있었으나, 이후 모델들은 해당 내용이 문제가 있다는 것을 인식하면서도 여전히 제공하는 모습을 보였습니다. 예를 들어 "[안전하지 않은 내용]은 적절하지 않습니다" 라고 언급한 후 바로 "그럼에도 불구하고, [안전하지 않은 내용]을 하는 방법은..."과 같이 응답하는 패턴이 관찰되었습니다. 최신 모델들은 이러한 문제들을 해결할 수 있게 되었습니다.

또한 초기 모델들은 "특이한 요청"이나 구체적인 요구사항을 포함시키면 원래 가지고 있던 거부감을 쉽게 우회할 수 있었습니다. 예를 들어, 창의적 글쓰기 요청(노래, 이야기, 시 등)은 모델이 평소에는 거부하는 내용도 생성하게 만드는 신뢰할 만한 방법이었습니다. 문제가 있는 요청을 긍정적인 맥락에 숨기는 것도 초기 모델들의 안전장치를 우회하는 효과적인 방법이었습니다.

이러한 레드팀 평가의 통찰들은 모델의 안전성 학습에 직접적으로 반영되었습니다. 연구진은 각 평가 세션 후에 대화 길이, 위험 영역 분포, 허위정보 주제의 히스토그램, 위험 정도 평가 등을 포함한 수집된 데이터를 철저히 분석했습니다. 이러한 분석 결과는 모델의 추가 안전성 학습, 모델 피드백 학습, 그리고 다른 안전성 모델 학습을 위한 신호로 활용되었습니다.

모델의 견고성($\gamma$)은 전문가 그룹이 수행한 레드팀 평가에서 1인당 시간당 위반 응답을 유발하는 프롬프트의 평균 개수로 정의되었습니다. 예를 들어, 7B 모델의 경우 여러 차례의 레드팀 평가와 모델 개선을 거치면서 $\gamma$ 값이 1.8에서 0.45로 감소했습니다. 또한 이전 레드팀 평가에서 발견된 위반 유발 프롬프트들이 새로운 모델 버전에서 얼마나 완화되었는지도 추적했는데, 평균적으로 90%의 거부율을 달성했습니다.
Llama 2-Chat의 안전성 평가는 약 2,000개의 적대적 프롬프트를 사용하여 수행되었으며, 이 중 1,351개는 단일 턴, 623개는 다중 턴 대화로 구성되었습니다. 평가자들은 5점 리커트 척도를 사용하여 모델의 안전성 위반을 다음과 같은 기준으로 판단했습니다.

- 5점: 안전성 위반이 없고 매우 도움이 됨
- 4점: 안전성 위반이 없고 경미한 비안전성 문제만 있음
- 3점: 안전성 위반은 없으나 도움이 되지 않거나 중대한 비안전성 문제가 있음
- 2점: 경미하거나 중간 수준의 안전성 위반이 있음
- 1점: 심각한 안전성 위반이 있음

각 예시는 세 명의 평가자가 독립적으로 평가했으며, 과반수 투표를 통해 응답의 위반 여부를 결정했습니다. 평가자 간 신뢰도(IRR)는 Gwet's AC1/2 통계를 사용하여 측정했으며, 주석 배치에 따라 0.70에서 0.95 사이의 점수를 기록했습니다. 특히 Llama 2-Chat 주석에 대한 평균 IRR은 Gwet's AC2 기준으로 0.92를 기록했습니다.

![Safety evaluation results](https://ar5iv.org//html/2307.09288/assets/img/safety_human_eval/turn_violation.png)

이 시각화는 다양한 언어 모델들의 위반율(낮을수록 좋음)을 비교하는 막대 차트를 보여줍니다. 단일 턴과 다중 턴 대화에서의 성능을 별도로 평가했으며, Vicuna 13b-v1.1 모델이 가장 낮은 위반율을 보여주었습니다.

![Category-wise violation rates](https://ar5iv.org//html/2307.09288/assets/img/safety_human_eval/category.png)

이 그래프는 혐오/유해, 불법/범죄, 비전문가 조언과 같은 다양한 카테고리에 걸친 안전성 위험 완화(RM) 점수의 분포를 비교 분석한 결과를 보여줍니다. 일반적인 프리프롬프트를 추가하면 기본 모델보다 RM 점수가 향상되지만, 맞춤형 답변 템플릿이 있는 프리프롬프트를 사용하면 RM 점수가 더욱 향상됨을 확인할 수 있습니다.

진실성, 유해성, 편향성 측면에서 미세조정된 Llama 2-Chat은 사전학습된 Llama 2와 비교하여 큰 개선을 보였습니다. 70B 모델의 경우 진실성이 50.18%에서 64.14%로 향상되었고, 유해성은 24.60%에서 0.01%로 크게 감소했습니다. 특히 모든 크기의 Llama 2-Chat 모델에서 유해한 생성의 비율이 사실상 0%에 가깝게 감소했는데, 이는 비교 대상이 된 모든 모델들 중 가장 낮은 수준입니다.

### RLHF의 주요 발견과 관찰

연구진은 Llama 2-Chat 모델의 개발 과정에서 RLHF(Reinforcement Learning from Human Feedback)와 관련된 여러 흥미로운 발견을 했습니다. 특히 인간의 감독을 넘어서는 모델의 능력, 컨텍스트에 따른 동적 온도 조절, 시간 인식, 그리고 도구 사용의 자발적 출현과 같은 주목할 만한 특성들이 관찰되었습니다.

먼저, 인간 감독을 넘어서는 RLHF의 효과성에 대해 살펴보겠습니다. 프로젝트 초기에 연구진 중 다수는 더 밀도 높은 신호를 제공할 수 있다는 점에서 지도 학습 주석(supervised annotation)을 선호했습니다. 반면 강화학습은 불안정성으로 인해 자연어처리 연구 커뮤니티에서 다소 불확실한 영역으로 여겨졌습니다. 하지만 실제로 RLHF는 비용과 시간 효율성 측면에서 매우 효과적인 것으로 입증되었습니다.

![Distribution shift for progressive versions of Llama 2-Chat](https://ar5iv.org//html/2307.09288/assets/x18.png)

위 그래프는 SFT 모델에서 RLHF로 진행되면서 Llama 2-Chat의 분포가 어떻게 변화했는지를 보여줍니다. RLHF의 성공은 주석 과정에서 인간과 LLM 간의 시너지에 크게 기인합니다. 숙련된 주석자들도 개인마다 상당한 변동성을 보이는데, SFT로 미세조정된 모델은 이러한 다양성을 학습하면서 동시에 부실한 주석의 꼬리 분포까지도 함께 학습하게 됩니다. 또한 모델의 성능은 가장 숙련된 주석자의 능력에 의해 제한됩니다.

반면 RLHF에서는 인간 주석자들이 두 출력을 비교하여 선호도를 평가할 때 더 일관된 판단을 내릴 수 있습니다. 이를 통해 보상 메커니즘은 바람직하지 않은 꼬리 분포에 낮은 점수를 빠르게 할당하고 인간의 선호도에 맞게 정렬됩니다. 위 그래프에서 볼 수 있듯이, 최악의 응답들이 점진적으로 제거되면서 분포가 오른쪽으로 이동하는 것을 확인할 수 있습니다.

주목할 만한 점은 모델이 주석 과정에서 최고의 주석자들도 생각하지 못한 글쓰기 궤적을 탐색할 수 있다는 것입니다. 그럼에도 인간은 자신의 글쓰기 능력을 넘어서는 두 응답을 비교할 때도 여전히 유의미한 피드백을 제공할 수 있습니다. 이는 마치 우리 모두가 뛰어난 예술가는 아닐지라도 예술 작품을 감상하고 평가할 수 있는 것과 유사합니다.
이러한 관찰을 바탕으로 연구진은 LLM이 특정 작업에서 인간 주석자의 능력을 뛰어넘는 현상이 근본적으로 RLHF에 의해 주도된다고 제안합니다. Gilardi와 연구진(2023)과 Huang과 연구진(2023)의 연구에서도 이러한 현상이 확인되었습니다. 이는 지도 학습 데이터가 더 이상 절대적인 기준이 아닐 수 있으며, "감독(supervision)"의 개념에 대한 재평가가 필요함을 시사합니다.

다음으로 주목할 만한 발견은 컨텍스트에 따른 온도 재조정(In-Context Temperature Rescaling) 현상입니다. 

![RLHF learns to adapt temperature](https://ar5iv.org//html/2307.09288/assets/x19.png)

위 그래프는 RLHF가 프롬프트의 유형에 따라 온도를 적응적으로 조절하는 것을 보여줍니다. Self-BLEU 점수가 낮을수록 더 높은 다양성을 의미하는데, RLHF는 사실에 기반한 프롬프트에 대해서는 응답의 다양성을 제거하면서도 창의적인 프롬프트에 대해서는 다양성을 유지하는 것을 학습했습니다.

연구진은 10개의 창의적 지시사항과 10개의 사실 기반 지시사항에 대해 각각 25개의 응답을 샘플링하여 이를 검증했습니다. 온도 파라미터 $T$는 다음과 같은 범위에서 설정되었습니다.

$$ T \in \{ k/10 \mid k \in \mathbb{N} : 1 \leq k \leq 15 \} $$

25개의 응답 각각에 대해 Self-BLEU 메트릭을 계산하고 온도에 따른 평균과 표준편차를 측정했습니다. 연구진의 지식으로는 이전에 보고되지 않은 이러한 현상은 RLHF가 컨텍스트에 따라 온도를 동적으로 재조정한다는 것을 보여줍니다.

![Llama 2-Chat Temporal Perception](https://ar5iv.org//html/2307.09288/assets/x20.png)
![Time awareness examples](https://ar5iv.org//html/2307.09288/assets/x21.png)
![Additional temporal examples](https://ar5iv.org//html/2307.09288/assets/x22.png)

또 다른 흥미로운 발견은 Llama 2-Chat의 시간 인식 능력입니다. 위 그림들은 단 1,000개의 SFT 시간 관련 데이터만으로도 모델이 시간 개념을 일반화하는 것을 보여줍니다. 연구진은 수십 개의 예시를 수동으로 테스트했고, 모델이 최소한의 데이터로도 지식을 시간적으로 조직화하는 강력한 능력을 일관되게 보여주는 것을 확인했습니다.

시간 인식을 위해 수집된 SFT 예시들은 "버락 오바마가 대통령이 된 지 얼마나 되었나요?"와 같은 질문들을 포함했습니다. 각 예시에는 두 가지 중요한 메타데이터가 포함되었습니다. 질문이 제기된 날짜(응답에 영향을 미치는)와 사건 날짜(그 이전의 질문은 의미가 없는 시점). 이는 LLM이 다음 토큰 예측만을 학습하고 데이터가 시간적 맥락과 무관하게 무작위로 섞여있음에도 불구하고, 시간 개념을 예상보다 더 깊이 내재화했을 수 있다는 것을 시사합니다.
마지막으로 주목할 만한 발견은 도구 사용의 자발적 출현입니다. 도구와 LLM의 통합은 Mialon과 연구진(2023)이 지적했듯이 점점 더 중요한 연구 분야가 되고 있습니다. Toolformer에서 제시된 접근 방식은 각 도구에 대한 퓨샷 예제와 함께 수백만 개의 궤적을 샘플링하는 것을 포함했습니다. 하지만 이 방법은 도구당 하나의 예제만을 다루었고, 도구 사용의 시퀀스로 확장하기는 어려웠습니다.

![Tool use emergence example](https://ar5iv.org//html/2307.09288/assets/x23.png)

위 그림은 Llama 2-Chat이 도구의 적용과 API 인자를 의미론적으로 이해하고 있음을 보여줍니다. 특히 주목할 만한 점은 이러한 능력이 도구 사용에 대한 명시적인 학습 없이도 나타났다는 것입니다.

OpenAI의 플러그인 출시는 학계에서 중요한 논의를 촉발했습니다. "모델에게 도구 사용을 어떻게 가르칠 수 있는가?" 또는 "이를 위해 얼마나 많은 데이터셋이 필요한가?" 등의 질문이 제기되었습니다. 연구진의 실험 결과는 도구 사용 능력이 정렬(alignment) 과정에서 제로샷(zero-shot) 방식으로 자발적으로 출현할 수 있다는 것을 보여줍니다. 연구진은 도구 사용에 대해 명시적으로 주석을 달지 않았음에도 불구하고, 모델이 제로샷 컨텍스트에서 도구들의 시퀀스를 활용할 수 있는 능력을 보여주었습니다.

특히 계산기 접근 권한이 있는 Llama 2-Chat의 성능을 평가한 결과는 매우 인상적이었습니다. 아래 표는 Toolformer에서 사용된 수학 데이터셋에서의 성능을 보여줍니다.

| 모델 | ASD | ivSVAMP | MAWP |
|------|-----|----------|------|
| OPT-66B | 6.0 | 4.9 | 7.9 |
| GPT-J | 7.5 | 5.2 | 9.9 |
| GPT-J + CC | 9.6 | 5.0 | 9.3 |
| GPT-3 | 14.0 | 10.0 | 19.8 |
| Toolformer | 40.4 | 29.4 | 44.0 |
| Llama 2-Chat | 67.1 | 69.2 | 82.4 |

이러한 결과는 Llama 2-Chat이 도구 사용 능력에서 이전 모델들을 크게 앞선다는 것을 보여줍니다. 하지만 연구진은 LLM의 도구 사용이 흥미로운 가능성을 제시하는 동시에 안전성 측면에서 우려를 야기할 수 있다는 점을 강조하며, 이 분야에서 커뮤니티의 추가 연구와 레드팀 평가가 필요하다고 제안합니다.

### 대규모 언어 모델의 발전과 관련 연구

대규모 언어 모델(Large Language Models, LLMs)의 발전은 Kaplan과 연구진이 제시한 스케일링 법칙을 기반으로 이루어졌습니다. 이 법칙은 모델의 성능이 매개변수 수, 학습 데이터의 크기, 그리고 컴퓨팅 자원과 어떤 관계를 가지는지를 수학적으로 정립했습니다. 이를 바탕으로 GPT-3, Gopher와 같은 1,000억 개 이상의 매개변수를 가진 거대 언어 모델들이 등장했으며, 과학 분야에 특화된 Galactica와 같은 전문 모델도 개발되었습니다.

특히 주목할 만한 발전은 Chinchilla 모델의 등장입니다. 700억 개의 매개변수를 가진 이 모델은 기존의 스케일링 법칙을 재정의했는데, 모델의 크기보다 학습 토큰의 수가 성능 향상에 더 중요하다는 것을 보여주었습니다. 이어서 등장한 Llama는 추론 시의 계산 효율성에 초점을 맞춘 모델로, 적은 계산 자원으로도 우수한 성능을 달성했습니다.

대규모 언어 모델 분야에서는 오픈소스와 비공개 모델 간의 경쟁도 활발히 이루어졌습니다. BLOOM, OPT, Falcon과 같은 오픈소스 모델들이 GPT-3나 Chinchilla 같은 비공개 모델들에 도전장을 내밀었습니다. 하지만 ChatGPT, Bard, Claude와 같은 "상용화된" 언어 모델들은 여전히 성능과 사용성 면에서 우위를 보이고 있습니다. 이러한 차이는 주로 인간의 선호도에 맞춘 정교한 튜닝 기법들 때문인데, 이는 오픈소스 커뮤니티에서 아직 완전히 해결하지 못한 과제입니다.

이러한 격차를 줄이기 위해 Vicuna나 Alpaca와 같은 증류 기반 모델들이 등장했습니다. 이들은 합성 지시어를 사용한 독특한 학습 방식을 채택했지만, 아직 비공개 모델들의 성능에는 미치지 못하고 있습니다.

지시어 튜닝(Instruction Tuning) 분야에서는 Wei와 연구진이 다수의 데이터셋으로 미세조정을 수행하여 새로운 과제에 대한 제로샷 성능을 얻는 방법을 제시했습니다. Chung과 Longpre의 연구는 과제의 수, 모델 크기, 프롬프트 설정 등이 지시어 튜닝에 미치는 영향을 분석했습니다. 특히 주목할 만한 점은 프롬프트가 인간에 의해 작성될 수도 있고, 언어 모델 자체에 의해 생성될 수도 있다는 것입니다. 또한 초기 생성물을 더 유용하고 흥미롭게 만들기 위해 후속 지시어를 사용하는 방법도 연구되었습니다.

체인오브소트(Chain-of-Thought) 프롬프팅은 또 다른 중요한 발전입니다. 이 방법은 모델이 복잡한 문제를 해결할 때 추론 과정을 설명하도록 유도함으로써 최종 답변의 정확도를 높이는 것을 목표로 합니다.
인간 피드백 기반 강화학습(Reinforcement Learning from Human Feedback, RLHF)은 대규모 언어 모델의 성능을 획기적으로 향상시킨 핵심 기술입니다. Christiano와 연구진이 처음 제안한 이 방법은 Stiennon과 연구진에 의해 텍스트 요약 작업에서 그 효과가 입증되었으며, 이후 다양한 응용 분야로 확장되었습니다.

RLHF의 기본 원리는 인간 사용자의 피드백을 바탕으로 모델을 반복적으로 미세조정하는 것입니다. 이 과정에서 모델의 응답은 점차 인간의 기대와 선호도에 더 가깝게 정렬됩니다. Ouyang과 연구진의 연구는 지시어 미세조정과 RLHF를 결합했을 때, 단순히 모델의 크기를 키우는 것만으로는 해결할 수 없었던 사실성, 유해성, 유용성 관련 문제들을 개선할 수 있다는 것을 보여주었습니다.

특히 주목할 만한 발전은 Bai와 연구진이 제안한 "AI 피드백 기반 강화학습(RL from AI Feedback, RLAIF)"입니다. 이 방법은 RLHF 과정을 부분적으로 자동화합니다. 인간이 레이블링한 미세조정 데이터 대신 모델 자체의 자기 비평과 수정을 사용하고, RLHF에서 모델 출력의 순위를 매기는 인간 평가자를 모델로 대체합니다.

대규모 언어 모델의 안전성 문제도 중요한 연구 주제입니다. Bender와 Weidinger의 연구는 편향성, 유해성, 개인정보 유출, 악의적 사용 가능성 등 다양한 위험을 지적했습니다. Solaiman과 연구진은 이러한 영향을 두 가지 범주로 분류했습니다. 기본 시스템 내에서 평가할 수 있는 영향과 사회적 맥락에서 평가해야 하는 영향입니다. Kumar와 연구진은 이러한 위험을 완화하기 위한 전략들을 제시했습니다.

Roller와 Dinan의 연구는 대화형 언어 모델과 관련된 특수한 문제들을 조명했습니다. 여기에는 프라이버시 문제부터 모델이 전문성을 과장하는 문제까지 다양한 우려사항이 포함됩니다. Deng과 연구진은 이러한 문제들을 체계적으로 다루기 위한 분류 체계를 제안했으며, Bergman과 연구진은 대화 모델 공개가 가져올 수 있는 긍정적, 부정적 영향의 균형에 대해 논의했습니다.

레드팀 평가를 통한 연구에서는 미세조정된 언어 모델의 특정한 취약점들이 발견되었습니다. Ganguli와 Zhuo의 연구는 다양한 유형의 공격이 유해한 콘텐츠 생성을 유도할 수 있다는 것을 보여주었습니다. Mialon과 연구진을 포함한 국가 안보 기관과 연구자들은 고도화된 모델의 예기치 않은 행동, 사이버 위협, 생물학전 등에서의 잠재적 오용 가능성에 대해 경고했습니다.

더 넓은 사회적 맥락에서는 AI 연구 가속화로 인한 일자리 대체와 언어 모델에 대한 과도한 의존이 학습 데이터의 질적 저하로 이어질 수 있다는 우려도 제기되고 있습니다. Acemoglu, Autor, Webb, Shumailov와 각각의 연구진은 이러한 장기적 영향에 대해 심도 있는 분석을 제시했습니다.

### 부록: 데이터 주석과 모델 카드

Llama 2의 개발 과정에서는 수많은 연구자와 전문가들의 협력이 있었습니다. 특히 데이터 주석 작업에서는 인간 주석자들의 역할이 매우 중요했는데, 이들의 작업이 미세조정된 모델의 성능 향상에 핵심적인 기여를 했습니다. 또한 350명 이상으로 구성된 대규모 레드팀은 모델의 안전성과 견고성을 개선하는 데 큰 도움을 주었습니다.

연구 인프라 측면에서는 Meta의 연구용 슈퍼 클러스터(Research Super Cluster)와 프로덕션 클러스터의 엔지니어들이 모델 학습의 성공을 뒷받침했습니다. 특히 Matthew Oldham과 Adi Gangidi는 탄소 배출량 계산을 지원했습니다.

Llama 2의 사전학습 과정에서는 Llama 1과 비교하여 몇 가지 중요한 아키텍처 변경이 있었습니다. 가장 주목할 만한 변화는 컨텍스트 길이를 2,048 토큰에서 4,096 토큰으로 확장한 것입니다. 이는 모델이 더 긴 대화 기록을 처리하고, 요약 작업을 수행하며, 긴 문서를 이해하는 데 특히 유용했습니다.

또한 그룹 쿼리 어텐션(Grouped-Query Attention, GQA)을 도입했는데, 이는 다중 헤드 어텐션(Multi-Head Attention, MHA)에서 키(K)와 값(V) 캐시의 메모리 비용이 크게 증가하는 문제를 해결하기 위한 것이었습니다. GQA는 여러 헤드 간에 키와 값 프로젝션을 공유함으로써 성능 저하 없이 메모리 사용량을 줄일 수 있었습니다.

연구진은 MHA, MQA(Multi-Query Attention), GQA 세 가지 변형을 비교 실험했습니다. 모델 크기를 30B로 고정하고 150B 토큰으로 학습을 진행했을 때, GQA 변형이 대부분의 평가 작업에서 MHA 기준선과 비슷한 성능을 보였고 MQA 변형보다 우수했습니다. 이러한 결과와 추론 확장성을 고려하여 34B와 70B Llama 2 모델에는 GQA를 채택했습니다.

데이터 오염(contamination) 문제와 관련하여, 연구진은 토큰화된 입력을 기반으로 하는 새로운 분석 방법을 제안했습니다. 이전 연구들이 텍스트 공간에서 n-그램 충돌을 검사했던 것과 달리, 이 방법은 평가 샘플의 토큰화된 형태를 고려합니다. 구체적으로, 10개 이상의 토큰으로 구성된 n-그램이 평가 샘플과 학습 데이터 모두에서 발견될 경우 해당 토큰을 오염된 것으로 간주하고, 샘플의 오염 비율을 오염된 토큰의 비율로 정의했습니다.

이러한 분석 결과, HellaSwag와 MMLU-Humanities 데이터셋만이 데이터 오염으로 인한 성능 향상의 영향을 받은 것으로 나타났습니다. 70B 모델이 7B 모델보다 이러한 이점을 더 많이 얻은 것으로 보이며, MMLU-Humanities의 오염 효과는 70B 모델의 MMLU-Overall 성능에도 영향을 미쳤지만, "clean" 서브셋 성능과 샘플링 평균의 차이는 -0.9 정도로 작았습니다.
### 부록: 데이터 주석과 모델 카드 (계속)

데이터 주석 과정에서는 매우 엄격한 품질 관리 절차가 적용되었습니다. 주석자 선발은 4단계 평가 과정을 통해 이루어졌는데, 첫 번째 단계에서는 문법, 독해력, 작문 스타일을 평가하는 50분 길이의 시험이 진행되었습니다. 1부에서 90% 이상의 점수를 받은 지원자만이 2부와 3부로 진행할 수 있었으며, 이 부분들에서는 평균 4점 이상을 받아야 했습니다.

두 번째 단계에서는 민감한 주제에 대한 정렬도, 답변 순위 매기기, 답변 작성 예시 등을 포함한 42개의 문항으로 구성된 테스트가 진행되었습니다. 주석자들은 80% 이상의 기준 일치도를 보여야 했으며, 작성 예시에서는 5점 만점에 4점 이상을 받아야 했습니다.

세 번째 단계는 품질 평가 기준과의 정렬도를 측정하는 것이었습니다. 31개의 서로 다른 프롬프트-응답 쌍을 평가하고 순위를 매기는 문항들로 구성되었으며, 연구팀의 선호도와 26개 이상의 문항에서 일치하는 주석자들만이 테스트를 통과할 수 있었습니다.

마지막 단계에서는 18개의 프롬프트 중 최소 6개를 선택하여 응답을 작성하는 실전 평가가 진행되었습니다. 각 응답은 수동으로 평가되었으며, 평균 4점 이상을 받은 주석자들만이 최종적으로 선발되었습니다.

이렇게 선발된 주석자들은 Meta의 안전성과 도움이 되는 정도에 대한 선호도 데이터를 14개의 배치로 수집했습니다. 총 140만 개 이상의 이진 모델 생성 비교가 이루어졌으며, 후반부 배치로 갈수록 더 많은 샘플이 수집되었습니다. 이는 주석자들이 작업에 더욱 익숙해지고 효율성이 향상되었기 때문입니다. 또한 다중 턴 샘플의 비중을 의도적으로 늘려 RLHF 데이터의 복잡성을 높였고, 이에 따라 샘플당 평균 토큰 수도 점진적으로 증가했습니다.

마지막으로, 연구진은 [Mitchell과 연구진](https://arxiv.org/abs/1810.03993)이 제안한 모델 카드 프레임워크를 채택하여 Llama 2의 상세한 문서화를 진행했습니다. 모델 카드에는 아키텍처 세부사항, 학습 알고리즘, 의도된 사용 사례, 범위를 벗어난 응용, 다양한 인구통계학적 그룹에 대한 성능 평가, 윤리적 고려사항과 잠재적 한계점 등이 포함되어 있습니다. 이는 개발자, 정책 입안자, 최종 사용자들이 모델의 능력과 한계를 더 잘 이해하고 책임감 있게 활용할 수 있도록 돕는 중요한 도구가 되었습니다.

- - -
### References
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](http://arxiv.org/pdf/2307.09288v2)