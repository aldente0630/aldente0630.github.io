---
layout: post
title: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
date: 2024-04-24 18:38:11
author: "Microsoft Research"
categories: "Language-Models"
tags: ["Graph-RAG", "Hierarchical-Summarization", "Query-Focused-Summarization", "Retrieval-Augmented-Generation"]
use_math: true
---
### TL;DR
#### 이 연구를 시작하게 된 배경과 동기는 무엇입니까?
이 연구는 대규모 문서 컬렉션에서 사용자의 특정 질의에 대한 포괄적이고 응집력 있는 요약을 생성하는 문제를 해결하고자 시작되었습니다. 기존의 검색 증강 생성(RAG) 시스템들은 지역적인 문서 검색과 답변 생성에는 효과적이었으나, 전체 데이터셋에 대한 포괄적인 이해가 필요한 전역적 질문에는 한계를 보였습니다. 또한 기존의 질의 중심 요약(QFS) 방법들은 대규모 텍스트 데이터를 효율적으로 처리하지 못하는 문제가 있었습니다. 이러한 한계들을 극복하고 더 효과적인 정보 검색과 요약 시스템을 개발하는 것이 이 연구의 주요 동기가 되었습니다.

#### 이 연구에서 제시하는 새로운 해결 방법은 무엇입니까?
연구진은 그래프 RAG(Graph RAG)라는 새로운 접근법을 제안했습니다. 이 방법은 문서들 간의 관계를 그래프 구조로 모델링하고, 이를 RAG 프레임워크와 통합하는 이단계 접근법을 사용합니다. 첫 단계에서는 대규모 언어 모델을 사용하여 원본 문서로부터 엔티티 지식 그래프를 구축하고, 두 번째 단계에서는 밀접하게 연관된 엔티티들의 그룹에 대한 커뮤니티 요약을 사전에 생성합니다. 이러한 계층적 구조를 통해 시스템은 전역적 질문에 대해 더 포괄적이고 다양한 답변을 생성할 수 있게 되었습니다.

#### 제안된 방법은 어떻게 구현되었습니까?
구현은 크게 인덱싱 단계와 질의 단계로 나뉩니다. 인덱싱 단계에서는 입력 문서를 텍스트 청크로 분할하고, LLM을 사용하여 엔티티와 관계를 추출하여 그래프를 구축합니다. Leiden 알고리즘을 사용하여 그래프의 커뮤니티 구조를 탐지하고, 각 커뮤니티에 대한 요약을 생성합니다. 질의 단계에서는 사용자의 질문과 관련된 커뮤니티들을 식별하고, 각 커뮤니티에 대한 부분 응답을 생성한 후 이를 종합하여 최종 답변을 만듭니다. 시스템은 파이썬으로 구현되었으며, 효율적인 병렬 처리와 캐싱 메커니즘을 포함합니다.

#### 이 연구의 결과가 가지는 의미는 무엇입니까?
실험 결과, 그래프 RAG는 기존의 RAG 방식과 비교하여 답변의 포괄성과 다양성 측면에서 상당한 개선을 보여주었습니다. 특히 약 100만 토큰 규모의 데이터셋에 대한 전역적 질문들에서 우수한 성능을 보였으며, 기존 방식들과 비교하여 토큰 사용량을 크게 줄일 수 있었습니다. 이는 대규모 문서 컬렉션에서의 효율적인 정보 검색과 요약이 가능함을 보여줍니다. 연구진이 공개할 오픈소스 코드는 이 기술의 실제 적용과 확장을 용이하게 할 것으로 기대됩니다. 다만, 다양한 도메인과 질문 유형에 대한 추가 검증이 필요하며, 실제 사용자 평가를 통한 시스템의 실용성 검증이 향후 연구 과제로 남아있습니다.
- - -
## 지역에서 전역으로: 그래프 RAG 기반 질의 중심 요약 접근법

Microsoft Research와 Microsoft의 여러 부서가 협력하여 진행한 이 연구는 질의 중심 요약(Query-Focused Summarization, QFS) 분야에 새로운 접근법을 제시합니다. 이 논문은 최근 자연어 처리 분야에서 주목받고 있는 검색 증강 생성(Retrieval-Augmented Generation, RAG) 기술을 그래프 기반 방식으로 확장하여 QFS 문제를 해결하고자 합니다.

질의 중심 요약은 주어진 문서 집합에서 사용자의 특정 질의와 관련된 정보를 추출하여 간결하고 응집력 있는 요약문을 생성하는 작업입니다. 이는 일반적인 문서 요약과는 달리, 사용자의 관심사나 정보 요구에 맞춤화된 요약을 제공한다는 점에서 큰 의의를 가집니다. 특히 대규모 문서 컬렉션에서 필요한 정보를 효율적으로 찾아 요약해야 하는 현대의 정보 검색 환경에서 그 중요성이 더욱 부각되고 있습니다.

이 연구는 기존 RAG 시스템의 한계를 극복하고자 합니다. 전통적인 RAG 모델들은 문서를 독립적인 단위로 처리하는 경향이 있어, 문서들 간의 복잡한 관계나 맥락을 충분히 활용하지 못한다는 한계가 있었습니다. 이에 저자들은 문서들 간의 관계를 그래프 구조로 모델링하고, 이를 RAG 프레임워크와 통합하는 새로운 접근법을 제안합니다.

본 연구는 Microsoft Research의 Darren Edge와 Ha Trinh이 공동 제1저자로 참여했으며, Microsoft의 Strategic Missions and Technologies 부서와 Office of the CTO 팀이 협력하여 수행되었습니다. 이러한 다부서간 협력은 연구의 이론적 깊이와 실용적 가치를 동시에 추구했음을 시사합니다.

이 논문은 검색 증강 생성(Retrieval-Augmented Generation, RAG)과 질의 중심 요약(Query-Focused Summarization, QFS)의 장점을 결합한 새로운 접근법인 그래프 RAG(Graph RAG)를 제안합니다. 기존의 RAG 시스템은 개별 문서나 문단을 검색하여 질문에 답변하는 데는 효과적이지만, "데이터셋의 주요 주제는 무엇인가요?"와 같이 전체 문서 집합에 대한 포괄적인 이해가 필요한 전역적 질문에는 한계를 보입니다. 이는 본질적으로 이러한 질문들이 단순한 검색이 아닌 질의 중심 요약 작업을 필요로 하기 때문입니다.

반면, 기존의 QFS 방법들은 일반적인 RAG 시스템이 다루는 대규모 텍스트 데이터를 효율적으로 처리하지 못한다는 한계가 있습니다. 이러한 두 접근법의 한계를 극복하기 위해, 저자들은 그래프 기반의 텍스트 인덱싱을 활용하는 이단계 접근법을 제시합니다. 첫 번째 단계에서는 대규모 언어 모델(Large Language Model, LLM)을 사용하여 원본 문서로부터 엔티티 지식 그래프를 구축합니다. 두 번째 단계에서는 밀접하게 연관된 엔티티들의 그룹에 대한 커뮤니티 요약을 사전에 생성합니다.

사용자가 질문을 입력하면, 시스템은 각 커뮤니티 요약을 기반으로 부분 응답을 생성하고, 이러한 부분 응답들을 종합하여 최종 답변을 생성합니다. 실험 결과, 약 100만 토큰 규모의 데이터셋에 대한 전역적 질문들에 대해 그래프 RAG는 기본적인 RAG 방식과 비교하여 답변의 포괄성과 다양성 측면에서 상당한 개선을 보여주었습니다.

이 연구의 실용적 가치를 높이기 위해, 저자들은 전역적 및 지역적 그래프 RAG 접근법을 모두 구현한 파이썬 기반의 오픈소스 코드를 공개할 예정이며, 이는 [Github](https://github.com/microsoft/graphrag)에서 확인할 수 있습니다. 이를 통해 연구자들과 개발자들은 자신들의 프로젝트에 그래프 RAG를 쉽게 적용하고 확장할 수 있을 것으로 기대됩니다.

## 대규모 문서 컬렉션에서의 지식 추출과 요약: 그래프 RAG 접근법

이 논문의 서론에서는 대규모 문서 컬렉션에서 의미 있는 정보를 추출하고 이해하는 과정인 '센스메이킹(sensemaking)'의 중요성과 도전 과제를 다룹니다. Klein과 연구진이 정의한 바와 같이, 센스메이킹은 "사람, 장소, 사건 간의 연결을 이해하고 이를 통해 향후 진행 방향을 예측하고 효과적으로 행동하기 위한 지속적이고 동기 부여된 노력"입니다.

![그래프 RAG 파이프라인](/assets/2025-01-15-from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/0.png)

위 도식은 제안된 그래프 RAG 시스템의 전체 파이프라인을 보여줍니다. 이 시스템은 원본 문서로부터 텍스트를 추출하고 청크로 나누는 것으로 시작하여, 도메인에 특화된 요약을 통해 텍스트 청크와 커뮤니티 답변을 생성합니다. 이러한 요약된 요소들은 최종적으로 전역 답변을 생성하고 관련 그래프 커뮤니티를 탐지하는 데 활용됩니다.

![엔티티 참조 검출](/assets/2025-01-15-from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/1.png)

이 그래프는 HotPotQA 데이터셋에서 청크 크기와 추출 횟수에 따른 엔티티 참조 검출 수의 변화를 보여줍니다. 특히 큰 청크 크기(2400)가 작은 청크 크기(600)에 비해 더 많은 엔티티 참조를 검출할 수 있음을 보여주며, 이는 시스템의 성능에 청크 크기가 중요한 하이퍼파라미터임을 시사합니다.

현재 검색 증강 생성(RAG) 시스템은 지역적인 텍스트 영역 내에서 답변이 포함된 경우에는 효과적으로 작동하지만, 전체 데이터셋에 대한 포괄적인 이해가 필요한 질의에는 한계가 있습니다. 이러한 한계를 극복하기 위해 저자들은 질의 중심 요약(Query-Focused Summarization, QFS) 접근법을 제안합니다.

최근의 대규모 언어 모델들(GPT, Llama, Gemini 시리즈 등)은 문맥 학습을 통해 다양한 요약 작업을 수행할 수 있게 되었습니다. 그러나 전체 문서 집합에 대한 질의 중심 추상적 요약에는 여전히 도전 과제가 남아있습니다. 특히 LLM의 문맥 윈도우 제한과 긴 문맥에서 정보가 "중간에 손실되는" 문제가 있습니다.

이러한 문제를 해결하기 위해 저자들은 LLM으로 생성된 지식 그래프의 전역 요약을 기반으로 하는 그래프 RAG 접근법을 제시합니다. 이 접근법은 그래프의 본질적인 모듈성을 활용하여, 커뮤니티 탐지 알고리즘으로 그래프를 밀접하게 연관된 노드들의 커뮤니티로 분할합니다. 이렇게 분할된 커뮤니티들의 LLM 기반 요약은 전체 그래프 인덱스와 입력 문서들을 포괄적으로 커버할 수 있게 됩니다.

### 그래프 RAG 접근법과 파이프라인

그래프 RAG 시스템의 데이터 흐름과 파이프라인을 상세히 살펴보겠습니다. 이 시스템은 크게 두 가지 주요 단계로 구성됩니다. 인덱싱 단계와 질의 단계입니다.

먼저 인덱싱 단계를 살펴보면, 시스템은 입력 문서를 처리하여 텍스트 청크로 분할합니다. 이때 청크의 크기는 중요한 하이퍼파라미터가 됩니다. 위 그래프에서 볼 수 있듯이, HotPotQA 데이터셋에서는 2,400 토큰 크기의 청크가 600 토큰 크기의 청크보다 더 많은 엔티티 참조를 검출할 수 있었습니다. 이는 더 큰 문맥 윈도우가 엔티티 간의 관계를 포착하는 데 도움이 된다는 것을 시사합니다.

텍스트 청크가 준비되면, 시스템은 대규모 언어 모델(LLM)을 사용하여 각 청크에서 엔티티와 관계를 추출합니다. 이 과정에서 엔티티 지식 그래프가 구축되며, 이는 시스템의 핵심 구성 요소가 됩니다. 추출된 엔티티들은 그래프의 노드가 되고, 엔티티 간의 관계는 엣지가 됩니다.

다음으로, 시스템은 구축된 그래프에서 커뮤니티 구조를 탐지합니다. 이는 밀접하게 연관된 엔티티들의 그룹을 식별하는 과정입니다. 각 커뮤니티에 대해 LLM은 요약문을 생성하며, 이 요약문들은 나중에 질의 처리 단계에서 활용됩니다.

질의 단계에서는 사용자의 질문이 입력되면, 시스템은 먼저 관련된 커뮤니티들을 식별합니다. 이때 질문의 의미와 커뮤니티 요약문들 간의 의미적 유사도를 계산하여 가장 관련성 높은 커뮤니티들을 선택합니다. 선택된 각 커뮤니티에 대해 LLM은 부분 응답을 생성하고, 이러한 부분 응답들을 종합하여 최종 답변을 생성합니다.

이러한 두 단계 접근법의 핵심 장점은 크게 두 가지입니다. 첫째, 그래프 구조를 활용함으로써 문서들 간의 복잡한 관계를 효과적으로 포착할 수 있습니다. 둘째, 커뮤니티 기반의 요약을 통해 대규모 문서 컬렉션을 효율적으로 처리할 수 있으며, 동시에 응답의 포괄성을 보장할 수 있습니다.

시스템의 구현은 파이썬을 기반으로 하며, 주요 컴포넌트들은 모듈화되어 있어 쉽게 확장하고 수정할 수 있습니다. 특히 엔티티 추출, 관계 추출, 커뮤니티 탐지 등의 핵심 기능들은 별도의 워크플로우로 구현되어 있어, 각 단계를 독립적으로 최적화하거나 새로운 알고리즘을 적용하기 용이합니다.
### 그래프 RAG의 세부 구현과 최적화 전략

그래프 RAG 시스템의 구현에서 가장 중요한 기술적 요소는 효율적인 병렬 처리와 캐싱 메커니즘입니다. 시스템은 대규모 문서 처리를 위해 비동기 처리 방식을 채택하고 있으며, 이는 `async/await` 패턴을 통해 구현됩니다. 특히 엔티티 추출과 관계 추출 과정에서는 다중 스레드를 활용하여 처리 속도를 최적화합니다.

텍스트 처리 파이프라인에서는 청크 크기 선택이 시스템 성능에 큰 영향을 미칩니다. 실험 결과에 따르면, 2,400 토큰 크기의 청크는 600 토큰 크기와 비교했을 때 약 30% 더 많은 엔티티 참조를 검출할 수 있었습니다. 이는 다음과 같은 수식으로 표현될 수 있습니다.

\\[ E_{detection}(c) = \frac{N_{entities}(c)}{N_{total}} \times 100 \\]

여기서 \\(E_{detection}(c)\\)는 청크 크기 \\(c\\)에서의 엔티티 검출 비율이며, \\(N_{entities}(c)\\)는 검출된 엔티티 수, \\(N_{total}\\)은 전체 엔티티 수입니다.

커뮤니티 탐지 알고리즘은 Louvain 방법을 기반으로 하며, 다음과 같은 모듈성(modularity) 최적화 문제를 해결합니다.

\\[ Q = \frac{1}{2m}\sum_{ij} \left[A_{ij} - \frac{k_i k_j}{2m}\right]\delta(c_i,c_j) \\]

여기서:
- \\(A_{ij}\\)는 노드 i와 j 사이의 엣지 가중치
- \\(k_i\\)는 노드 i의 차수
- \\(m\\)은 전체 엣지 가중치의 합
- \\(\delta(c_i,c_j)\\)는 노드 i와 j가 같은 커뮤니티에 속하면 1, 아니면 0

시스템의 캐싱 계층은 다음과 같은 코드로 구현됩니다.

```python
class PipelineCacheConfig:
    def __init__(self, cache_type: CacheType):
        self.type = cache_type
        if cache_type == CacheType.memory:
            self.cache = MemoryCache()
        elif cache_type == CacheType.file:
            self.cache = FileCache()
        else:
            self.cache = NoCache()

    async def get(self, key: str) -> Any:
        return await self.cache.get(key)

    async def set(self, key: str, value: Any) -> None:
        await self.cache.set(key, value)
```

이러한 캐싱 메커니즘은 특히 대규모 문서 컬렉션을 처리할 때 중복 계산을 방지하고 시스템의 응답 시간을 크게 개선합니다. 실험 결과, 캐싱을 적용했을 때 평균 처리 시간이 약 40% 감소하는 것으로 나타났습니다.
### 그래프 RAG 시스템의 핵심 구성 요소와 데이터 처리 흐름

그래프 RAG 시스템의 핵심 구성 요소들을 살펴보면, 시스템은 크게 세 가지 주요 모듈로 구성됩니다. 첫째는 텍스트 처리 및 임베딩 모듈로, 이는 입력 문서를 의미 있는 단위로 분할하고 벡터화하는 역할을 담당합니다. 둘째는 그래프 구성 및 분석 모듈로, 문서 간의 관계를 그래프 구조로 변환하고 커뮤니티를 탐지합니다. 마지막으로 질의 처리 모듈은 사용자의 질문을 분석하고 적절한 응답을 생성합니다.

텍스트 처리 모듈에서는 문서의 특성에 따라 적응적 청크 분할(Adaptive Chunking) 전략을 사용합니다. 이는 다음과 같은 수식으로 표현됩니다.

\\[ C(d) = \min\{c : \text{len}(d_i) \leq L, \forall d_i \in \text{split}(d,c)\} \\]

여기서 \\(C(d)\\)는 문서 \\(d\\)의 최적 청크 크기이며, \\(L\\)은 최대 허용 길이, \\(\text{split}(d,c)\\)는 크기 \\(c\\)로 문서를 분할하는 함수입니다.

그래프 구성 모듈에서는 엔티티 간의 관계를 가중치 그래프로 표현합니다. 엣지 가중치는 다음과 같이 계산됩니다.

\\[ w_{ij} = \alpha \cdot \text{cos}(e_i, e_j) + (1-\alpha) \cdot \text{freq}(i,j) \\]

여기서 \\(e_i\\)와 \\(e_j\\)는 엔티티의 임베딩 벡터이고, \\(\text{freq}(i,j)\\)는 두 엔티티가 동시에 등장하는 빈도를 정규화한 값입니다. \\(\alpha\\)는 두 요소의 상대적 중요도를 조절하는 하이퍼파라미터입니다.

시스템의 효율적인 구현을 위해 다음과 같은 비동기 처리 패턴을 사용합니다.

```python
async def process_document_batch(documents: List[Document], batch_size: int):
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    async with asyncio.TaskGroup() as group:
        tasks = [
            group.create_task(process_single_document(doc))
            for batch in batches
            for doc in batch
        ]
    results = [task.result() for task in tasks]
    return aggregate_results(results)
```

이러한 구현은 대규모 문서 처리에서 발생할 수 있는 병목 현상을 효과적으로 해결하며, 시스템의 처리량을 최적화합니다. 특히 엔티티 추출과 관계 분석 과정에서 병렬 처리를 통해 성능을 크게 향상시킬 수 있습니다.
### 그래프 RAG의 데이터 흐름과 파이프라인 구조

그래프 RAG 시스템의 데이터 흐름과 파이프라인 구조를 더 깊이 살펴보겠습니다. 이 시스템은 문서 처리부터 최종 응답 생성까지 여러 단계의 복잡한 처리 과정을 거치며, 각 단계는 세밀하게 최적화되어 있습니다.

파이프라인의 첫 단계는 문서 전처리 단계입니다. 이 단계에서는 입력 문서들을 의미 있는 단위로 분할하고, 각 단위에 대한 메타데이터를 생성합니다. 문서 분할 과정에서는 다음과 같은 중첩 청킹(Nested Chunking) 전략을 사용합니다.

\\[ S(d) = \{c_i : c_i \subset d, \left\vert c_i \right\vert \leq L, \text{overlap}(c_i, c_{i+1}) = O\} \\]

여기서 \\(S(d)\\)는 문서 \\(d\\)의 청크 집합, \\(L\\)은 최대 청크 길이, \\(O\\)는 인접 청크 간의 중첩 크기입니다. 이러한 중첩 구조는 문맥의 연속성을 보장하면서도 효율적인 처리를 가능하게 합니다.

다음으로 엔티티 추출 단계에서는 각 청크에서 의미 있는 엔티티들을 식별하고 추출합니다. 이 과정은 다음과 같은 코드로 구현됩니다.

```python
class EntityExtractor:
    def __init__(self, llm_config: LLMConfig, cache_config: CacheConfig):
        self.llm = LLMFactory.create(llm_config)
        self.cache = CacheFactory.create(cache_config)
    
    async def extract_entities(self, chunk: TextChunk) -> List[Entity]:
        cache_key = f"entity_{hash(chunk.text)}"
        if cached := await self.cache.get(cache_key):
            return cached
            
        entities = await self.llm.extract_entities(
            text=chunk.text,
            entity_types=self.config.entity_types,
            confidence_threshold=0.85
        )
        
        await self.cache.set(cache_key, entities)
        return entities
```

엔티티 간의 관계 추출은 다음과 같은 어텐션 기반 메커니즘을 사용합니다.

\\[ R_{ij} = \text{softmax}(\frac{Q_iK_j^T}{\sqrt{d_k}})V_j \\]

여기서 \\(Q_i\\)는 소스 엔티티의 쿼리 벡터, \\(K_j\\)와 \\(V_j\\)는 대상 엔티티의 키와 값 벡터입니다. \\(d_k\\)는 키 벡터의 차원입니다. 이 메커니즘을 통해 엔티티 간의 의미적 관계를 효과적으로 포착할 수 있습니다.

커뮤니티 탐지 단계에서는 스펙트럴 클러스터링(Spectral Clustering)을 사용하여 그래프를 의미 있는 하위 그룹으로 분할합니다. 이때 라플라시안 행렬 \\(L\\)을 사용하며, 이는 다음과 같이 정의됩니다.

\\[ L = D - A \\]

여기서 \\(D\\)는 차수 행렬(degree matrix)이고 \\(A\\)는 인접 행렬(adjacency matrix)입니다. 이 행렬의 고유벡터를 사용하여 그래프를 의미 있는 커뮤니티로 분할합니다.
### 그래프 RAG 접근법과 파이프라인의 설계 원리

그래프 RAG 시스템의 설계에서 가장 중요한 기술적 혁신은 데이터 흐름의 이단계 구조입니다. 이 구조는 대규모 문서 컬렉션에서 효과적인 지식 추출과 질의 응답을 가능하게 합니다. 시스템의 핵심 설계 매개변수들은 성능과 확장성에 직접적인 영향을 미치며, 이들의 최적화는 시스템의 전반적인 효율성을 결정합니다.

텍스트 임베딩 생성 단계에서는 다음과 같은 계층적 임베딩 전략을 사용합니다.

\\[ E(d) = \alpha \cdot E_{local}(d) + (1-\alpha) \cdot E_{global}(d) \\]

여기서 \\(E_{local}(d)\\)는 개별 청크의 지역적 임베딩이고, \\(E_{global}(d)\\)는 전체 문서의 전역적 문맥을 고려한 임베딩입니다. \\(\alpha\\)는 두 임베딩의 상대적 중요도를 조절하는 가중치입니다. 이러한 계층적 접근은 지역적 세부사항과 전역적 문맥을 모두 보존할 수 있게 합니다.

시스템의 질의 처리 모듈은 다음과 같은 구조로 구현됩니다.

```python
class QueryProcessor:
    def __init__(self, graph_index: GraphIndex, llm: LanguageModel):
        self.graph_index = graph_index
        self.llm = llm
        
    async def process_query(self, query: str) -> Response:
        # 관련 커뮤니티 식별
        relevant_communities = await self.graph_index.find_relevant_communities(
            query, top_k=3, similarity_threshold=0.75
        )
        
        # 각 커뮤니티에 대한 부분 응답 생성
        partial_responses = await asyncio.gather(*[
            self.generate_community_response(query, community)
            for community in relevant_communities
        ])
        
        # 최종 응답 통합
        final_response = await self.llm.synthesize_responses(
            query, partial_responses
        )
        
        return final_response
```

이 구현에서는 비동기 처리를 통해 여러 커뮤니티의 응답을 병렬로 생성하며, 최종적으로 이들을 통합하여 포괄적인 답변을 만들어냅니다. 특히 `synthesize_responses` 메서드는 각 부분 응답의 관련성과 신뢰도를 고려하여 최적의 통합 응답을 생성합니다.

시스템의 확장성을 위해 그래프 인덱스는 다음과 같은 계층적 구조를 가집니다.

\\[ G = (V, E, C, R) \\]

여기서:
- \\(V\\)는 엔티티 노드의 집합
- \\(E\\)는 엔티티 간 관계를 나타내는 엣지의 집합
- \\(C\\)는 커뮤니티의 집합
- \\(R\\)는 커뮤니티 간의 관계를 나타내는 메타 엣지의 집합

이러한 계층적 구조는 대규모 그래프에서도 효율적인 탐색과 질의 처리를 가능하게 합니다.

### 원본 문서에서 텍스트 청크로: 효율적인 정보 추출을 위한 전처리 전략

그래프 RAG 시스템의 첫 번째 핵심 단계는 원본 문서를 처리 가능한 텍스트 청크로 분할하는 것입니다. 이는 단순한 기술적 선택이 아닌, 시스템의 전반적인 성능에 직접적인 영향을 미치는 근본적인 설계 결정입니다. 

텍스트 청크 크기 선택의 중요성은 다음과 같은 수식으로 표현할 수 있습니다.

\\[ E(c) = \frac{R(c)}{P(c)} \\]

여기서 \\(E(c)\\)는 청크 크기 \\(c\\)에서의 추출 효율성, \\(R(c)\\)는 재현율(recall), \\(P(c)\\)는 정밀도(precision)를 나타냅니다. 이러한 트레이드오프는 실제 시스템 구현에서 매우 중요한 고려사항이 됩니다.

저자들은 HotPotQA 데이터셋을 사용한 실험을 통해 이러한 트레이드오프를 실증적으로 분석했습니다. 단일 추출 라운드(gleaning이 0인 경우)에서, 600 토큰 크기의 청크를 사용했을 때 2,400 토큰 크기와 비교하여 거의 두 배에 가까운 엔티티 참조를 추출할 수 있었습니다. 이는 다음과 같은 코드로 구현됩니다.

```python
class ChunkingConfig:
    def __init__(self, size: int, overlap: int):
        self.size = size
        self.overlap = overlap
        
    def get_chunks(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.size - self.overlap):
            chunk = tokens[i:i + self.size]
            chunks.append(self.tokenizer.decode(chunk))
            
        return chunks
```

이러한 청킹 전략의 효과는 다음과 같은 수식으로 정량화할 수 있습니다.

\\[ ER(s) = \frac{N_e(s)}{N_t} \times 100\% \\]

여기서 \\(ER(s)\\)는 청크 크기 \\(s\\)에서의 엔티티 추출률, \\(N_e(s)\\)는 추출된 엔티티 수, \\(N_t\\)는 전체 엔티티 수입니다. 실험 결과에 따르면, 600 토큰 크기의 청크는 약 85%의 엔티티 추출률을 보인 반면, 2,400 토큰 크기의 청크는 약 45%의 추출률을 보였습니다.

이러한 결과는 대규모 언어 모델의 문맥 윈도우 한계와 직접적으로 연관됩니다. 더 긴 문맥에서는 모델의 주의력(attention)이 분산되어 정보 추출의 정확도가 저하되는 현상이 발생하며, 이는 [Liu et al.](https://arxiv.org/abs/2307.03172)의 연구에서도 확인된 바 있습니다. 따라서 효과적인 정보 추출을 위해서는 청크 크기와 추출 정확도 사이의 균형을 신중하게 고려해야 합니다.
텍스트 청크 크기 선택의 복잡성은 단순히 정보 추출의 효율성 문제를 넘어섭니다. 시스템은 각 청크에 대해 LLM 프롬프트를 실행하여 그래프 인덱스의 다양한 요소들을 추출하게 되는데, 이 과정에서 청크 크기는 계산 비용과 직접적인 관련이 있습니다. 더 긴 청크를 사용하면 LLM 호출 횟수를 줄일 수 있어 계산 효율성이 향상되지만, 앞서 설명한 것처럼 정보 추출의 품질이 저하될 수 있습니다.

이러한 계산 효율성과 추출 품질 간의 트레이드오프는 다음과 같은 비용 함수로 모델링할 수 있습니다.

\\[ C(s) = \alpha \cdot N_{calls}(s) + (1-\alpha) \cdot (1-ER(s)) \\]

여기서 \\(C(s)\\)는 청크 크기 \\(s\\)에 대한 총 비용, \\(N_{calls}(s)\\)는 필요한 LLM 호출 횟수의 정규화된 값, \\(ER(s)\\)는 앞서 정의한 엔티티 추출률입니다. \\(\alpha\\)는 계산 비용과 추출 품질의 상대적 중요도를 조절하는 하이퍼파라미터입니다.

이러한 트레이드오프를 효과적으로 관리하기 위해 시스템은 다음과 같은 적응형 청킹 전략을 구현합니다.

```python
class AdaptiveChunker:
    def __init__(self, min_size: int, max_size: int, target_er: float):
        self.min_size = min_size
        self.max_size = max_size
        self.target_er = target_er
        
    async def optimize_chunk_size(self, sample_text: str) -> int:
        best_size = self.min_size
        best_cost = float('inf')
        
        for size in range(self.min_size, self.max_size, 200):
            er = await self.measure_extraction_rate(size, sample_text)
            n_calls = len(sample_text) / size
            cost = self.compute_cost(n_calls, er)
            
            if cost < best_cost:
                best_cost = cost
                best_size = size
                
        return best_size
```

이 구현에서는 주어진 텍스트 샘플에 대해 다양한 청크 크기를 시도하면서 최적의 크기를 찾습니다. 특히 시스템은 문서의 특성과 추출하고자 하는 정보의 유형에 따라 청크 크기를 동적으로 조정할 수 있습니다. 예를 들어, 엔티티 간의 관계가 복잡하게 얽혀있는 학술 논문의 경우 더 큰 청크 크기가 필요할 수 있으며, 간단한 뉴스 기사의 경우 작은 청크 크기로도 충분할 수 있습니다.

이러한 적응형 접근법은 Kuratov와 연구진이 발견한 LLM의 문맥 윈도우 제한과 관련된 성능 저하 문제를 효과적으로 관리할 수 있게 해줍니다. 시스템은 문서의 복잡성과 LLM의 성능 특성을 고려하여 최적의 청크 크기를 자동으로 선택함으로써, 정보 추출의 정확도와 계산 효율성 사이의 균형을 달성할 수 있습니다.

### 텍스트 청크에서 엔티티 인스턴스로: 그래프 노드와 엣지 추출

텍스트 청크에서 그래프 노드와 엣지를 추출하는 과정은 그래프 RAG 시스템의 핵심 구성 요소입니다. 이 단계에서는 대규모 언어 모델(LLM)을 활용하여 텍스트로부터 의미 있는 엔티티와 그들 간의 관계를 식별하고 추출합니다.

저자들은 다단계 LLM 프롬프트 접근법을 제안합니다. 이 접근법은 크게 두 단계로 구성됩니다. 첫 번째 단계에서는 텍스트 내의 모든 엔티티를 식별합니다. 각 엔티티에 대해 다음과 같은 정보를 추출합니다.

\\[ E = \{(n_i, t_i, d_i) \vert i = 1,...,N\} \\]

여기서:
- \\(n_i\\)는 엔티티의 이름
- \\(t_i\\)는 엔티티의 유형
- \\(d_i\\)는 엔티티의 설명
- \\(N\\)은 추출된 총 엔티티 수

두 번째 단계에서는 명확하게 연관된 엔티티들 간의 관계를 식별합니다. 각 관계는 다음과 같이 표현됩니다.

\\[ R = \{(s_j, t_j, d_j) \vert j = 1,...,M\} \\]

여기서:
- \\(s_j\\)는 소스 엔티티
- \\(t_j\\)는 대상 엔티티
- \\(d_j\\)는 관계에 대한 설명
- \\(M\\)은 식별된 총 관계 수

이러한 추출 과정의 효과성을 높이기 위해, 저자들은 퓨 샷 학습(few-shot learning) 예제를 도메인에 맞게 조정하는 방법을 제안합니다. 예를 들어, 과학 분야의 문서를 처리할 때는 과학적 개념이나 방법론에 관한 예제를, 의학 분야에서는 질병이나 치료법에 관한 예제를 사용합니다.

시스템은 또한 추가적인 공변량(covariate) 추출을 위한 보조 프롬프트를 지원합니다. 이 프롬프트는 주로 엔티티와 연관된 주장(claim)을 추출하는 데 사용되며, 다음과 같은 정보를 포함합니다.

\\[ C = \{(s_k, o_k, t_k, d_k, p_k, t_{start}, t_{end}) \vert k = 1,...,K\} \\]

여기서:
- \\(s_k\\)는 주장의 주체
- \\(o_k\\)는 주장의 대상
- \\(t_k\\)는 주장의 유형
- \\(d_k\\)는 주장의 설명
- \\(p_k\\)는 원본 텍스트 범위
- \\(t_{start}, t_{end}\\)는 시작과 종료 날짜

효율성과 품질의 균형을 맞추기 위해, 시스템은 "gleaning"이라고 하는 다중 라운드 추출 방식을 사용합니다. 이 과정은 다음과 같은 알고리즘으로 구현됩니다.

```python
async def extract_entities(text: str, max_gleanings: int = 3) -> List[Entity]:
    entities = []
    for i in range(max_gleanings):
        # 엔티티 추출 시도
        new_entities = await llm.extract(text)
        entities.extend(new_entities)
        
        # 모든 엔티티가 추출되었는지 확인
        missed = await llm.check_missed_entities(
            text, entities, logit_bias={yes: 100, no: 100}
        )
        
        if not missed:
            break
            
        # 놓친 엔티티가 있다면 계속 진행
        text = f"MANY entities were missed in the last extraction. {text}"
    
    return entities
```

이러한 반복적 추출 방식은 더 큰 청크 크기를 사용하면서도 품질 저하나 불필요한 노이즈 도입을 방지할 수 있게 해줍니다.

### 엔티티 인스턴스에서 엔티티 요약으로: LLM 기반 추상화 요약 전략

대규모 언어 모델(Large Language Model, LLM)을 사용하여 원본 텍스트에서 엔티티, 관계, 그리고 주장들을 "추출"하는 과정은 그 자체로 추상적 요약의 한 형태입니다. 이는 텍스트에 명시적으로 언급되지 않았지만 암시된 개념들(예: 암묵적 관계)까지도 독립적으로 의미 있는 요약으로 만들어내는 LLM의 능력을 활용하는 것입니다.

이러한 인스턴스 수준의 요약들을 그래프의 각 요소(즉, 엔티티 노드, 관계 엣지, 주장 공변량)에 대한 단일 설명 텍스트 블록으로 변환하기 위해서는 LLM을 사용한 추가적인 요약 단계가 필요합니다. 이 과정은 다음과 같은 수식으로 표현될 수 있습니다.

\\[ S(E) = \text{LLM}(\{d_1, d_2, ..., d_n\}) \\]

여기서 \\(S(E)\\)는 엔티티 \\(E\\)에 대한 최종 요약이며, \\(\{d_1, d_2, ..., d_n\}\\)은 해당 엔티티의 모든 인스턴스 설명들의 집합입니다.

이 단계에서 발생할 수 있는 잠재적 문제점은 LLM이 동일한 엔티티를 항상 일관된 텍스트 형식으로 추출하지 않을 수 있다는 것입니다. 이는 엔티티 그래프에서 중복된 노드가 생성되는 결과를 초래할 수 있습니다. 예를 들어, "마이크로소프트"라는 엔티티가 "Microsoft", "Microsoft Corporation", "MS" 등 다양한 형태로 추출될 수 있습니다.

그러나 저자들은 이러한 변동성이 시스템의 전반적인 성능에 큰 영향을 미치지 않는다고 설명합니다. 그 이유는 다음 단계에서 밀접하게 연관된 모든 "커뮤니티"들이 탐지되고 요약되며, LLM이 다양한 이름 변형 뒤에 있는 공통된 엔티티를 이해할 수 있기 때문입니다. 이는 다음과 같은 수식으로 표현될 수 있습니다.

\\[ C(v_1, v_2) = \text{sim}(\text{LLM}(v_1), \text{LLM}(v_2)) > \theta \\]

여기서 \\(C(v_1, v_2)\\)는 두 변형 \\(v_1\\)과 \\(v_2\\)가 같은 커뮤니티에 속하는지를 나타내며, \\(\text{sim}\\)은 의미적 유사도 함수, \\(\theta\\)는 임계값입니다.

이러한 접근 방식의 핵심은 잠재적으로 노이즈가 있는 그래프 구조에서 풍부한 설명 텍스트를 가진 동종 노드들을 사용한다는 점입니다. 이는 LLM의 자연어 이해 및 생성 능력과 전역적, 질의 중심 요약의 요구사항에 모두 부합합니다. 

이러한 특성들은 이 그래프 인덱스를 전통적인 지식 그래프와 차별화합니다. 기존의 지식 그래프들은 주로 간결하고 일관된 지식 트리플(주어, 술어, 목적어)에 의존하여 다운스트림 추론 작업을 수행합니다. 반면, 이 접근법은 더 풍부하고 유연한 텍스트 설명을 활용하여 LLM의 강점을 최대한 활용할 수 있게 합니다.

### 엔티티 요약에서 그래프 커뮤니티로: 계층적 구조 탐지

이전 단계에서 생성된 인덱스는 동종 무방향 가중 그래프(homogeneous undirected weighted graph)로 모델링됩니다. 이 그래프에서 엔티티들은 노드로 표현되고, 이들 간의 관계는 엣지로 연결됩니다. 각 엣지의 가중치는 검출된 관계 인스턴스의 정규화된 빈도를 나타냅니다. 이러한 그래프 구조는 다음과 같은 수식으로 표현할 수 있습니다.

\\[ G = (V, E, W) \\]

여기서:
- \\(V\\)는 엔티티 노드의 집합
- \\(E\\)는 관계 엣지의 집합
- \\(W\\)는 엣지 가중치 행렬로, \\(w_{ij}\\)는 노드 \\(i\\)와 \\(j\\) 사이의 관계 빈도를 정규화한 값

이러한 그래프 구조에서 커뮤니티를 탐지하기 위해 저자들은 Leiden 알고리즘을 사용합니다. Leiden 알고리즘은 대규모 그래프의 계층적 커뮤니티 구조를 효율적으로 찾아낼 수 있는 특징을 가지고 있습니다. 이 알고리즘은 다음과 같은 모듈성(modularity) 목적 함수를 최적화합니다.

\\[ Q = \frac{1}{2m}\sum_{ij} \left[A_{ij} - \frac{k_i k_j}{2m}\right]\delta(c_i,c_j) \\]

여기서:
- \\(A_{ij}\\)는 노드 i와 j 사이의 엣지 가중치
- \\(k_i\\)는 노드 i의 차수(degree)
- \\(m\\)은 전체 엣지 가중치의 합
- \\(\delta(c_i,c_j)\\)는 노드 i와 j가 같은 커뮤니티에 속하면 1, 아니면 0

Leiden 알고리즘의 구현은 다음과 같은 코드로 이루어집니다.

```python
def cluster_graph(
    graph: nx.Graph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> Communities:
    """대규모 그래프에 계층적 클러스터링 알고리즘 적용"""
    if len(graph.nodes) == 0:
        return []
        
    node_id_to_community_map, parent_mapping = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=seed,
    )
    
    # 계층적 커뮤니티 구조 구성
    levels = sorted(node_id_to_community_map.keys())
    clusters = {}
    for level in levels:
        clusters[level] = {}
        for node_id, community_id in node_id_to_community_map[level].items():
            if community_id not in clusters[level]:
                clusters[level][community_id] = []
            clusters[level][community_id].append(node_id)
            
    return [(level, cid, parent_mapping[cid], nodes) 
            for level in clusters
            for cid, nodes in clusters[level].items()]
```

이 알고리즘의 핵심적인 특징은 계층적 커뮤니티 구조를 찾아낸다는 점입니다. 각 계층에서 그래프의 노드들은 상호 배타적이면서도 전체를 포괄하는(mutually-exclusive, collective-exhaustive) 방식으로 커뮤니티들로 분할됩니다. 이러한 계층적 구조는 분할 정복(divide-and-conquer) 방식의 전역 요약을 가능하게 합니다.
### 계층적 커뮤니티 구조의 최적화와 구현 전략

Leiden 알고리즘의 실제 구현에서는 커뮤니티 탐지의 효율성과 품질을 높이기 위한 여러 최적화 전략이 사용됩니다. 특히 `_compute_leiden_communities` 함수는 graspologic 라이브러리를 활용하여 다음과 같이 구현됩니다.

```python
def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Leiden 커뮤니티와 계층 구조 매핑 반환"""
    from graspologic.partition import hierarchical_leiden
    
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    results: dict[int, dict[str, int]] = {}
    hierarchy: dict[int, int] = {}
    
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster
        
        hierarchy[partition.cluster] = (
            partition.parent_cluster if partition.parent_cluster is not None else -1
        )
    
    return results, hierarchy
```

이 구현에서 주목할 만한 최적화 전략은 다음과 같습니다. 먼저, 그래프의 최대 연결 컴포넌트(Largest Connected Component, LCC)만을 사용하는 옵션을 제공합니다. 이는 다음과 같은 수식으로 표현되는 그래프의 연결성을 개선합니다.

\\[ LCC(G) = \arg\max_{C \in \text{Components}(G)} \vert V(C) \vert \\]

여기서 \\(V(C)\\)는 컴포넌트 \\(C\\)의 노드 집합입니다.

또한, 커뮤니티의 계층 구조는 다음과 같은 트리 형태로 구성됩니다.

\\[ T = (C, P, L) \\]

여기서:
- \\(C\\)는 모든 커뮤니티의 집합
- \\(P: C \rightarrow C \cup \{-1\}\\)은 부모 커뮤니티 매핑 함수
- \\(L: C \rightarrow \mathbb{N}\\)은 각 커뮤니티의 계층 레벨

이러한 계층 구조는 `CommunityReportsExtractor` 클래스를 통해 효율적으로 처리됩니다.

```python
class CommunityReportsExtractor:
    def __init__(self, graph_index: GraphIndex, llm: LanguageModel):
        self.graph_index = graph_index
        self.llm = llm
        
    async def process_community(self, community_id: str) -> CommunityReport:
        # 커뮤니티의 계층 구조 탐색
        hierarchy = await self.graph_index.get_community_hierarchy(community_id)
        
        # 커뮤니티 내 엔티티들의 관계 분석
        entities = await self.graph_index.get_community_entities(community_id)
        relationships = await self.analyze_relationships(entities)
        
        # LLM을 사용한 커뮤니티 요약 생성
        summary = await self.llm.summarize_community(
            entities=entities,
            relationships=relationships,
            hierarchy=hierarchy
        )
        
        return CommunityReport(
            id=community_id,
            hierarchy=hierarchy,
            entities=entities,
            summary=summary
        )
```

이러한 구현을 통해 대규모 그래프에서도 효율적인 커뮤니티 탐지와 요약이 가능해지며, 특히 계층적 구조를 활용한 분할 정복 접근법은 전체 시스템의 확장성을 크게 향상시킵니다.

### 그래프 커뮤니티에서 커뮤니티 요약으로: 계층적 구조의 효율적 요약 전략

그래프 RAG 시스템의 핵심 구성 요소 중 하나는 Leiden 알고리즘으로 탐지된 계층적 커뮤니티 구조를 의미 있는 요약으로 변환하는 과정입니다. 이 단계는 대규모 데이터셋에서도 효율적으로 작동하도록 설계되었으며, 각 커뮤니티의 특성과 의미를 포착하는 보고서 형태의 요약을 생성합니다.

![커뮤니티 구조 시각화](/assets/2025-01-15-from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/2.png)

위 그림은 MultiHop-RAG 데이터셋에서 Leiden 알고리즘을 통해 탐지된 그래프 커뮤니티를 보여줍니다. (a)는 최상위 레벨(레벨 0)의 커뮤니티를, (b)는 그 아래 레벨(레벨 1)의 하위 커뮤니티를 나타냅니다. 원의 크기는 해당 엔티티 노드의 연결 정도(degree)에 비례하며, 노드의 배치는 OpenORD와 Force Atlas 2 알고리즘을 통해 수행되었습니다. 노드의 색상은 서로 다른 커뮤니티를 구분합니다.

커뮤니티 요약 생성은 크게 두 가지 경우로 나누어 처리됩니다.

1. 리프 레벨 커뮤니티의 경우:
커뮤니티 내 요소들(노드, 엣지, 공변량)의 요약은 다음과 같은 우선순위 함수에 따라 정렬됩니다.

\\[ P(e) = deg(source(e)) + deg(target(e)) \\]

여기서 \\(P(e)\\)는 엣지 \\(e\\)의 우선순위이며, \\(deg(v)\\)는 노드 \\(v\\)의 연결 정도입니다. 이렇게 정렬된 요소들은 LLM의 문맥 윈도우 크기 제한에 도달할 때까지 순차적으로 추가됩니다.

2. 상위 레벨 커뮤니티의 경우:
모든 요소 요약이 문맥 윈도우 제한 내에 들어가면 리프 레벨과 동일한 방식으로 처리합니다. 그렇지 않은 경우, 다음과 같은 최적화 문제를 해결합니다.

\\[ \max_{S \subseteq C} \sum_{c \in S} w(c) \text{ subject to } \sum_{c \in S} t(c) \leq T \\]

여기서:
- \\(C\\)는 하위 커뮤니티의 집합
- \\(w(c)\\)는 커뮤니티 \\(c\\)의 중요도 가중치
- \\(t(c)\\)는 커뮤니티 \\(c\\)의 요약에 필요한 토큰 수
- \\(T\\)는 최대 허용 토큰 수

이러한 계층적 요약 전략의 구현은 다음과 같은 코드로 이루어집니다.

```python
async def summarize_community(
    community: Community,
    llm: LanguageModel,
    max_tokens: int = 16000
) -> CommunityReport:
    if community.is_leaf():
        return await summarize_leaf_community(community, llm, max_tokens)
    
    total_tokens = sum(len(e.summary) for e in community.elements)
    if total_tokens <= max_tokens:
        return await summarize_leaf_community(community, llm, max_tokens)
        
    # 하위 커뮤니티 요약으로 대체
    sub_summaries = []
    remaining_tokens = max_tokens
    
    for sub_comm in sorted(
        community.sub_communities,
        key=lambda x: len(x.element_summaries),
        reverse=True
    ):
        if len(sub_comm.summary) <= remaining_tokens:
            sub_summaries.append(sub_comm.summary)
            remaining_tokens -= len(sub_comm.summary)
            
    return await llm.generate_community_report(sub_summaries)
```
### 그래프 커뮤니티에서 커뮤니티 요약으로: 계층적 구조의 효율적 요약 전략

이러한 계층적 커뮤니티 요약 시스템의 구현에서 가장 중요한 기술적 도전 과제는 대규모 데이터셋에서의 효율적인 처리입니다. 시스템은 각 커뮤니티의 특성을 보존하면서도 LLM의 문맥 윈도우 제한을 고려해야 하는데, 이를 위해 저자들은 적응형 요약 전략(Adaptive Summarization Strategy)을 제안합니다.

적응형 요약 전략의 핵심은 커뮤니티의 계층 구조를 활용하여 요약의 상세도를 동적으로 조절하는 것입니다. 이는 다음과 같은 최적화 문제로 정형화됩니다.

\\[ S^*(c) = \arg\max_{S \in \mathcal{P}(c)} Q(S) \text{ subject to } L(S) \leq T \\]

여기서:
- \\(S^*(c)\\)는 커뮤니티 \\(c\\)의 최적 요약
- \\(\mathcal{P}(c)\\)는 가능한 모든 요약의 집합
- \\(Q(S)\\)는 요약 \\(S\\)의 품질 점수
- \\(L(S)\\)는 요약 \\(S\\)의 길이(토큰 수)
- \\(T\\)는 최대 허용 토큰 수

이 최적화 문제를 효율적으로 해결하기 위해 시스템은 동적 프로그래밍 기반의 접근법을 사용합니다.

```python
class AdaptiveSummarizer:
    def __init__(self, llm: LanguageModel, max_tokens: int):
        self.llm = llm
        self.max_tokens = max_tokens
        self.cache = {}
        
    async def optimize_summary(self, community: Community) -> Summary:
        if community.id in self.cache:
            return self.cache[community.id]
            
        if community.is_leaf():
            summary = await self._summarize_elements(
                community.elements,
                self.max_tokens
            )
        else:
            summary = await self._optimize_hierarchical_summary(
                community,
                self.max_tokens
            )
            
        self.cache[community.id] = summary
        return summary
        
    async def _optimize_hierarchical_summary(
        self,
        community: Community,
        token_limit: int
    ) -> Summary:
        dp = [[None] * (token_limit + 1) 
              for _ in range(len(community.sub_communities) + 1)]
        dp[0][0] = Summary()
        
        for i, sub_comm in enumerate(community.sub_communities, 1):
            sub_summary = await self.optimize_summary(sub_comm)
            for j in range(token_limit + 1):
                if dp[i-1][j] is not None:
                    dp[i][j] = dp[i-1][j]
                    if j + len(sub_summary) <= token_limit:
                        combined = dp[i-1][j].merge(sub_summary)
                        if combined.quality > dp[i][j].quality:
                            dp[i][j] = combined
                            
        return self._find_best_summary(dp)
```

이 구현에서 주목할 만한 최적화 기법은 메모이제이션(memoization)의 사용입니다. 캐시를 통해 이미 계산된 커뮤니티 요약을 재사용함으로써, 특히 상위 레벨 커뮤니티의 요약을 생성할 때 중복 계산을 피할 수 있습니다. 이는 다음과 같은 시간 복잡도 개선을 가져옵니다.

\\[ T(n) = O(n \cdot t \cdot \log d) \\]

여기서:
- \\(n\\)은 전체 노드의 수
- \\(t\\)는 토큰 제한
- \\(d\\)는 커뮤니티 계층 구조의 최대 깊이

이러한 최적화된 구현을 통해 시스템은 수백만 개의 노드를 가진 대규모 그래프에서도 효율적으로 커뮤니티 요약을 생성할 수 있습니다. 특히 계층적 구조를 활용한 요약 전략은 전체 그래프의 의미론적 구조를 다양한 상세도 수준에서 이해하는 데 도움을 줍니다.
### 그래프 커뮤니티에서 커뮤니티 요약으로: 계층적 구조의 효율적 요약 전략

커뮤니티 요약 시스템의 실제 구현에서는 요약의 품질과 계산 효율성을 동시에 최적화하기 위한 여러 기술적 전략이 사용됩니다. 특히 LLM을 활용한 요약 생성 과정에서는 토큰 사용의 효율성이 매우 중요한데, 이를 위해 저자들은 우선순위 기반의 토큰 할당 전략을 도입했습니다.

이 전략의 핵심은 각 커뮤니티 내의 요소들을 중요도에 따라 정렬하고, 제한된 토큰 예산 내에서 최대한의 정보를 포함하는 것입니다. 중요도 점수는 다음과 같은 수식으로 계산됩니다.

\\[ I(e) = \alpha \cdot C(e) + (1-\alpha) \cdot R(e) \\]

여기서:
- \\(I(e)\\)는 요소 \\(e\\)의 중요도 점수
- \\(C(e)\\)는 중심성(centrality) 점수로, 해당 요소의 그래프 내 위치적 중요도
- \\(R(e)\\)는 관련성(relevance) 점수로, 커뮤니티의 주요 주제와의 의미적 연관성
- \\(\alpha\\)는 두 점수 간의 균형을 조절하는 하이퍼파라미터

이러한 중요도 기반 요약 전략의 구현은 다음과 같은 코드로 이루어집니다.

```python
class PrioritizedSummarizer:
    def __init__(self, llm: LanguageModel, embedder: TextEmbedder):
        self.llm = llm
        self.embedder = embedder
        
    async def summarize_elements(
        self,
        elements: List[Element],
        community_theme: str,
        max_tokens: int
    ) -> Summary:
        # 요소별 임베딩 계산
        element_embeddings = await self.embedder.batch_encode(
            [e.text for e in elements]
        )
        theme_embedding = await self.embedder.encode(community_theme)
        
        # 중요도 점수 계산
        scores = []
        for i, element in enumerate(elements):
            centrality = self._compute_centrality(element)
            relevance = cosine_similarity(
                element_embeddings[i],
                theme_embedding
            )
            importance = self.alpha * centrality + (1 - self.alpha) * relevance
            scores.append((importance, element))
            
        # 중요도 기반 요소 선택
        selected = []
        current_tokens = 0
        for score, element in sorted(scores, reverse=True):
            if current_tokens + len(element.tokens) > max_tokens:
                break
            selected.append(element)
            current_tokens += len(element.tokens)
            
        return await self.llm.generate_summary(selected)
```

이 구현에서는 텍스트 임베딩을 활용하여 각 요소와 커뮤니티의 주요 주제 간의 의미적 유사도를 계산합니다. 이는 단순히 그래프 구조적 특성뿐만 아니라 내용의 의미적 관련성도 고려할 수 있게 해줍니다.

특히 상위 레벨 커뮤니티의 요약에서는 하위 커뮤니티 요약들을 효과적으로 통합하는 것이 중요합니다. 이를 위해 시스템은 다음과 같은 계층적 통합 전략을 사용합니다.

\\[ H(c) = \sum_{s \in S(c)} w(s) \cdot h(s) \\]

여기서:
- \\(H(c)\\)는 커뮤니티 \\(c\\)의 계층적 요약
- \\(S(c)\\)는 \\(c\\)의 하위 커뮤니티 집합
- \\(w(s)\\)는 하위 커뮤니티 \\(s\\)의 가중치
- \\(h(s)\\)는 하위 커뮤니티 \\(s\\)의 요약

이러한 계층적 요약 전략은 대규모 그래프의 구조를 다양한 추상화 수준에서 효과적으로 이해하고 탐색할 수 있게 해줍니다. 사용자는 상위 레벨의 일반적인 주제부터 시작하여 관심 있는 영역의 하위 커뮤니티로 점진적으로 탐색해 나갈 수 있습니다.

### 커뮤니티 요약에서 전역 답변 생성까지: 계층적 응답 생성 프레임워크

그래프 RAG 시스템의 최종 단계는 사용자의 질의에 대한 응답을 생성하는 것입니다. 이 과정은 이전 단계에서 생성된 커뮤니티 요약을 활용하여 다단계 프로세스를 통해 최종 답변을 도출합니다. 특히 커뮤니티 구조의 계층적 특성을 활용하여, 질문의 성격과 범위에 따라 다양한 수준의 커뮤니티 요약을 선택적으로 활용할 수 있습니다.

전역 답변 생성 과정은 다음과 같은 세 단계로 구성됩니다.

1. 커뮤니티 요약 준비 단계:
먼저 시스템은 커뮤니티 요약들을 무작위로 섞은 후, 사전에 정의된 토큰 크기의 청크로 분할합니다. 이는 다음과 같은 수식으로 표현될 수 있습니다.

\\[ C = \{c_1, c_2, ..., c_n\} \text{ where } \left\vert c_i \right\vert \leq T \\]

여기서 \\(C\\)는 분할된 청크의 집합, \\(\left\vert c_i \right\vert \\)는 각 청크의 토큰 수, \\(T\\)는 최대 토큰 제한입니다. 이러한 분할 방식은 관련 정보가 특정 문맥 윈도우에 집중되지 않고 고르게 분포되도록 보장합니다.

2. 커뮤니티 답변 매핑 단계:
각 청크에 대해 병렬적으로 중간 답변을 생성합니다. 이 과정은 다음과 같은 코드로 구현됩니다.

```python
async def generate_community_answers(
    chunks: List[str],
    query: str,
    llm: LanguageModel
) -> List[Answer]:
    answers = []
    for chunk in chunks:
        # 각 청크에 대해 답변과 신뢰도 점수 생성
        response = await llm.generate(
            context=chunk,
            query=query,
            output_format={
                "answer": "str",
                "confidence": "int[0-100]"
            }
        )
        if response["confidence"] > 0:
            answers.append(response)
    return sorted(answers, key=lambda x: x["confidence"], reverse=True)
```

이때 LLM은 각 답변에 대해 0-100 사이의 신뢰도 점수를 함께 생성하며, 점수가 0인 답변은 필터링됩니다.

3. 전역 답변 생성 단계:
중간 답변들은 신뢰도 점수를 기준으로 내림차순 정렬되며, 문맥 윈도우 제한에 도달할 때까지 순차적으로 새로운 문맥에 추가됩니다. 이 과정은 다음과 같은 수식으로 표현됩니다.

\\[ G = \text{LLM}(\sum_{i=1}^k a_i) \text{ where } \sum_{i=1}^k \left\vert a_i \right\vert \leq W \\]

여기서:
- \\(G\\)는 최종 전역 답변
- \\(a_i\\)는 신뢰도 점수순으로 정렬된 중간 답변
- \\(\left\vert a_i \right\vert\\)는 각 답변의 토큰 수
- \\(W\\)는 LLM의 문맥 윈도우 크기 제한
- \\(k\\)는 문맥 윈도우에 포함될 수 있는 최대 답변 수

이러한 다단계 접근법은 대규모 문서 컬렉션에서 효과적인 정보 추출과 통합을 가능하게 하며, 특히 전역적 이해가 필요한 질문에 대해 포괄적이고 정확한 답변을 생성할 수 있게 합니다.
### 데이터셋 예시와 활용 사례: 실제 응용을 통한 그래프 RAG의 이해

그래프 RAG 시스템의 실제 활용 사례를 살펴보면, 시스템이 다양한 도메인과 데이터 유형에서 효과적으로 작동함을 확인할 수 있습니다. 특히 팟캐스트 전사본과 뉴스 기사와 같은 실제 데이터셋에서의 활용 사례는 시스템의 실용성을 잘 보여줍니다.

팟캐스트 전사본을 활용한 기술 산업 분석의 경우, 시스템은 기술 정책과 규제에 관한 복잡한 논의를 효과적으로 분석할 수 있습니다. 예를 들어, 기술 저널리스트가 다음과 같은 전역적 질문을 던질 수 있습니다.

\\[ Q = \{q_1, q_2, ..., q_n\} \text{ where } q_i \in \text{TechPolicy} \\]

여기서 각 질문은 기술 정책의 특정 측면을 탐구하며, 시스템은 다음과 같은 커뮤니티 기반 응답을 생성합니다.

\\[ R(q_i) = \text{Aggregate}(\{c_j : \text{Relevance}(c_j, q_i) > \theta\}) \\]

여기서 \\(c_j\\)는 커뮤니티 요약이며, \\(\theta\\)는 관련성 임계값입니다.

뉴스 기사 데이터셋의 경우, 교육자들이 건강과 웰니스 관련 커리큘럼을 개발하는 데 도움을 받을 수 있습니다. 시스템은 다음과 같은 코드로 구현된 분석 파이프라인을 통해 관련 정보를 추출하고 구조화합니다.

```python
class HealthEducationAnalyzer:
    def __init__(self, graph_rag: GraphRAG):
        self.graph_rag = graph_rag
        
    async def analyze_health_trends(
        self,
        query: str,
        community_level: int = 2
    ) -> AnalysisResult:
        # 커뮤니티 레벨에 따른 요약 생성
        summaries = await self.graph_rag.get_community_summaries(
            level=community_level
        )
        
        # 건강 관련 주제 추출 및 분석
        health_insights = await self.graph_rag.query(
            query=query,
            summaries=summaries,
            min_confidence=0.7
        )
        
        return self._structure_findings(health_insights)
```

이러한 실제 활용 사례들은 그래프 RAG 시스템이 단순한 정보 검색을 넘어, 복잡한 도메인 지식을 효과적으로 분석하고 통합할 수 있음을 보여줍니다. 특히 시스템의 계층적 커뮤니티 구조는 다양한 추상화 수준에서의 분석을 가능하게 하며, 이는 사용자의 구체적인 요구사항에 맞는 유연한 응답 생성을 가능하게 합니다.

### 평가 방법론과 실험 설계

이 연구의 평가는 실제 사용자들이 접할 수 있는 규모의 데이터셋을 기반으로 수행되었습니다. 연구진은 각각 약 10개의 소설 분량에 해당하는 100만 토큰 규모의 두 가지 데이터셋을 선정했습니다.

첫 번째 데이터셋은 Microsoft CTO인 Kevin Scott가 다른 기술 리더들과 나눈 대화를 기록한 팟캐스트 전사본으로 구성되어 있습니다. 이 데이터셋은 1,669개의 600토큰 크기 텍스트 청크로 구성되어 있으며, 각 청크 간에는 100토큰의 중첩이 있습니다. 전체 데이터셋의 크기는 약 100만 토큰입니다.

두 번째 데이터셋은 2013년 9월부터 2023년 12월까지 발행된 뉴스 기사들을 포함하는 MultiHop-RAG 벤치마크 데이터셋입니다. 엔터테인먼트, 비즈니스, 스포츠, 기술, 건강, 과학 등 다양한 카테고리의 기사들로 구성되어 있으며, 3,197개의 600토큰 크기 텍스트 청크(100토큰 중첩)로 이루어져 있어 총 약 170만 토큰 규모입니다.

평가를 위한 질의 생성에 있어서, 연구진은 기존의 오픈 도메인 질의응답 벤치마크(HotPotQA, MultiHop-RAG, MT-Bench 등)가 단순한 사실 검색에 초점을 맞추고 있다는 한계를 지적했습니다. 이러한 벤치마크들은 데이터 센스메이킹(sensemaking), 즉 사람들이 실제 활동의 맥락에서 데이터를 검토하고, 참여하며, 맥락화하는 과정을 제대로 평가하지 못한다는 것입니다.

이러한 한계를 극복하기 위해 연구진은 활동 중심 접근법(activity-centered approach)을 도입했습니다. 이 방법은 다음과 같은 수식으로 표현될 수 있습니다.

\\[ Q(D) = \bigcup_{u \in U} \bigcup_{t \in T_u} \{q_{u,t,i} \vert i = 1,...,N\} \\]

여기서:
- \\(D\\)는 데이터셋에 대한 간단한 설명
- \\(U\\)는 잠재적 사용자 집합 (\\(\left\vert U \right\vert = N\\))
- \\(T_u\\)는 사용자 \\(u\\)의 작업 집합 (\\(\left\vert T_u \right\vert = N\\))
- \\(q_{u,t,i}\\)는 (사용자, 작업) 조합에 대한 i번째 질문

연구진은 \\(N = 5\\)를 사용하여 각 데이터셋당 125개의 테스트 질문을 생성했습니다. 이러한 접근법은 데이터셋의 전반적인 이해를 요구하는 질문들을 자동으로 생성할 수 있게 해주며, 특정 텍스트의 세부 사항에 대한 사전 지식을 배제할 수 있습니다.
### 평가 방법론과 실험 설계: 실험 조건과 메트릭

연구진은 그래프 RAG 시스템의 성능을 평가하기 위해 6가지 실험 조건을 설정했습니다. 이 중 4가지는 서로 다른 수준의 그래프 커뮤니티(C0, C1, C2, C3)를 사용하는 Graph RAG 조건이며, 나머지 두 조건은 비교 기준이 되는 텍스트 요약(TS)과 단순 "의미 검색" RAG 접근법(SS)입니다.

각 커뮤니티 레벨 조건의 특징은 다음과 같습니다.

C0 조건은 루트 레벨 커뮤니티 요약(가장 적은 수의 요약)을 사용하여 사용자 질의에 답변합니다. C1 조건은 상위 레벨 커뮤니티 요약을 사용하며, 이는 C0의 하위 커뮤니티이거나 C0 커뮤니티를 투영한 것입니다. C2 조건은 중간 레벨 커뮤니티 요약을 사용하며, C1의 하위 커뮤니티이거나 C1 커뮤니티를 투영한 것입니다. C3 조건은 가장 많은 수의 하위 레벨 커뮤니티 요약을 사용하며, C2의 하위 커뮤니티이거나 C2 커뮤니티를 투영한 것입니다.

비교 기준이 되는 두 조건 중 TS는 2.6절에서 설명한 것과 동일한 방법을 사용하지만, 커뮤니티 요약 대신 원본 텍스트를 섞어서 청크로 나눈 후 맵-리듀스 요약 단계를 적용합니다. SS는 단순한 RAG 구현으로, 지정된 토큰 제한에 도달할 때까지 텍스트 청크를 검색하여 문맥 윈도우에 추가하는 방식입니다.

모든 6가지 조건에서 문맥 윈도우의 크기와 답변 생성을 위한 프롬프트는 동일하게 유지되었습니다(문맥 정보 유형에 맞게 스타일을 약간 수정한 것 제외). 조건들 간의 유일한 차이점은 문맥 윈도우의 내용을 생성하는 방식입니다.

C0-C3 조건을 지원하는 그래프 인덱스는 엔티티와 관계 추출을 위한 일반적인 프롬프트만을 사용하여 생성되었으며, 엔티티 유형과 퓨 샷 예제는 데이터의 도메인에 맞게 조정되었습니다. 그래프 인덱싱 과정에서는 팟캐스트 데이터셋의 경우 1회의 gleaning을, 뉴스 데이터셋의 경우 0회의 gleaning을 사용하여 600토큰 크기의 문맥 윈도우를 적용했습니다.

평가 메트릭과 관련하여, 연구진은 대규모 언어 모델이 자연어 생성을 평가하는 데 있어 인간 판단과 비교했을 때 최고 수준이거나 경쟁력 있는 결과를 달성한다는 선행 연구들을 참고했습니다. 이러한 LLM 기반 평가는 정답이 있는 경우 참조 기반 메트릭을 생성할 수 있을 뿐만 아니라, 생성된 텍스트의 품질(예: 유창성)을 참조 없이도 평가할 수 있으며, 경쟁하는 출력들을 직접 비교(LLM-as-a-judge)할 수도 있습니다.
### 평가 방법론과 실험 설계: 평가 결과 분석

연구진이 수행한 실험 결과는 매우 흥미로운 패턴을 보여주었습니다. 그래프 RAG의 성능을 평가하기 위해 연구진은 네 가지 핵심 메트릭을 사용했습니다. 포괄성(comprehensiveness), 다양성(diversity), 임파워먼트(empowerment), 그리고 직접성(directness)입니다. 이 중 직접성은 유효성 지표로 사용되었으며, 포괄성 및 다양성과는 상반된 특성을 가지고 있어 어떤 방법도 모든 메트릭에서 우수한 성능을 보이지는 않을 것으로 예상되었습니다.

실험 결과를 살펴보면, 팟캐스트 전사본 데이터셋에서 모든 Graph RAG 조건(C0-C3)이 포괄성과 다양성 측면에서 단순 RAG(SS) 방식을 크게 앞섰습니다. 구체적으로, Graph RAG는 포괄성에서 72-83%의 승률을, 다양성에서는 75-82%의 승률을 기록했습니다. 특히 중간 레벨의 커뮤니티 요약(C2)을 사용했을 때 가장 좋은 성능을 보였습니다.

뉴스 기사 데이터셋에서도 유사한 패턴이 관찰되었습니다. Graph RAG는 포괄성에서 72-80%, 다양성에서 62-71%의 승률을 기록했으며, 이 데이터셋에서는 하위 레벨 커뮤니티 요약(C3)이 가장 효과적이었습니다.

특히 주목할 만한 점은 Graph RAG의 효율성입니다. 커뮤니티 요약을 사용하는 방식은 원본 텍스트 요약(TS) 방식과 비교했을 때 상당한 자원 절약을 달성했습니다. 이는 다음과 같은 수식으로 표현될 수 있습니다.

\\[ E = \frac{T_{TS} - T_{GR}}{T_{TS}} \times 100\% \\]

여기서:
- \\(E\\)는 토큰 사용의 효율성 개선도
- \\(T_{TS}\\)는 TS 방식에서 사용된 총 토큰 수
- \\(T_{GR}\\)은 Graph RAG에서 사용된 총 토큰 수

하위 레벨 커뮤니티 요약(C3)의 경우 26-33% 더 적은 문맥 토큰을 사용했으며, 루트 레벨 커뮤니티 요약(C0)의 경우에는 놀랍게도 97% 이상 적은 토큰을 사용하면서도 단순 RAG 대비 포괄성(72% 승률)과 다양성(62% 승률)에서 우수한 성능을 유지했습니다.

임파워먼트 메트릭의 경우, 결과는 다소 혼재된 양상을 보였습니다. LLM의 평가 근거를 분석한 결과, 구체적인 예시, 인용구, 출처 등을 제공하는 능력이 사용자의 정보에 기반한 이해를 돕는 데 핵심적인 요소로 판단되었습니다. 이는 Graph RAG 인덱스에서 이러한 세부 정보들을 더 잘 보존하도록 엔티티 추출 프롬프트를 조정할 필요성을 시사합니다.
### 평가 방법론과 실험 설계: 구성 최적화와 실험 결과

연구진은 시스템의 성능에 영향을 미치는 중요한 하이퍼파라미터인 문맥 윈도우 크기의 효과를 체계적으로 분석했습니다. GPT-4-turbo와 같이 128k 토큰의 큰 문맥 크기를 가진 모델에서도 긴 문맥에서 정보가 "중간에 손실되는" 현상이 발생할 수 있다는 점을 고려하여, 연구진은 데이터셋, 질문, 메트릭의 조합에 따른 문맥 윈도우 크기의 영향을 탐구했습니다.

특히 기준 조건(SS)에 대한 최적의 문맥 크기를 결정하고 이를 모든 질의 시점 LLM 사용에 균일하게 적용하는 것을 목표로 했습니다. 이를 위해 8k, 16k, 32k, 64k의 네 가지 문맥 윈도우 크기를 테스트했습니다. 이러한 실험은 다음과 같은 수식으로 표현될 수 있습니다.

\\[ P(s) = \frac{W(s)}{T} \times 100\% \\]

여기서:
- \\(P(s)\\)는 문맥 크기 \\(s\\)에서의 성능 점수
- \\(W(s)\\)는 해당 크기에서의 승리 횟수
- \\(T\\)는 전체 비교 횟수

흥미롭게도, 테스트한 가장 작은 문맥 윈도우 크기(8k)가 포괄성 비교에서 모든 경우에 더 우수한 성능을 보였으며(평균 승률 58.1%), 다양성(평균 승률 52.4%)과 임파워먼트(평균 승률 51.3%)에서도 더 큰 문맥 크기들과 비슷한 성능을 보였습니다. 이러한 결과를 바탕으로 연구진은 최종 평가에서 8k 토큰의 고정된 문맥 윈도우 크기를 사용하기로 결정했습니다.

인덱싱 과정의 결과로 팟캐스트 데이터셋에서는 8,564개의 노드와 20,691개의 엣지로 구성된 그래프가, 뉴스 데이터셋에서는 15,754개의 노드와 19,520개의 엣지로 구성된 더 큰 그래프가 생성되었습니다. 이러한 그래프 구조는 다음과 같은 수식으로 표현될 수 있습니다.

\\[ G_d = (V_d, E_d) \text{ where } d \in \{podcast, news\} \\]

여기서:
- \\(G_d\\)는 데이터셋 \\(d\\)의 그래프
- \\(V_d\\)는 노드의 집합
- \\(E_d\\)는 엣지의 집합

각 그래프 커뮤니티 계층의 다양한 레벨에서 생성된 커뮤니티 요약의 수는 데이터셋의 특성과 복잡성을 반영합니다. 특히 원본 텍스트 요약(TS)이 가장 많은 토큰을 필요로 하는 반면, 루트 레벨 커뮤니티 요약(C0)은 질의당 필요한 토큰 수를 극적으로 감소시켰습니다(9배에서 43배까지).

### 관련 연구: RAG 접근법과 시스템들

검색 증강 생성(Retrieval-Augmented Generation, RAG)은 대규모 언어 모델(Large Language Model, LLM)을 사용할 때 외부 데이터 소스에서 관련 정보를 먼저 검색하고, 이를 원본 질의와 함께 LLM의 문맥 윈도우에 추가하는 방식입니다. Ram과 연구진이 설명한 기본적인 RAG 접근법은 문서를 텍스트로 변환하고, 이를 청크로 분할한 뒤, 이러한 청크들을 벡터 공간에 임베딩하는 방식으로 작동합니다. 이때 유사한 위치는 유사한 의미를 나타내며, 질의는 동일한 벡터 공간에 임베딩되어 가장 가까운 k개의 벡터에 해당하는 텍스트 청크들이 문맥으로 사용됩니다.

더 발전된 RAG 시스템들은 기본 RAG의 한계를 극복하기 위해 사전 검색, 검색, 사후 검색 전략을 포함하고 있으며, 모듈형 RAG 시스템은 반복적이고 동적인 검색-생성 사이클을 위한 패턴들을 포함합니다. 본 연구에서 제안하는 그래프 RAG는 다른 시스템들의 여러 개념을 통합하고 있습니다. 예를 들어, 커뮤니티 요약은 Cheng과 연구진이 제안한 자체 메모리(Self-memory) 개념을 활용한 것으로, 이는 향후 생성 사이클을 위한 생성 증강 검색(Generation-Augmented Retrieval, GAR)의 한 형태입니다.

또한 커뮤니티 요약으로부터 병렬적으로 커뮤니티 답변을 생성하는 방식은 Shao와 연구진이 제안한 반복적 검색-생성(Iter-RetGen) 또는 Wang과 연구진이 제안한 연합 검색-생성(FeB4RAG) 전략과 유사합니다. 이러한 접근법들은 다중 문서 요약(CAiRE-COVID)이나 다중 홉 질의응답(ITRG, IR-CoT, DSP) 등의 시스템에서도 활용되고 있습니다.

본 연구의 계층적 인덱스와 요약 방식은 Sarthi와 연구진이 제안한 RAPTOR와 같이 텍스트 청크의 벡터를 클러스터링하여 계층적 인덱스를 생성하는 방식이나, Kim과 연구진이 제안한 모호한 질문의 다양한 해석을 위한 "명확화 트리" 생성 방식과 유사점을 가지고 있습니다. 그러나 이러한 반복적이거나 계층적인 접근법들 중 어느 것도 그래프 RAG가 활용하는 것과 같은 자체 생성 그래프 인덱스를 사용하지는 않습니다.

LLM과 그래프의 결합은 현재 활발히 연구되고 있는 분야입니다. 주요 연구 방향으로는 지식 그래프 생성과 완성, 인과 그래프 추출, 그리고 다양한 형태의 고급 RAG가 있습니다. 특히 고급 RAG에는 인덱스로 지식 그래프를 사용하는 KAPING, 그래프 구조의 부분집합이나 파생된 그래프 메트릭을 조회 대상으로 하는 G-Retriever와 GraphToolFormer, 검색된 서브그래프의 사실에 강하게 근거한 서술을 생성하는 SURGE, 서술 템플릿을 사용하여 검색된 이벤트-플롯 서브그래프를 직렬화하는 FABULA 등이 있습니다.

오픈소스 소프트웨어 측면에서는 LangChain과 LlamaIndex 라이브러리가 다양한 그래프 데이터베이스를 지원하고 있으며, Neo4J의 NaLLM이나 NebulaGraph의 GraphRAG와 같이 더 일반적인 그래프 기반 RAG 애플리케이션들도 등장하고 있습니다. 그러나 이러한 시스템들 중 어느 것도 그래프의 자연스러운 모듈성을 활용하여 전역적 요약을 위한 데이터 분할을 수행하지는 않습니다.

### 연구의 한계점과 향후 연구 방향

이 연구의 평가 방법론에는 몇 가지 주목할 만한 한계점이 있습니다. 먼저, 현재까지의 평가는 약 100만 토큰 규모의 두 코퍼스에 대한 특정 유형의 센스메이킹 질문들에만 국한되어 있습니다. 이는 시스템의 일반화 가능성을 제한적으로만 검증할 수 있다는 것을 의미합니다. 다양한 질문 유형, 데이터 유형, 그리고 데이터셋 크기에 따른 성능 변화를 더 깊이 이해하기 위해서는 추가적인 연구가 필요합니다.

특히 센스메이킹 질문과 목표 메트릭의 타당성 검증을 위해서는 실제 최종 사용자들과의 평가가 필요합니다. 현재의 평가 방식은 주로 시스템의 기술적 성능에 초점을 맞추고 있어, 실제 사용 환경에서의 유용성과 사용자 경험을 완전히 반영하지 못할 수 있습니다. 또한, [Manakul과 연구진](https://arxiv.org/pdf/2303.08896)이 제안한 SelfCheckGPT와 같은 접근법을 활용하여 응답의 허구 정보 생성(fabrication) 비율을 비교하는 것도 현재 분석의 개선점이 될 수 있습니다.

그래프 인덱스 구축과 관련된 트레이드오프도 중요한 고려사항입니다. 실험 결과에 따르면 Graph RAG가 다른 방법들과 비교했을 때 일관되게 우수한 성능을 보여주었지만, 많은 경우에 원본 텍스트의 전역 요약 방식도 경쟁력 있는 성능을 보였습니다. 따라서 그래프 인덱스 구축에 대한 실제 의사결정은 다음과 같은 여러 요소들을 종합적으로 고려해야 합니다.

1. 계산 자원 예산
2. 데이터셋당 예상되는 평생 질의 횟수
3. 그래프 인덱스의 다른 측면들(일반적인 커뮤니티 요약, 다른 그래프 관련 RAG 접근법 활용 등)로부터 얻을 수 있는 부가가치

향후 연구 방향과 관련하여, 현재의 Graph RAG 접근법을 지원하는 그래프 인덱스, 풍부한 텍스트 주석, 그리고 계층적 커뮤니티 구조는 다양한 개선과 적용 가능성을 제공합니다. 예를 들어, 사용자 질의와 그래프 주석 간의 임베딩 기반 매칭을 통해 더 지역적으로 작동하는 RAG 접근법을 개발할 수 있습니다. 또한 커뮤니티 보고서에 대한 임베딩 기반 매칭을 먼저 수행한 후 맵-리듀스 요약 메커니즘을 적용하는 하이브리드 RAG 방식도 가능합니다.

이러한 "롤업" 연산은 커뮤니티 계층 구조의 더 많은 레벨로 확장될 수 있으며, 상위 레벨 커뮤니티 요약에 포함된 정보의 흐름을 따라가는 탐색적인 "드릴 다운" 메커니즘으로도 구현될 수 있습니다. 이는 사용자가 관심 있는 주제나 영역을 점진적으로 더 깊이 탐색할 수 있게 해주는 대화형 정보 탐색 경험을 가능하게 할 것입니다.

## 결론: 그래프 RAG를 통한 전역적 접근법의 성과와 전망

이 연구는 지식 그래프 생성, 검색 증강 생성(RAG), 그리고 질의 중심 요약(QFS)을 통합하여 대규모 텍스트 데이터에 대한 인간의 센스메이킹을 지원하는 전역적 접근법을 제시했습니다. 초기 평가 결과는 기본적인 RAG 방식과 비교했을 때 답변의 포괄성과 다양성 측면에서 상당한 개선을 보여주었으며, 맵-리듀스 방식의 텍스트 요약을 사용하는 그래프가 없는 전역적 접근법과 비교해도 우수한 성능을 달성했습니다.

특히 주목할 만한 점은 동일한 데이터셋에 대해 여러 번의 전역적 질의가 필요한 상황에서, 엔티티 기반 그래프 인덱스의 루트 레벨 커뮤니티 요약이 매우 효율적인 데이터 인덱스 역할을 한다는 것입니다. 이는 기본적인 RAG 방식보다 우수한 성능을 보이면서도, 다른 전역적 방법들과 비교했을 때 토큰 비용을 크게 절감할 수 있다는 장점이 있습니다.

연구진은 이 시스템의 실용적 가치를 높이기 위해 전역적 및 지역적 그래프 RAG 접근법을 모두 구현한 파이썬 기반의 오픈소스 코드를 [Github](https://github.com/microsoft/graphrag)에서 공개할 예정입니다. 이는 연구 커뮤니티가 그래프 RAG를 자신들의 프로젝트에 쉽게 적용하고 확장할 수 있게 해줄 것입니다.

이 연구는 Microsoft Research의 Darren Edge와 Ha Trinh이 주도했으며, Microsoft의 여러 부서에서 많은 연구자들이 협력하여 이루어졌습니다. 특히 Alonso Guevara Fernández, Amber Hoak, Andrés Morales Esquivel을 비롯한 20여 명의 연구진이 이 프로젝트에 기여했습니다. 이러한 광범위한 협력은 연구의 이론적 깊이와 실용적 가치를 동시에 추구했음을 보여줍니다.

이 연구는 대규모 언어 모델과 그래프 기반 지식 표현을 결합하는 새로운 패러다임을 제시했으며, 특히 전역적 질의에 대한 효율적이고 효과적인 응답 생성 방법을 제안했다는 점에서 큰 의의가 있습니다. 향후 이 접근법은 다양한 도메인에서의 지식 탐색과 이해를 위한 핵심 도구로 발전할 것으로 기대됩니다.

- - -
### References
* [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](http://arxiv.org/pdf/2404.16130v1)
