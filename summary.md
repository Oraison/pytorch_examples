```python
a = torch.rand(5,5)
b = a[:, 0::2]
c = a[:, 1::2]

# b = 짝수 index 추출
# c = 홀수 index 추출
# n::m = index % m == n 인 index의 집합


torch.triu(a)

# i > j인 a[i][j]를 0으로 만든다
# upper triangle matrix를 얻는다


a.masked_fill(a == 1, float(0.0))

# a의 원소중 1인 원소를 0으로 만든다


a.transpose(0,1)

# a를 i=i에 대해 대칭
```

### Attention
* Q(query)
    - 단어에 대한 가중치
    - 지금 이런 값이 나왔는데 원래 나와야하는 값은 무엇인가
* K(key)
    - 단어와 query의 연관성을 비교하는 가중치
* V(value)
    - 의미에 대한 가중치

### Scaled Dot-Product Attention
* $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
    - Dot-product attention에 $\sqrt{d_k}$ 를 추가한 형태
    - $d_k$ 값이 작을 때는 Dot-product attention과 Scaled dot-product attention이 거의 동일하게 작동하지만, $d_k$ 값이 클 때에는 scaling없이 dot-product attention보다 더 뛰어난 성능을 보인다

* 연산과정
1. query와 key의 dot-product를 계산하고 각각을 $\sqrt{d_k}$ 로 나눈다.
    - $\sqrt{d_k}$ 로 나누기 때문에 scaled이다
    - query와 key에 대한 dot-product를 계산하면 둘의 유사도를 알 수 있다
        + cosine similarity : dot-product / 백터의 크기
2. softmax 결과를 value와 곱한다
    - softmax 결과를 거친 값이 큰 경우 query와 유사하다
        + 더 중요한 value이기 때문에 더 높은 값을 준다


### Mulit-Head Attention
* $Multi\ Head(Q, K, V) = Concat(head_1, head_2, \dots, head_h)W^O$
    - where $head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$
        + $W^Q, W^K, W^V$ : 각각 query, key, value에 대한 weight matrix이다
* 각 Attention head마다 다른 가중치로 단어 사이의 관계를 분석하기 때문에 다양한 관계를 분석할 수 있다

### mask
* Masking
    - 해당 값을 attention에서 제외하기 위해 매우 작은 음수값(-INF 등)을 넣어준다
* Padding Mask
    - <PAD>가 key로 있는 열을 마스킹한다
* Look-ahead mask
    - 다음 단어를 예측하기 위해서는 이전의 단어만을 사용해야 하므로 다음 단어들에 대한 정보를 마스킹한다

## Self-Attention

### Encoder Self-Attention
* query, key, value 모두 이전 layer의 output에서 온다
    - 이전 layer의 모든 potision에 attention을 줄 수 있다
    - 첫 layer라면 input은 positional encoding이 더해진 input embedding이다

### Decoder Self-Attention
* masked self-attention
* auto-regressive한 특성을 위해 making out된 scaled dot-product attention을 적용한다

### Encoder-Decoder Attention Layer
* query는 이전 dexoder layer에서 오고 key와 value는 enxoder의 output에서 온다
    - decoder의 모든 position에서 encoder output의 모든 position에 attention을 줄 수 있다