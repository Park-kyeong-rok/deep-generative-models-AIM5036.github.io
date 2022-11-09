제가 리뷰하게 된 논문은 ["NICE: NON-LINEAR INDEPENDENT COMPONENTSESTIMATION"](https://arxiv.org/pdf/1410.8516.pdf) 입니다. 해당 
논문의 목적은 normalizing flow를 이용해 
true likelihood와 가장 유사한 likelihood(
**$p(x|h)$**
)를 
학습하는 것입니다. 우선 리뷰를 시작하기전에 리뷰에 필요한 몇가지 background에 대해 설명드리겠습니다.

## 0. Background 
### 0-1. Transformation of random variables

우선 가장 첫번째로 알아야 하는 개념은 Transformation of random variables입니다. 해당 개념은 문자 그대로 확률 변수(random variables)를 다른 확률 변수로 변환(transformation)하는 기법입니다. 만약 연속 확률변수 $X$, $Y$가 존재하고 $Y = g(X)$의 관계가 만족할 경우에 아래와 같은 식을 통해 $Y$의 pdf를 구할 수 있습니다.

$$ f_{Y}(Y) = f_{X}(X)|{{\partial{x}}\over{\partial{y}}}| $$
