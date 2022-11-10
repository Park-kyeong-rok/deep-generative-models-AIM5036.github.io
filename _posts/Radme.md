제가 리뷰하게 된 논문은 ["NICE: NON-LINEAR INDEPENDENT COMPONENTSESTIMATION"](https://arxiv.org/pdf/1410.8516.pdf) 입니다. 해당 
논문의 목적은 normalizing flow를 이용해 
true likelihood와 가장 유사한 likelihood(
**$p(x|h)$**
)를 
학습하는 것입니다. 우선 리뷰를 시작하기전에 리뷰에 필요한 몇가지 background에 대해 설명드리겠습니다.

## 0. Background 
### 0-1. Transformation of random variables

우선 가장 첫번째로 알아야 하는 개념은 Transformation of random variables입니다. 해당 개념은 문자 그대로 확률 변수(random variables)를 다른 확률 변수로 변환(transformation)하는 기법입니다. 만약 연속 확률변수 $X$, $Y$가 존재하고 $Y = g(X)$가 단조 증가, 단조 감수 함수일 경우에 아래와 같은 식을 통해 $Y$의 pdf를 구할 수 있습니다.(편의상, 또한 이에 대한 많은 자료가 있기에 증명은 생략토록 하겠습니다.)

$$ f_{Y}(Y) = f_{X}(X)|{{\partial{x}}\over{\partial{y}}}| = f_{X}(g^{-1}(y))|{{\partial}\over{\partial{y}}}g^{-1}(y)| = f_{X}(g^{-1}(y)){{1}\over{|J|}}$$

위 식에서 $J = {{\partial{y}}\over{\partial{x}}}$를 jacobian matrix, $|J|$를 determinant라 합니다. 본 논문에서 determinant의 역할에 대해서 계속해 언급되는데 간단한 예를 통해 알아볼 수 있도록 하겠습니다. 

만약 아래와 같은 확률 변수 $X$와 pdf $f_X$를 가정하겠습니다.
$$f_X(x) = {{1} \over{2}}, \\{x|0 \le x \le 2\\}$$ 
<p align="center"><img src="https://user-images.githubusercontent.com/78490586/201008299-af9901d5-c385-472c-bf7b-70d7180ee1ad.png" height="200px" width="300px"></p>

이후 $Y = g(X) = {{X}\over{2}}, Z = g'(X) = 2X$인 확률 변수 $Y, Z$를 가정하고 다음과 같이 $f_{Y}(y), f_{Z}(z)$를 구해보도록 하겠습니다.

$$f_{Y}(y) = f_{X}(g^{-1}(y)){{1}\over{|J|}}$$

$$ = {{1}\over{2}} \times {{1}\over{|{{1}\over{2}}|}}, \\{y|0 \le y \le 1\\}$$

<p align="center"><img src="https://user-images.githubusercontent.com/78490586/201008285-7d12c828-6da3-4cc9-bc10-5f02f217838f.png" height="200px" width="300px"></p>

$$f_{Z}(z) = f_{X}({g^{'}}^{-1}(z)){{1}\over{|J|}}$$

$$ = {{1}\over{2}} \times {{1}\over{|2|}} \\{z|0 \le z \le 4\\}$$


<p align="center"><img src="https://user-images.githubusercontent.com/78490586/201008256-6a9d94a4-a24f-4b98-9c44-81bd8243c17b.png" height="200px" width="300px"></p>

$Y$의 경우 $|J| = {{1}\over{2}}$이고 그림을 통해 분포가 좁아진 것을 확인할 수 있습니다. $Z$의 경우 $|J| = 2$이고 분포가 넓어진 것을 확인할 수 있습니다. 즉 determinant $|J|$는 분포의 퍼진 정도에 영향을 주며 커질수록 분포가 넓어지고(expansion) 작아질수록 분포가 좁아지는(contraction) 현상을 관찰할 수 있습니다. 비록 위 예는 아주 간단한 uniform distribution을 통해 현상을 관찰하였지만 좀 더 복잡한 분포에서도 이러한 성질은 유지됩니다. Expansion과 contraction은 본 논문에서도 반복해서 사용되는 용어이기 때문에 위 예와 함께 determinant의 역할을 기억해주시면 앞으로 리뷰를 이해하는데 큰 도움이 될 수 있습니다.

### 0-2. Normalizing Flow

0-1에서도 관찰 할 수 있듯이 특정 확률 변수와 그 분포는 어떠한 변환(함수)에 의해 다른 분포를 가질 수 있습니다. 위에서 설명드린 예시들은 uniform distribution을 uniform distribution으로 변환 시키는 과정을 보여드렸지만 조금 더 복잡한 형태의 변환을 보여드리고 normalizing flow에 대해 설명드리겠습니다. 일단은 전과 같이 uniform distribution을 가지는 확률 변수 $X$를 가정하겠습니다.(위와 같은 확률 변수 $X$입니다.)
$$f_X(x) = {{1} \over{2}}, \\{x|0 \le x \le 2\\}$$ 
<p align="center"><img src="https://user-images.githubusercontent.com/78490586/201008299-af9901d5-c385-472c-bf7b-70d7180ee1ad.png" height="200px" width="300px"></p>
이때 $Y=g(X)= X^{2}+X$의 학률 분포는 다음과 같습니다.

$$ f_{Y}(y) = {{1}\over{2\sqrt{1+4y}}}, \\{y|0 \le y \le 6\\}$$

<p align="center"><img src="https://user-images.githubusercontent.com/78490586/201016263-4c433d8b-202a-4c6e-87df-bec59d01f389.png" height="200px" width="300px"></p>

이때 또 한번 $Z=g'(Y)=log(Y+1)$을 만족하는 확률 변수 $Z$를 정의하면 그 확률 분포는 다음과 같습니다.(편의상 계산 과정은 생략하였습니다.)

$$P_{Z}(z) = {{1}\over{2\sqrt{4e^{z}-3}}}e^{z}, \\{z|0 \le  z \le ln7\\}$$


식만 보아도 형태가 비교적 복잡한 것을 확인할 수 있습니다. 이 분포를 그림으로 표현하면 아래와 같습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/78490586/201087091-23f59c38-a805-4dcb-8132-83b50b2da0bb.png" height="200px" width="300px"></p>


