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

$X, Y, Z$의 각각 분포를 관찰하면 uniform 형태에서 비선형 형태, 단순 증가, 감소함수가 아닌 형태로 분포가 조금씩 변형되는 것을 확인할 수 있습니다. 즉, 비교적 간단한 형태의 분포도 transformation of random variables를 사용하여 형태를 조금씩 변형해줄 수록 좀 더 복잡한 형태의 분포를 가질 수 있습니다. 만약 충분히 많은 횟수, 충분히 복잡한 함수를 사용하여 변형한다면 상당히 복잡한 형태의 분포도 표현할 수 있을 것으로 생각됩니다.

'Normalizing flow'란 위와 같은 성질을 이용하는 기법이라고 할 수 있습니다. 많은 종류의 생성모델은 loglikelihood($p(X)$)를 최대화하려는 목적을 가지고 있습니다. 이러한 관점에서 볼 때 $p(X)$가 주어진 데이터를 표현할 수 있을 만큼 충분히 복잡한 형태이면 좋을 것 입니다. 그러나 기존 'VAE'와 같은 방법론들은 $p(X)$를 복잡하게 표현하기에 한계가 있기에 normalizing flow에서는 아래와 같은 과정을 통해 이러한 한계를 극복하고자 합니다. 

$$z_{i-1} \sim p_{i-1}(z_{i-1})$$

$$z_{i} = f_{i}(z_{i-1}), thus z_{i-1} = f_{i}^{-1}(z_{i})$$

$$p_{i}(z_{i}) = p_{i-1}(f_{i}^{-1}(z_{i}))|\det{{{df_{i}^{-1}}\over{dz_{i}}}}|$$

$$p_{i}(z_{i}) = p_{i-1}(f_{i}^{-1}(z_{i})){\| \det({{df_{i}}\over{dz_{i-1}}})^{-1}\|}$$

$$p_{i}(z_{i}) = p_{i-1}(z_{i-1}){\| \det{{df_{i}}\over{dz_{i-1}}}\|}^{-1}$$

위 식은 $z_{i}$의 분포가 $z_{i-1}$의 분포로부터 $f_{i}$의 transformation을 통해 생성되는 연속적인 과정을 나타내고 있습니다. 이를 그림으로 나타내면 아래와 같습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/78490586/201247340-ba9108a8-a5f2-4ed5-a4b4-9580051d5b59.png" ></p>
Transformation이 진행될 수록 좀 더 복잡한 형태의 분포가 생성되는 것을 관찰할 수 있습니다. 이제는 이를 좀 더 일반화한 식으로 표현해 보도록 하겠습니다.

$$p_{K}(z_{K}) = p_{K-1}(z_{K-1})|\det{{{df_{K}}\over{dz_{K-1}}}}|^{-1}$$

$$p_{K}(z_{K}) = p_{K-2}(z_{K-2})|\det{{{df_{K}}\over{dz_{K-1}}}}|^{-1}|\det{{{df_{K-1}}\over{dz_{K-2}}}}|^{-1}$$

$$\vdots$$

$$p_{K}(z_{K}) = p_{0}(z_{0})\prod_{i=1}^{K}{|\det{{{df_{i}}\over{dz_{i-1}}}}|^{-1}}$$

이를 계산하기 좀 더 쉽도록 log형태로 바꾸고 $z_{K} = x$임을 활용하면 다음과 같이 나타낼 수 있습니다.

$$x = z_{k} = f_{K}\circ f_{K-1}\circ f_{K-2}\circ f_{K-3}\circ\cdots \circ f_{0}(z_{0})$$

$$log p(x) =log p_{0}(z_{0}) - \sum_{i=1}^{K} log |\det{{{df_{i}}\over{dz_{i-1}}}}|$$

즉 normalizing flow란, 위와같은 식을 통하여 비교적 간단한 형태의 분포($p_{0}(z_{0})$)로 부터 복잡한 형태의 분포($p(x)$)를 계산해내는 기법이며 이를통해 기존의 다른 생성 모델에서는 얻지 못했던 복잡한 형태의 $p_{x}$를 얻을 수 있다. 다만 이에 대한 몇가지 제한이 있는데 이는 다음과 같습니다.


1. 역함수( $f^{-1}$ )가 계산하기 쉬운 형태여야 한다.
2. Determinant를 계산할 수 있어야 한다. $\leftarrow$ Jacobian이 정사각 행렬 형태로 나오도록 x, z의 차원수가 같아하고 deteminant를 계산하기 용이한 형태여야 한다.

따라서 normalizing flow를 활용하는 많은 생성 모델들은 위 제한을 지키되 복잡한 $p(x)$를 보장하도록 $f$의 형태를 복잡화하는데 주안점을 두고 있습니다.

## 1. Objective function of NICE

이제 본격적으로 본 논문의 모델 NICE에 대해 알아보도록 하겠습니다. NICE는 위에서 설명한 normalizing flow를 사용하는 flow 계열 모델입니다. 아주 간단한 prior( $p_{H}(h)$ )로 부터 복잡한 likehood( $p_{X}(x)$ )를 최대화할 수 있도록 학습을 진행합니다. 이때 deteminant를 계산할 수 있도록 input vector( $x$ )와 hidden vector( $h$ )의 차원은 같게 하도록 하며 $p_{h}(h)$ 는 각 성분이 독립이고 다음과 같이 factorize하게 분해되는 형태로 가정합니다.

$$p_{H}(h) = \prod_{d}^{D}{p_{H_{d}}}(h_{d})$$

이러한 prior 확률 변수 $h$와 input 확률 변수 $x$가 $h=f(x)$의 관계를 만족 시키면 다음 식과 같이 transformation을 적용할 수 있습니다.

$$p_{X}(x) = p_{H}(f(x))|\det{{{\partial{f(x)}}\over{\partial{x}}}}|$$

이후 다음과 같이 연산하기 쉽도록 log 형태로 바꾸어 줍니다. 아래의 $f(x)$ 가 $f_{d}(x)$ 로 분해되는 과정은 $H$의 성분 별로 다른 transformation을 적용하여 좀 더 복잡한 $H$를 만들기 위함입니다. 이는 뒤에서 좀 더 다룰 수 있도록 하겠습니다.

$$log(p_{X}(x)) = log(p_{H}(f(x)))+log(|\det{{{\partial{f(x)}}\over{\partial{x}}}}|)$$

$$log(p_{X}(x)) = \sum_{d=1}^{D}{log(p_{H_{d}}(f_{d}(x)))}+log(|\det{{{\partial{f(x)}}\over{\partial{x}}}}|)$$

위와 같은 형태는 determistic한 식으로써 연산간에 sampling이 전혀 필요가 없습니다. 따라서 일반적인 opimization 방법들(gradient ascent)을 사용해서 쉽게 maximization할 수 있습니다. 이제 maximize해야 하는 objective function을 결정했으니 역함수와 determinant를 쉽게 구할 수 있는 $f(x)$를 결정할 차례입니다.

## 2. Coupling Layer

### 2-1. General Coupling Layer

우선은 determinant를 쉽게 구할 수 있는 $f_(x)$를 만드는데 집중해보도록 하겠습니다. Determinant를 쉽게 구할 수 있는 행렬은 대표적으로 삼각행렬이 있습니다. 삼각행렬의 determinant 값은 다음과 같이 대각 성분의 곱을 통해 구할 수 있습니다.

$$ 
A_{n,n} = 
\det{(\begin{pmatrix}
a_{1,1} & 0 & \cdots & 0 \\
a_{2,1} & a_{2,2} & \cdots & 0 \\
\vdots  & \vdots  & \ddots & \vdots  \\
a_{n,1} & a_{n,2} & \cdots & a_{n,n} 
\end{pmatrix})} = \prod_{i=1}^{n}a_{i,i}
$$

따라서 본 논문에서도 jacobian을 삼각행렬로 만들기 위한 $f_(x)$를 구성합니다. 이를 위해 input $x$와 output $y$를 분해해서 연산을 진행하며 이러한 구조를 **coupling layer**라고 합니다. 분해 방식은 다음과 같습니다. $x$가 $D$차원 vector일 때 $|I_{1}|=d$와 $|I_{2}|=D-d$크기를 갖는 $x_{I_{1}}, x_{I_{2}}$으로 분해합니다. 그리고 output $y_{I_{1}}, y_{I_{2}}$를 아래와 같이 구성합니다.

$$y_{I_{1}} = x_{I_{1}}$$

$$y_{I_{2}} = g(x_{I_{2}};m(x_{I_{1}}))$$

전에 언급했듯이 determinant를 계산하기 위해서는 jacobian matrix가 정방 형태여야 합니다. 따라서 $y$도 $D$ 차원이 될 수 있도록 $g$ 함수는 $\mathbb{R}^{D-d} \times m(\mathbb{R}^{d}) \rightarrow \mathbb{R}^{D-d}$ 형태로 구성되어야 합니다. 또한 $y$의 연산 과정에 비선형 활성화 함수를 포함한 **MLP(Mulit Layer Perceptron)**($m(x)$)를 포함시켜 충분히 복잡한 output을 생성할 수 있도록했습니다. Coubling layer의 이러한 형태로 인해 아래와 같은 삼각행렬 형태의 jacobian을 얻을 수 있습니다.

$${{\partial{y}}\over{\partial{x}}} = 
\begin{bmatrix} 
I_{d} & 0 \\
{{\partial{y_{I_{2}}}}\over{\partial{x_{I_{1}}}}} & {{\partial{y_{I_{2}}}}\over{\partial{x_{I_{2}}}}}
\end{bmatrix} 
$$

결론적으로 해당 jacobian의 determinant는 대각 성분의 곱으로 아래와 같이 되며 $y_{2}$에 복잡한 형태의 $m(x_{I_{1}})$가 포함되어 있지만 삼각행렬의 determinant 정의에 따라 ${{\partial{y_{I_{2}}}}\over{\partial{x_{I_{1}}}}}$는 전혀 계산할 필요가 없어집니다. 

$$\det{{{\partial{y}}\over{\partial{x}}}} = \det{{{\partial{y_{I_{2}}}}\over{\partial{x_{I_{2}}}}}}$$

이러한 deteminant를 갖도록 설계한 layer를 **General Coupling Layer**라고 합니다. 이제는 이러한 general한 형태의 coupling layer로 부터 $\det{{{\partial{y_{I_{2}}}}\over{\partial{x_{I_{2}}}}}}$ 와 그 역함수를 쉽게 계산할 수 있는 $g$를 선택해야 합니다. 




**1. VAE(Variational Auto-Encoder)**
VAE 모델의 본디 목적$p(x)$를 최대화하는 것이은 flow 모델과 같습니다. 


