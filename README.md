# NeuralArithmeticLogicalUnit-NALU
NALU and NAC from recent Deepmind paper in Pytorch. </br>
Neural networks can learn to represent and manipulate numerical information, but they seldom generalize well outside of the range of numerical values encountered during training. Even representing basic functions like addition and multiplication gives a very large error when test data is highly extrapolated. Activations like PReLU which learn to be highly linear, reduces error somewhat but sigmoid and tanh, which are sharply non-linear fail consistently.</br>
These two models, NAC and NALU are able to learn and represent numbers in a systematic way. NAC supports ability to accumulate quantities additively hence making effective predictions for extrapolated data. NALU on other hand uses NAC as it's basis and which supports multiplicative extrapolation as well.</br>
For NAC the transformation matrix W is used which contains values from set {-1, 0, 1} only. This prevents arbritary scaling of the outputs and effectively it works just like a addition/subtraction gate. </br>
Using such hard constraint for weights of W is not feasible hence W is formed by the following technique:
#### W = tanh(W_hat) * sigmoid(M_hat)
where W_hat and M_hat are weights matrices with kaiming normal initialization.</br>
This allows the weights of W to be in [-1, 1] being close to -1, 0 or 1. The model further doesn't incorprate any bias term.</br>
NALU is one step ahead of NAC in the sense that it can accumulate multiplicative extrapolation. The gates here work as follows:</br>
Let's define input as X and two weight matrices, G and W (not the one from NAC), initialized randomly (or kaiming uniform).</br>
First the NAC output is calculated:
#### NAC_OUT = NAC(input)
#### G_OUT = SIGMOID(LINEAR(INPUT, G))
#### ADDITIVE_OUT = G_OUT * NAC_OUT
This gives us the additive/subtractive gate</br>
For multiplicative/divide gate, we use logarithm
#### LOG_OUT = EXP(LINEAR(LOG(INPUT), W))
#### MULTIPLICATIVE_OUT = (1 - G_OUT) * LOG_OUT
The sigmoid from G_OUT gave us probability of additive gate hence we subtract it from 1 to get our multiplicative gate probability.
#### FINAL OUTPUT = MULTIPLICATIVE_OUT + ADDITIVE_OUT
### Model Illustration:
![](https://github.com/kevinzakka/NALU-pytorch/blob/master/imgs/arch.png)
### Usage:
Import the files.</br>
```python
from nac import NAC
from nalu import NALU
```
#### NAC
```python
net = NAC(*dims) # Creates a feed forward NAC network with dimensions given
```
#### NALU
```python
net = NALU(*dims) # Created a feed forward NALU network with dimensions given
```
