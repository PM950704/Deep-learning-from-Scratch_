import numpy as np
import matplotlib.pylab as plt

# 계단함수 
def step_function(x):
  return np.array(x > 0, dtype = np.int)

# 3.2.3 계단 함수의 그래프
x = np.arrange(-5.0, 5.0, 0.1)
y = step_function(x)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


x = np.arrange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x,y)
plt.ylim(-0.1, 1.1) #y축의 범위 지정
plt.show()

def relu(x):
    return np.maximum(0, x)
