import sys
import os
import pickle
import numpy as np
sys.path.append(os.pardir) # 부모 디렉토리의 파일을 가져올 수 있도록 선정
form dataset.mnist import load_mnist
from common.functions import sigmoid, softmax



(x_train, y_train), (x_test, y_test) = load_mnist(flatten = True, normalize = False)

print(x_train.shape) #(60000, 784)
print(t_train.shape) #(60000, )
print(x_test.shape) #(10000, 784)
print(t_test.shape) #(10000, )


def get_data():
  (x_train, y_train), (x_test, t_test) = load_mnist(flatten = True, normalize = True, one_hot_label = False)
  return x_test, y_test

def init_network():
  with open("sample_weight.pk1", 'rb') as f:
    network = pickle.load(f)
    
    return network
  

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = newtork['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(x, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(x, W3) + b3
    y = softmax(a3)
    
    return y
  
x, t = get_data()
network = init_network()
accuracy_cnt = 0

batch_size = 100

for i in range(0, len(x), batch_size):
  x_batch = x[i:i+batch_size]
  y_batch = predict(network, x_batch)
  p = np.argmax(y_batch, axis=1)
  accuracy_cnt +=sum(p == t[i:i+batch_size])
  
  
print("Accuracy:" + str(float(accuracy_cnt) / len(x))
