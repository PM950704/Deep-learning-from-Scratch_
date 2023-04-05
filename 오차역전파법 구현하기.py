#데이터 읽기
  (x_train, t_train), (x_test, t_test) = \
      load_mnist(normalize = True, one_hot_label = True)
  
  network = TwoLayerNet(input size = 784, hidden size = 50, output_size = 10)
  
  x_batch = x_train[:3]
  t_batch = t_train[:3]
  
  grad_numerical = network.numerical_gradient(x_batch, t_batch)
  grad_backprop = network.gradient(x_batch, t_batch)
  
  # 각 가중치의 차이의 절대값을 구한 후, 그 절대값들의 평균을 낸다.
  for key in grad_numerical_keys():
      diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
      print(key + ":" + str(diff))
      """
      b1:1.194 ---
      W1:2.861 ---
      b2:1.205 ---
      W2:9.712 ---
      수치 미분과 오차역전파법으로 구한 기울기의 차이가 매우 작다.
      실수 없이 구현되었을 확률이 높다
      정밀도가 유한하기 때문에 오차가 0이 되지는 않는다.
      """
