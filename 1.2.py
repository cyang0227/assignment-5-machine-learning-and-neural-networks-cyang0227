import numpy as py
from scipy.special import expit, logit

def sigmoid(x):
    return expit(x)

def linear_regression(x, w, b):
    return py.dot(x, w) + b

def relu(x):
    return py.maximum(0, x)

def loss(y, fx):
    return -y * py.log(fx)

def main():
    x = py.array([1, 2, 3, 4, 5])
    v1 = py.array([0.79, -0.14, 0.13, -0.24, -0.4])
    v2 = py.array([-0.77, 0.76, 0.78, -0.51, -0.92])
    b1 = 0.02
    b2 = -0.01
    h1 = relu(linear_regression(x, v1, b1))
    h2 = relu(linear_regression(x, v2, b2))
    
    w1 = py.array([0.8, 0.58])
    w2 = py.array([0.18, 0.32])
    w3 = py.array([0.94, -0.24])
    b_w1 = 0
    b_w2 = 0.01
    b_w3 = 0.03
    
    o1 = sigmoid(linear_regression(py.array([h1, h2]), w1, b_w1))
    o2 = sigmoid(linear_regression(py.array([h1, h2]), w2, b_w2))
    o3 = sigmoid(linear_regression(py.array([h1, h2]), w3, b_w3))
    
    y = py.array([0, 1, 0])
    loss1 = loss(y[0], o1)
    loss2 = loss(y[1], o2)
    loss3 = loss(y[2], o3)
    
    print("output1: ", o1)
    print("output2: ", o2)
    print("output3: ", o3)
    
    print("loss1: ", loss1)
    print("loss2: ", loss2)
    print("loss3: ", loss3)
    
if __name__ == "__main__":
    main()