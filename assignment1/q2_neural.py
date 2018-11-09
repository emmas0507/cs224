import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def cross_entropy_grad(scores, labels):
    """
    :param scores: scores is the input to softmax function, dimension N, C
    :param label: true label, N, C
    :return: grads respect to scores matrix, dimension N, C
    """
    softmax_ = softmax(scores)
    cross_entropy_grads = softmax_ - labels
    return cross_entropy_grads

def affine_grads(upstream_grads, x, W, b):
    """
    :param upstream_grads:
    :param x: input data to affine transformation, N, D
    :param W: weight matrix, N, H
    :param b: bias, H
    :return: grads respect to x, W, b
    """
    grads_x = np.dot(upstream_grads, np.transpose(W))
    grads_W = np.dot(np.transpose(x), upstream_grads)
    grads_b = np.sum(upstream_grads, axis=0)
    return grads_x, grads_W, grads_b

def sigmoid_input_grad(upstream_grad, sigmoid_output):
    """
    :param upstream_grad: upstream grad, same dimension as sigmoid_output
    :param sigmoid_output: element wise function: 1/(1+exp(-x)) of input data x
    :return:
    """
    return np.multiply(upstream_grad, sigmoid_grad(sigmoid_output))

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    affine_1 = np.dot(data, W1) + b1
    sigmoid_1 = sigmoid(affine_1)
    affine_2 = np.dot(sigmoid_1, W2) + b2
    scores = sigmoid(affine_2)
    cost = - np.sum(np.multiply(np.log(softmax(scores)), labels))
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    cross_entropy_grad_ = cross_entropy_grad(scores, labels)
    sigmoid_2_grads = sigmoid_input_grad(cross_entropy_grad_, scores)
    x_2_grad, gradW2, gradb2 = affine_grads(sigmoid_2_grads, sigmoid_1, W2, b2)
    sigmoid_1_grads = sigmoid_input_grad(x_2_grad, sigmoid_1)
    x_1_grad, gradW1, gradb1 = affine_grads(sigmoid_1_grads, data, W1, b1)
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def affine_check():
    N = 2
    D = 5
    H = 3

    upstream_grads = np.random.randn(N, H)
    x = np.random.randn(N, D)
    W = np.random.randn(D, H)
    b = np.random.randn(H)

    def affine_cost_x_grads(upstream_grads, x, W, b):
        b = b.reshape(1, len(b))
        affine_ = np.dot(x, W) + b
        out = np.sum(np.multiply(upstream_grads, affine_))
        grads = affine_grads(upstream_grads, x, W, b)[0]
        return (out, grads)

    def affine_cost_W_grads(upstream_grads, x, W, b):
        b = b.reshape(1, len(b))
        affine_ = np.dot(x, W) + b
        out = np.sum(np.multiply(upstream_grads, affine_))
        grads = affine_grads(upstream_grads, x, W, b)[1]
        return (out, grads)

    def affine_cost_b_grads(upstream_grads, x, W, b):
        b = b.reshape(1, len(b))
        affine_ = np.dot(x, W) + b
        out = np.sum(np.multiply(upstream_grads, affine_))
        grads = affine_grads(upstream_grads, x, W, b)[2]
        return (out, grads)

    print('running affine grads check')
    print('running affine W grads check')
    gradcheck_naive(lambda W: affine_cost_W_grads(upstream_grads, x, W, b), W)
    print('running affine b grads check')
    gradcheck_naive(lambda b: affine_cost_b_grads(upstream_grads, x, W, b), b)
    print('running affine x grads check')
    gradcheck_naive(lambda x: affine_cost_x_grads(upstream_grads, x, W, b), x)

def sigmoid_check():
    N = 2
    C = 3
    upstream_grads = np.random.randn(N, C)
    x = np.random.randn(N, C)
    def sigmoid_cost_grads(upstream_grads, x):
        cost = np.sum(np.multiply(sigmoid(x), upstream_grads))
        grad = sigmoid_input_grad(upstream_grads, sigmoid(x))
        return (cost, grad)
    print('running sigmoid grads check')
    gradcheck_naive(lambda x: sigmoid_cost_grads(upstream_grads, x), x)

def cross_entropy_check():
    N = 2
    dimension = 2
    scores = np.random.randn(N, dimension)   # each row will be a datum
    labels = np.zeros((N, dimension))
    for i in range(N):
        labels[i,random.randint(0,dimension-1)] = 1

    def cross_entropy_cost_grads(scores, labels):
        return (- np.sum(np.multiply(np.log(softmax(scores)), labels)), cross_entropy_grad(scores, labels))
    print('running cross check grads check')
    gradcheck_naive(lambda scores: cross_entropy_cost_grads(scores, labels), scores)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
    # cross_entropy_check()
    # sigmoid_check()
    # affine_check()
