import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    
    ### YOUR CODE HERE
    pos_x = np.multiply(x>=0, x)
    neg_x = np.multiply(x<0, x)
    sig_pos_x = np.multiply(x>=0, 1/(1+np.exp(-pos_x)))
    sig_neg_x = np.multiply(x<0, np.exp(neg_x)/(1+np.exp(neg_x)))
    return sig_neg_x + sig_pos_x
    # ### END YOUR CODE
    #
    # return x

def sigmoid_grad(f):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input f should be the sigmoid
    function value of your original input x. 
    """
    
    ### YOUR CODE HERE
    sig_g = np.multiply(f, (1-f))
    ### END YOUR CODE
    
    return sig_g

def test_sigmoid_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print(f)
    assert np.amax(f - np.array([[0.73105858, 0.88079708], 
        [0.26894142, 0.11920292]])) <= 1e-6
    print(g)
    assert np.amax(g - np.array([[0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])) <= 1e-6
    print("You should verify these results!\n")

def test_sigmoid(): 
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    test_sigmoid_basic();
    # test_sigmoid()
