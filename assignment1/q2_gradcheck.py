import numpy as np
import random

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x)  # Evaluate function value at original point
    h_ = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    numgrad = np.zeros_like(x)
    while not it.finished:
        ix = it.multi_index

        ### try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later
        ### YOUR CODE HERE:
        x_adjust1 = x.copy()
        x_adjust1[ix] = x_adjust1[ix] + h_
        x_adjust2 = x.copy()
        x_adjust2[ix] = x_adjust2[ix] - h_
        random.setstate(rndstate)
        (f_x_adjust1, _) = f(x_adjust1)
        random.setstate(rndstate)
        (f_x_adjust2, _) = f(x_adjust2)
        random.setstate(rndstate)
        (f_x, grad) = f(x)
        numgrad[ix] = (f_x_adjust1 - f_x_adjust2) / h_ / 2
        ### END YOUR CODE
        # Compare gradients
        # reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        # if reldiff > 1e-5:
        #     print("Gradient check failed.")
        #     print("First gradient error found at index %s" % str(ix))
        #     print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad[ix]))
        #     # return
        it.iternext() # Step to next dimension

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        reldiff = abs(numgrad[ix] - grad[ix]) / max(1, abs(numgrad[ix]), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad[ix]))
            # return
        it.iternext()
    print("Gradient check passed!")

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print("")

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    sum_log = lambda x: (np.sum(x**3), (x ** 2) * 3)

    print("Running sanity checks...")
    gradcheck_naive(sum_log, np.array(123.456))      # scalar test
    gradcheck_naive(sum_log, np.random.randn(3,))    # 1-D test
    gradcheck_naive(sum_log, np.random.randn(4,5))   # 2-D test
    print("")
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
