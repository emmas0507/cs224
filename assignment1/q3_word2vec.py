import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_neural import cross_entropy_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HEREå
    (N, D) = x.shape
    x_row_norm = np.linalg.norm(x, axis=1).reshape(N, 1)
    x = x/x_row_norm
    ### END YOUR CODE
    return x

def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print(x)
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print("")

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version), dim: D, central word,
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  
    
    ### YOUR CODE HERE
    # cost
    (V, D) = outputVectors.shape

    scores = np.dot(outputVectors, predicted)
    scores_prob = softmax(scores)
    cost = -np.log(scores_prob[target])
    # gradient
    label_ = np.zeros_like(scores)
    label_[target] = 1
    scores_grad = scores_prob.reshape(V, 1) * outputVectors
    gradPred = np.sum(scores_grad, axis=0) - outputVectors[target, :]

    grad = np.repeat(predicted.reshape(1, D), V, axis=0)
    grad = scores_prob.reshape(V, 1) * grad
    grad[target, :] = grad[target, :] - predicted
    return cost, gradPred, grad


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset_,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE

    (V, D) = outputVectors.shape
    pos_sigmoid = sigmoid(np.inner(predicted, outputVectors[target,:]))
    pos_ = - np.log(pos_sigmoid)
    sample_index = [dataset_.sampleTokenIdx() for i in range(K)]
    negative_sample_vec = outputVectors[sample_index,:]
    neg_sigmoid = sigmoid(-np.dot(negative_sample_vec, predicted.reshape(D, 1)))
    neg_ = - np.sum(np.log(neg_sigmoid))
    cost = pos_ + neg_
    ### END YOUR CODE

    def log_sigmoid_grad(sig):
        return (sig) * (1-sig) / sig

    gradPred_pos = - log_sigmoid_grad(pos_sigmoid) * outputVectors[target,:]
    gradPred_neg = np.sum(log_sigmoid_grad(neg_sigmoid) * outputVectors[sample_index,:], axis=0)
    gradPred = (gradPred_pos + gradPred_neg).reshape(D)

    grad_pos = - log_sigmoid_grad(pos_sigmoid) * predicted.reshape(D, 1)
    grad_neg = log_sigmoid_grad(neg_sigmoid) * np.repeat(predicted.reshape(1, D), K, axis=0)
    grad = np.zeros_like(outputVectors)
    grad[target,:] = grad_pos.reshape(D)
    for i in range(len(sample_index)):
        sindex_ = sample_index[i]
        grad[sindex_,:] = grad[sindex_,:] + grad_neg[i]
    # print('in this iteration of negative sampling')
    # print('cost: {}'.format(cost))
    # print('gradPred: {}'.format(gradPred[0:5]))
    # print('grad: {}'.format(grad[sample_index[0:3],0:3]))
    # import pdb; pdb.set_trace()
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    cost = 0
    (_, D) = inputVectors.shape
    gradIn = np.zeros_like(inputVectors)
    gradOut = np.zeros_like(outputVectors)
    # currentWord_index = tokens[currentWord]
    predicted_ = inputVectors[tokens[currentWord],:].reshape(D)
    # print('target is {}, contextWords: are {}'.format(currentWord, contextWords))
    for contextword_ in contextWords:
        target = tokens[contextword_]
        cost_, gradPredicted_, grad_ = word2vecCostAndGradient(predicted_, target, outputVectors, dataset)
        # print('target index: {}, contextword_: {}'.format(target, contextword_))
        # print(gradPredicted_)
        cost = cost + cost_
        gradIn[tokens[currentWord],:] = gradIn[tokens[currentWord],:] + gradPredicted_
        gradOut = gradOut + grad_
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def check_softmaxCostAndGradient_grad():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram grad ====")

    target = 3
    predicted = dummy_vectors[1,:]
    outputVectors = dummy_vectors[5:,:]
    print("check softmaxCostAndGradient predicted vector ")
    def predicted_helper(predicted, target, outputVectors, dataset):
        cost, gradPredicted, grad = softmaxCostAndGradient(predicted, target, outputVectors, dataset)
        return (cost, grad)
    gradcheck_naive(lambda outputVectors_: predicted_helper(predicted, target, outputVectors_, dataset), outputVectors)

def check_negSamplingCostAndGradient_grad():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram grad ====")

    target = 3
    predicted = dummy_vectors[1,:]
    outputVectors = dummy_vectors[5:,:]
    print("check negSamplingCostAndGradient output vector ")
    def predicted_helper(predicted, target, outputVectors, dataset):
        cost, gradPredicted, grad = negSamplingCostAndGradient(predicted, target, outputVectors, dataset)
        return (cost, grad)
    gradcheck_naive(lambda outputVectors_: predicted_helper(predicted, target, outputVectors_, dataset), outputVectors)

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.
    # Input/Output specifications: same as the skip-gram model
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #
    #################################################################

    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    cost = 0
    (_, D) = inputVectors.shape
    gradIn = np.zeros_like(inputVectors)
    gradOut = np.zeros_like(outputVectors)
    contextWord_index = [tokens[x] for x in contextWords]
    # import pdb; pdb.set_trace()
    predicted_ = inputVectors[contextWord_index,:].mean(axis=0).reshape(D)
    # print('target is {}, contextWords: are {}'.format(currentWord, contextWords))
    target = tokens[currentWord]
    cost, gradPredicted_, grad_ = word2vecCostAndGradient(predicted_, target, outputVectors, dataset)
    # import pdb; pdb.set_trace()
    # print(gradPredicted_)
    for i in range(len(contextWord_index)):
        contextWord_index_i = contextWord_index[i]
        gradIn[contextWord_index_i,:] = gradIn[contextWord_index_i,:] + gradPredicted_ / len(contextWord_index)
    gradOut = grad_
    ### END YOUR CODE
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in range(batchsize):
        # print('batch: {}'.format(i))
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        # context = [x for x in context if x != centerword]
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    # print("==== Gradient check for vallina skip-gram ====")
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    # print("==== Gradient check for negsampling ====")
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    # print("\n==== Gradient check for CBOW ====")
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
    # check_softmaxCostAndGradient_grad()
    # check_negSamplingCostAndGradient_grad()
