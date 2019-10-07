from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0] #500
    dims = X.shape[1] # 3073
    classes = dW.shape[1] #10
  
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    XW = X.dot(W)
    XW -= np.max(XW, axis=1).reshape(XW.shape[0], 1)
    # for each training examples unnormalized probability result 
    for i in range(XW.shape[0]):
        exp_scores = np.exp(XW[i])/np.sum(np.exp(XW[i]))
        loss += -np.log(exp_scores[y[i]])
        for d in range(dims):
            for k in range(classes):
                if k == y[i]:
                    dW[d, k] += X.T[d, i] * (exp_scores[k]-1)
                else:
                    dW[d, k] += X.T[d, i] * exp_scores[k]
                  
                           
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss /= XW.shape[0]
    loss += reg * np.sum(W**2) 
    
    dW /= XW.shape[0]
    dW += reg * W
                     
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    XW = X.dot(W) # probability scores
    # for numeric stability
    XW -= np.max(XW, axis=1).reshape(XW.shape[0], 1)
    XW = np.exp(XW)
    XW /= np.sum(XW, axis=1).reshape(XW.shape[0], 1)
    loss = np.sum(-np.log(XW[np.arange(XW.shape[0]), y]))
    loss /= XW.shape[0]
    loss += reg * np.sum(W**2) 
    
    XW[np.arange(XW.shape[0]), y] -= 1
    dW = np.dot(X.T, XW)
    dW /= XW.shape[0]
    dW += reg * W
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
