# Place your EWU ID and Name here. 

### Delete every `pass` statement below and add in your own code. 



# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 



import numpy as np
import math
from . import math_util as mu
from .math_util import MyMath
from .nn_layer import NeuralLayer


class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        
        new_layer = NeuralLayer(d,act)
        self.layers.append(new_layer)
        self.L+=1
    

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''
        #np.random
        for l in range(1, self.L + 1):
            current = self.layers[l]
            previous = self.layers[l-1]

            current.W = np.random.rand(previous.d +1, current.d)
            current.W = (current.W * 2-1)/ math.sqrt(current.d)

    
    
        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.
        n, d = X.shape
        num_of_batches = np.ceil(n/mini_batch_size)
        indices = np.random.permutation(n)
        X =X[indices]
        Y =Y[indices]
        if(SGD or SGD != True):
            for t in range(iterations):
                batch_num = t % num_of_batches
                start = batch_num * mini_batch_size
                end = (batch_num + 1) * mini_batch_size
                if (end > n):
                    end = n
                X_prime = X[int(start): int(end)]
                n_prime = X_prime.shape[0]
                Y_prime = Y[int(start): int(end)]
                # L is the amount of samples
                # k is classes/labels like what we are doing 0-9 images, so 10
                self.fd(X_prime)
                X_L = self.layers[self.L].X[:, 1:]
                S = self.layers[self.L].S
                act_de = self.layers[self.L].act_de
                self.layers[self.L].Delta = 2 * (X_L - Y_prime) * act_de(S)
                self.layers[self.L].G = np.einsum('ij, ik -> jk', self.layers[self.L - 1].X,
                                                  self.layers[self.L].Delta) / n_prime
                self.bp(n_prime)
                self.update_weights(eta)




        
        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions. 

        ## prep the data: add bias column; randomly shuffle data training set. 

        ## for every iteration:
        #### get a minibatch and use it for:
        ######### forward feeding
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
        ######### use the gradients to update all the weight matrices. 


    
    
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
         '''

        X = self.fd(X)
        output = self.layers[self.L].X[:, 1:]
        return np.reshape(np.argmax(output, 1), (-1, 1))

    def fd(self, X):
        self.layers[0].X = np.insert(X,0,1,axis=1)
        for l in range(1, self.L + 1):
            current_l = self.layers[l]
            prev_l = self.layers[l-1]
            current_l.S = prev_l.X @ current_l.W
            current_l.X = current_l.act(current_l.S)
            current_l.X = np.insert(current_l.X, 0, 1, axis=1)

        return X[:, 1:]


    def bp(self,N):
        for l in range(self.L - 1, 0, -1):
            current_l = self.layers[l]
            next_l = self.layers[l + 1]
            prev_l = self.layers[l - 1]
            current_l.Delta = current_l.act_de(current_l.S) * (next_l.Delta @ (next_l.W[1:]).T)
            current_l.G = np.einsum('ij, ik -> jk', prev_l.X, current_l.Delta) / N
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''

        n, d = X.shape
        X_pred = self.predict(X)
        Y_pred = np.reshape(np.argmax(Y, 1), (-1, 1))
        MSE = np.sum(X_pred != Y_pred) / n
        return MSE

    def update_weights(self, eta):
        for l in range(1, self.L + 1):
            current_l = self.layers[l]
            current_l.W = current_l.W - eta * current_l.G

 
