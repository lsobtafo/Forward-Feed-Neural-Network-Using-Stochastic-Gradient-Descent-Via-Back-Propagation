# Place your EWU ID and name here

## delete the `pass` statement in every function below and add in your own code. 


import numpy as np



# Various math functions, including a collection of activation functions used in NN.




class MyMath:

    def tanh(x):
        ''' tanh function.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        turn_into_np_array = np.array(x)
        return np.tanh(turn_into_np_array)

    def tanh_de(x):
        ''' Derivative of the tanh function.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''
        turn_into_np_array = np.array(x)
        math = np.tanh(turn_into_np_array)
        return 1 - np.power(math, 2)

    def logis(x):
        ''' Logistic function.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of
                    the corresponding element in array x
        '''
        x = np.array(x)
        return 1 / 2 * (np.tanh(x / 2) + 1)

    def logis_de(x):
        ''' Derivative of the logistic function.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of
                    the corresponding element in array x
        '''
        x = np.array(x)
        return MyMath.tanh_de(x / 2) / 4

    def iden(x):
        ''' Identity function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        x = np.array(x)
        return x

    def iden_de(x):
        ''' The derivative of the identity function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        x = np.array(x)
        return np.ones(x.shape)

    def relu(x):
        ''' The ReLU function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the max of: zero vs. the corresponding element in x.
        '''
        x = np.array(x)
        vmax = np.vectorize(max)
        return vmax(x, 0)

    def _relu_de_scaler(x):
        ''' The derivative of the ReLU function. Scaler version.

            x: a real number
            return: 1, if x > 0; 0, otherwise.
        '''
        x = np.array(x)
        if x >= 0:
            return x
        else:
            return 0

    def relu_de(x):
        ''' The derivative of the ReLU function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the _relu_de_scaler of the corresponding element in x.
        '''
        x = np.array(x)
        return np.where(x > 0, 1, 0)

    