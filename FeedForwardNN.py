# Math libraries
import math
import numpy as np

class FeedForwardNN :
    def __init__ (self, dimensions, phi, dphi, lr=0.0, rp=0.0) :
        '''
        Constructs an artificial neural network based on specified dimensions.
        The activation function and their derivatives should be specified for
        each layer. Activation functions phi are a map on R. If passed None,
        that layer's activation function acts as identity.
        Weights and biases are uniformly randomly initialized from [0., 1.].
        '''
        self.dims = dimensions
        self.depth = len(dimensions)
        self.phi = phi
        self.dphi = dphi
        self.W = [None]
        self.b = [None]
        self.x = [None]
        self.z = [None]

        # Other parameters
        self.lr = lr    # Learning rate
        self.rp = rp    # Regularization parameter

        if len(dimensions) < 2 :
            raise ValueError("At least two layers needed.")
        # Randomly initialize weights.
        for i in range(1, self.depth) :
            self.W.append(
                    np.random.random_sample((dimensions[i-1], dimensions[i])))
            self.b.append(np.random.random_sample(dimensions[i]))
        # Construct x tables.
        for i in range(1, self.depth) :
            self.z.append(np.zeros(dimensions[i]))
            self.x.append(np.zeros(dimensions[i]))

    def FeedForward (self, x) :
        '''
        Perform a feedforward pass.
        Returns the prediction according to current weights.
        '''
        if( x.shape[0] != self.dims[0] ) :
            raise ValueError("Input dimension incorrect.")
        self.x[0] = x.copy()
        for i in range(1, self.depth) :
            self.z[i] = self.x[i-1].dot(self.W[i])+self.b[i]
            if self.phi[i] :
                self.x[i] = self.phi[i](self.z[i])
            else :
                self.x[i] = self.z[i]
        return self.x[self.depth-1]

    def BackPropagate (self, nabla) :
        '''
        Performs a backpropagation pass.
        Updates the weights according to the specified gradient.
        '''
        if( nabla.shape[0] != self.dims[self.depth-1] ) :
            raise ValueError("Input dimension incorrect.")
        if self.dphi[self.depth-1] :
            dz = nabla*self.dphi[self.depth-1](
                    self.z[self.depth-1], self.x[self.depth-1])
        else :
            dz = nabla
        for i in range(self.depth-1, 0, -1) :
            dW = np.outer(self.x[i-1], dz)
            dW += self.rp * self.W[i]
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * dz
            if i > 1  :
                if self.dphi[i-1] :
                    dz = (dz.dot(np.transpose(self.W[i]))*
                            self.dphi[i-1](self.z[i-1], self.x[i-1]))
                else :
                    dz = dz.dot(np.transpose(self.W[i]))

    def SetLearningRate(self, rate) :
        '''
        Sets learning rate.
        '''
        self.lr = rate

    def SetRegularizationParameter(self, param) :
        '''
        Sets regularization parameter.
        '''
        self.rp = param
