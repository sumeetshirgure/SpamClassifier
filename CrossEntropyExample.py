from FeedForwardNN import FeedForwardNN as FFN

import numpy as np
import matplotlib.pyplot as plt
import random

def sech2(z) :
    return 1 - np.tanh(z)**2

def softplus(z) :
    return np.log(1 + np.exp(z))

def logistic(z) :
    return 1 / (1 + np.exp(-z))

def delsig(z) :
    r = logistic(z)
    return r - r * r

def softmax(z) :
    r = np.exp(z - np.max(z))
    return r / np.sum(r)

def plot_decision_boundary(X, y, pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    h = 0.05
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([ pred_func(z) for z in Z ])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def gold_standard (x):
    foo = (x[0]-0.1) ** 2 - x[1] ** 2 - 0.04
    rn = random.random() # Add random noise. Maybe not so golden after all.
    return 1 if rn <= logistic(10 * foo) else 0

if __name__ == '__main__' :
    # Generate "random" data.
    X = 2 * np.array([ np.random.random_sample(2) - .5 for _ in range(500) ])
    y = np.array([ gold_standard (x) for x in X ])

    # Show "target" plot
    plt.scatter([x[0] for x in X], [x[1] for x in X], c=y)
    plt.savefig('Sample_Target.png')
    nnlr = 0.01
    nnrp = 0.005

    # Train net
    nn = FFN(dimensions=[2, 5, 5, 2],
            phi=[None, np.tanh, np.tanh, None],
            dphi=[None, sech2, sech2, None],
            lr=nnlr,
            rp=nnrp)

    for i in [1, 2] : # Normalize weights
        nn.W[i] = (nn.W[i]-0.5) * 0.1
        nn.b[i] = (nn.b[i]-0.5) * 0.1

    def predict (model, x) :
        op = model.FeedForward(x)
        return np.argmax(softmax(op))

    def epoch (nn, X, y):
        n = len(X)
        L = list(range(n))
        random.shuffle(L)
        for i in L :
            op = nn.FeedForward(X[i])
            op = softmax(op)
            label = np.zeros(2)
            label[y[i]] = 1.
            nn.BackPropagate(op - label)

    for iteration in range(int(input('Enter epochs (500) :'))) :
        # Decay learning rate over epochs
        # nn.SetLearningRate( nnlr / ( 1 + np.sqrt(1+iteration) ) )
        epoch(nn, X, y)
        print("\rEpoch #", iteration, "rate=", nn.lr, end='')
        if iteration % 50 == 0 :
            err = 0
            for i in range(len(X)) :
                # Cross entropy loss
                op = softmax(nn.FeedForward(X[i]))
                label = np.zeros(2)
                label[y[i]] = 1
                loss = label * np.log(op) + (1-label) * np.log(1-op)
                err -= loss
                # Can add weight regularisation loss here.
            print(" Error=", err / len(X), end='\r')
            plot_decision_boundary(X, y, lambda x : predict(nn, x))
            plt.show()
    print()
    plt.clf()
    plot_decision_boundary(X, y, lambda x : predict(nn, x))
    plt.savefig('Sample_Learnt.png')
