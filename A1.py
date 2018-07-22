import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from FeedForwardNN import FeedForwardNN

def sigmoid(z) :
    return 1 / ( 1 + np.exp(-z) )

def delsig(z, x) :
    return x * (1 - x)

if __name__ == '__main__' :
    # Iterations
    iterations = int(sys.argv[1])
    plotfreq   = int(sys.argv[2])
    plotfilename = sys.argv[3]
    logfilename  = sys.argv[4]
    logfile = open(logfilename, 'w')

    # Read .pkl files.
    with open('dict.pkl', 'rb') as df :
        dct = pickle.load(df)
        input_dimension = len(dct)
    with open('trvec.pkl', 'rb') as f :
        trvec = pickle.load(f)
    with open('tsvec.pkl', 'rb') as f :
        tsvec = pickle.load(f)
    print('Data loaded.')

    # Construct a network.
    net = FeedForwardNN([input_dimension, 100, 50, 1],
                        phi = [None, sigmoid, sigmoid, sigmoid],
                        dphi = [None, delsig, delsig, delsig],
                        lr = 0.1,
                        rp = 0.0)
    # Scale up weights. This is done to make the model converge faster.
    # Initialized between [0., 1.], scale to [-1., 1.]
    for i in [1, 2, 3] :
        net.W[i] = 2 * net.W[i] - 1
        net.b[i] = 2 * net.b[i] - 1

    def Prediction(x) :
        return net.FeedForward(x) >= 0.5

    def SquaredError(Data) :
        '''
        Returns the squared error loss function the given data set.
        '''
        err = 0
        for entry in Data :
            gold = 1 if entry[0] else 0
            err += (net.FeedForward(entry[1]) - gold) ** 2
        return err / (2 * len(Data))

    # In sample and out of sample errors
    training_error = SquaredError(trvec)
    testing_error  = SquaredError(tsvec)
    e_in = [training_error]
    e_out = [testing_error] # Plot points

    def SGD(iterations, plotfreq, index=0) :
        '''
        Run stochastic gradient descent for given number of iterations.
        Input is assumed to be already randomized.
        '''
        while iterations > 0 :
            op = net.FeedForward(trvec[index][1])
            grad = op - (1 if trvec[index][0] else 0)
            net.BackPropagate(np.array(grad))
            print('\rIterations left :%10d' % iterations, end='\r')
            # Reduce frequency
            if( iterations % plotfreq == 0 ) : # Plot after a few iterations.
                training_error = SquaredError(trvec)
                testing_error  = SquaredError(tsvec)
                e_in.append(training_error)
                e_out.append(testing_error)
                print('\nEin =', training_error, 'Eout = ', testing_error)
                logfile.write('Ein = %f Eout=%f' %
                        (training_error, testing_error))
                logfile.write(' Iterations = %d\n' % iterations)
            index += 1
            if( index == len(trvec) ) :
                index = 0
            iterations -= 1
        training_error = SquaredError(trvec)
        testing_error  = SquaredError(tsvec)
        print('\nDone. Ein =', training_error, 'Eout = ', testing_error)
        logfile.write('Ein = %f Eout = %f.\n' % (training_error, testing_error))

    # Run SGD with given parameters
    SGD(iterations, plotfreq)
    # Compute accuracy.
    acc = sum([1 if Prediction(data[1]) == data[0] else 0 for data in tsvec])
    acc /= len(tsvec)
    print('Accuracy on testing set : %2.4f' % (acc * 100) )
    logfile.write('Accuracy on testing set : %2.4f%%\n' % (acc * 100))
    plt.plot(range(len(e_in)), e_in, label='Ein')
    plt.legend()
    plt.plot(range(len(e_out)), e_out, label='Eout')
    plt.legend()
    plt.savefig(plotfilename)
    logfile.close()
