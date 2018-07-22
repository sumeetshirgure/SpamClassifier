import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from FeedForwardNN import FeedForwardNN

def sech2(z, x) :
    return 1 - x ** 2

def softmax(x) :
    r = np.exp(x - np.max(x)) # Avoid overflow
    return r / r.sum()

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
    net = FeedForwardNN([input_dimension, 100, 50, 2],
                        phi = [None, np.tanh, np.tanh, None],
                        dphi = [None, sech2, sech2, None],
                        lr = 0.1,
                        rp = 0.0)
    # Scale down weights. (Initialized between [0., 1.])
    for i in [1, 2, 3] :
        net.W[i] = 0.02 * ( net.W[i] - 0.5 )
        net.b[i] = 0.01 * ( net.b[i] - 0.5 )

    def Prediction(x) :
        '''
        Classifies x based on the model.
        '''
        return np.argmax(net.FeedForward(x))

    def LogCrossError(Data) :
        '''
        Returns the cross entropy loss for the given data set.
        '''
        err = 0
        for entry in Data :
            op = softmax(net.FeedForward(entry[1]))
            gold = np.zeros(2)
            gold[1 if entry[0] else 0] = 1
            err -= (gold * np.log(op)).sum()
        return err / len(Data)

    # In sample and out of sample errors
    training_error = LogCrossError(trvec)
    testing_error  = LogCrossError(tsvec)
    e_in = [training_error]
    e_out = [testing_error] # Plot points

    def SGD(iterations, plotfreq, index=0) :
        '''
        Run stochastic gradient descent for given number of iterations.
        Input is assumed to be already randomized.
        '''
        while iterations > 0 :
            op = softmax(net.FeedForward(trvec[index][1]))
            gold = np.zeros(2)
            gold[1 if trvec[index][0] else 0] = 1
            net.BackPropagate(op-gold)
            print('\rIterations left :%10d' % iterations, end='\r')
            # Reduce frequency
            if( iterations % plotfreq == 0 ) : # Plot after a few iterations.
                training_error = LogCrossError(trvec)
                testing_error  = LogCrossError(tsvec)
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
        training_error = LogCrossError(trvec)
        testing_error  = LogCrossError(tsvec)
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
