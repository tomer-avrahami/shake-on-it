#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of utility functions for Lloyd Max Quantizer.
@author: Ninnart Fuengfusin
"""
import numpy as np
import scipy.integrate as integrate


def normal_dist(x, mean=0.0, vari=1.0):
    """A normal distribution function created to use with scipy.integral.quad
    """
    return (1.0/(np.sqrt(2.0*np.pi*vari)))*np.exp((-np.power((x-mean),2.0))/(2.0*vari))


def expected_normal_dist(x, mean=0.0, vari=1.0):
    """A expected value of normal distribution function which created to use with scipy.integral.quad
    """
    return (x/(np.sqrt(2.0*np.pi*vari)))*np.exp((-np.power((x-mean),2.0))/(2.0*vari))


def laplace_dist(x, mean=0.0, vari=1.0):
    """ A laplace distribution function to use with scipy.integral.quad
    """
    #In laplace distribution beta is used instead of variance so, the converting is necessary.
    scale = np.sqrt(vari/2.0)
    return (1.0/(2.0*scale))*np.exp(-(np.abs(x-mean))/(scale))

def expected_laplace_dist(x, mean=0.0, vari=1.0):
    """A expected value of laplace distribution function which created to use with scipy.integral.quad
    """
    scale = np.sqrt(vari/2.0)
    return x*(1.0/(2.0*scale))*np.exp(-(np.abs(x-mean))/(scale))

#def variance(x, mean=0.0, std=1.0):
#    """
#    create normal distribution 
#    """
#    return (1.0/(std*np.sqrt(2.0*np.pi)))*np.power(x-mean,2)*np.exp((-np.power((x-mean),2.0)/(2.0*np.power(std,2.0))))

def MSE_loss(x, x_hat_q):
    """Find the mean square loss between x (orginal signal) and x_hat (quantized signal)
    Args:
        x: the signal without quantization
        x_hat_q: the signal of x after quantization
    Return:
        MSE: mean square loss between x and x_hat_q
    """
    #protech in case of input as tuple and list for using with numpy operation
    x = np.array(x)
    x_hat_q = np.array(x_hat_q)
    assert np.size(x) == np.size(x_hat_q)
    MSE = np.sum(np.power(x-x_hat_q,2))/np.size(x)
    return MSE


class LloydMaxQuantizer(object):
    """A class for iterative Lloyd Max quantizer.
    This quantizer is created to minimize amount SNR between the orginal signal
    and quantized signal.
    """
    @staticmethod
    def start_repre(x, bit):
        """
        Generate representations of each threshold using 
        Args:
            x: input signal for
            bit: amount of bit
        Return:
            threshold:
        """
        assert isinstance(bit, int)
        x = np.array(x)
        num_repre  = np.power(2,bit)
        step = (np.max(x)-np.min(x))/num_repre
        
        middle_point = np.mean(x)
        repre = np.array([])
        for i in range(int(num_repre/2)):
             repre = np.append(repre, middle_point+(i+1)*step)
             repre = np.insert(repre, 0, middle_point-(i+1)*step)
        return repre

    @staticmethod
    def threshold(repre):
        """
        """
        t_q = np.zeros(np.size(repre)-1)
        for i in range(len(repre)-1):
            t_q[i] = 0.5*(repre[i]+repre[i+1])
        return t_q
    
    @staticmethod
    def represent(thre, expected_dist, dist):
        """
        """
        thre = np.array(thre)
        x_hat_q = np.zeros(np.size(thre)+1)
        #prepare for all possible integration range
        thre = np.append(thre, np.inf)
        thre = np.insert(thre, 0, -np.inf)
    
        for i in range(len(thre)-1):
             x_hat_q[i] = integrate.quad(expected_dist, thre[i], thre[i+1])[0]/(integrate.quad(dist,thre[i],thre[i+1])[0])
        return x_hat_q
    
    @staticmethod
    def quant(x, thre, repre):
        """Quantization operation. 
        """
        thre = np.append(thre, np.inf)
        thre = np.insert(thre, 0, -np.inf)
        x_hat_q = np.zeros(np.shape(x))
        for i in range(len(thre)-1):
            if i == 0:
                x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]),
                                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
            elif i == range(len(thre))[-1]-1:
                x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]), 
                                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
            else:
                x_hat_q = np.where(np.logical_and(x > thre[i], x < thre[i+1]), 
                                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
        return x_hat_q


"""main.py for Lloyd Max Quantizer.
To run with 8 bits with 1,000,000 iterations using: python3 main.py -b 8 -i 1000000
In case 8 bits, after 1,000,000 iterations, the minimum MSE loss is around 3.6435e-05.
@author: Ninnart Fuengfusin
"""
import argparse, os
#Guard protection for error: No module named 'tkinter'
try: import matplotlib.pyplot as plt
except ModuleNotFoundError:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
import numpy as np
from utils import normal_dist, expected_normal_dist, MSE_loss, LloydMaxQuantizer

parser = argparse.ArgumentParser(description='lloyd-max iteration quantizer')
parser.add_argument('--bit', '-b', type=int, default=8, help='number of bit for quantization')
parser.add_argument('--iteration', '-i', type=int, default=1_000_000, help='number of iteration')
parser.add_argument('--range', '-r', type=int, default=10, help='range of the initial distribution')
parser.add_argument('--resolution', '-re', type=int, default=100, help='resolution of the initial distribution')
parser.add_argument('--save_location', '-s', type=str, default='outputs', help='save location of representations and ')
args = parser.parse_args()

if __name__ == '__main__':
    bit = 2
    #Generate the 1000 simple of input signal as the gaussain noise in range [0,1].
    x = np.random.normal(0, 1, 1000)
    repre = LloydMaxQuantizer.start_repre(x, bit)
    min_loss = 1.0

    for i in range(args.iteration):
        thre = LloydMaxQuantizer.threshold(repre)
        #In case wanting to use with another mean or variance, need to change mean and variance in untils.py file
        repre = LloydMaxQuantizer.represent(thre, expected_normal_dist, normal_dist)
        x_hat_q = LloydMaxQuantizer.quant(x, thre, repre)
        loss = MSE_loss(x, x_hat_q)

        # Print every 10 loops
        if(i%10 == 0 and i != 0):
            print('iteration: ' + str(i))
            print('thre: ' + str(thre))
            print('repre: ' + str(repre))
            print('loss: ' + str(loss))
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        # Keep the threhold and representation that has the lowest MSE loss.
        if(min_loss > loss):
            min_loss = loss
            min_thre = thre
            min_repre = repre

    print('min loss' + str(min_loss))
    print('min thre' + str(min_thre))
    print('min repre' + str(min_repre))
    
    # Save the best thresholds and representations in the numpy format for further using.
    try: os.mkdir(args.save_location)
    except FileExistsError:
        pass
    np.save(args.save_location + '/' + 'MSE_loss', min_loss)
    np.save(args.save_location + '/' + 'thre', min_thre)
    np.save(args.save_location + '/' + 'repre', min_repre)    

    #x_hat_q with the lowest amount of loss.
    best_x_hat_q = LloydMaxQuantizer.quant(x, min_thre, min_repre)
    fig = plt.figure()
    ax = fig.add_subplot(3,1,1)
    ax.plot(range(np.size(x)), x, 'b')
    ax = fig.add_subplot(3,1,2)
    ax.plot(range(np.size(best_x_hat_q)), best_x_hat_q, 'rx')
    ax = fig.add_subplot(3,1,3)
    ax.plot(range(np.size(best_x_hat_q)), best_x_hat_q, 'y')
    plt.show()
    fig.savefig(args.save_location + '/' + 'results.png', dpi=fig.dpi)

