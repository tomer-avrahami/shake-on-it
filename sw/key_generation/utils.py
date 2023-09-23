import os.path
import sys
sys.path.append("../cascade-python")
import numpy as np
import pyldpc as ldpc
#Dor plot TB graph only:
import networkx as nx
import matplotlib.pyplot as plt
import random
from cascade.key import Key
from cascade.shuffle import Shuffle
import time
import csv

global start_time
global curr_time
global prev_time
global fields_to_log
fields_to_log = {}


start_time = time.time()
prev_time = start_time

def str2array(my_str):
    my_arr = np.zeros(len(my_str),int)
    for i in range(len(my_str)):
        my_arr[i] = int(my_str[i])
    return my_arr

def array2str(my_array):
    str_out = ''
    for i in range(len(my_array)):
        str_out = str_out + f'{my_array[i]:0}'
    return str_out
def encode_no_errors(tG, v, seed=None):
    """Encode a binary message and adds Gaussian noise.

    Parameters
    ----------
    tG: array or scipy.sparse.csr_matrix (m, k). Transposed coding matrix
    obtained from `pyldpc.make_ldpc`.

    v: array (k, ) or (k, n_messages) binary messages to be encoded.


    Returns
    -------
    y: array (n,) or (n, n_messages) coded messages + noise.

    """
    d = ldpc.utils.binaryproduct(tG, v)
    x = (-1) ** d

    return x





def encode_message_awgn(tG, v, snr, seed=None):
    """Encode a random message given a generating matrix tG and a SNR.

    Parameters
    ----------
    tG: array or scipy.sparse.csr_matrix (m, k). Transposed coding matrix
    obtained from `pyldpc.make_ldpc`.
    snr: float. Signal-Noise Ratio. SNR = 10log(1 / variance) in decibels.

    Returns
    -------
    v: array (k,) random message generated.
    y: array (n,) coded message + noise.

    """
    rng = ldpc.utils.check_random_state(seed)
    n, k = tG.shape

    d = ldpc.utils.binaryproduct(tG, v)
    x = (-1) ** d

    sigma = 10 ** (-snr / 20)

    e = rng.randn(n) * sigma

    y = x + e

    return y

def array2d_2_nested_array(in_arr):
    y,x = in_arr.shape
    out_arr = []
    for i in range(y):
        out_arr.append(np.array(in_arr[i,:]))
    return out_arr


def create_systematic_generator(H, n, k):
    G_systematic = np.zeros((n, k), dtype=int)
    G_systematic[:k, :] = np.identity(k)
    G_systematic[k:, :] = H[:n-k, :k]
    return G_systematic

def plot_bp_graph(graph):
    fig = plt.figure()
    top = nx.bipartite.sets(graph)[0]
    labels = {node: d["label"] for node, d in g.nodes(data=True)}
    nx.draw_networkx(graph,
                     with_labels=True,
                     node_color=[d["color"] for d in g.nodes.values()],
                     pos=nx.bipartite_layout(graph, top),
                     labels=labels)
    fig.show()
    fig.savefig("tanner_graph.png")

def generate_key_pair(size = 100, error_rate = 0.1):
    Shuffle.set_random_seed(4)
    correct_key = Key.create_random_key(size)
    noisy_key = correct_key.copy(error_rate, Key.ERROR_METHOD_EXACT)
    return str(correct_key), str(noisy_key)

def add_noise_to_key(key_str:str,error_rate):
    my_key = Key()
    my_key = my_key.create_static_key(key_str)
    return str(my_key.copy(error_rate, Key.ERROR_METHOD_EXACT))


def randomize_keys(num_keys=1, key_len_list=[1000], error_rate_list=[0.05, 0.1]):
    master_key_list = []
    slave_key_list = []
    for key_len in key_len_list:
        for error_rate in error_rate_list:
            for key_num in range(num_keys):
                #Generate random keys:
                master_key_str, slave_key_str = generate_key_pair(size=key_len, error_rate=error_rate)
                master_key_list.append(master_key_str)
                slave_key_list.append(slave_key_str)
    return master_key_list, slave_key_list

def generate_rabdom_distribution(N):
    """ based on https://merl.com/publications/docs/TR2011-057.pdf
    """
    max_len = 50
    delta_strength = 10
    num_deltas = random.randint(0, 10)
    num_non_deltas = random.randint(1, max_len-num_deltas)
    probabilities = np.zeros(N+1)
    index_list = random.sample(range(1,max_len), num_deltas+num_non_deltas)
    delta_index_list = random.sample(index_list,num_deltas)
    prob = 1/(num_deltas*delta_strength + num_non_deltas)

    for i in index_list:
        if i in delta_index_list:
            probabilities[i] = delta_strength*prob
        else:
            probabilities[i] = prob

    probabilities_sum = sum(probabilities)
    EPSILON = 0.01
    assert probabilities_sum >= 1 - EPSILON and probabilities_sum <= 1 + EPSILON, "The ideal distribution should be standardized"
    return probabilities

def estimate_key(key):
    return np.array([1 if node_llr < 0 else 0 for node_llr in key])

def encode_no_errors(tG, v, seed=None):
    """Encode a binary message and adds Gaussian noise.

    Parameters
    ----------
    tG: array or scipy.sparse.csr_matrix (m, k). Transposed coding matrix
    obtained from `pyldpc.make_ldpc`.

    v: array (k, ) or (k, n_messages) binary messages to be encoded.


    Returns
    -------
    y: array (n,) or (n, n_messages) coded messages + noise.

    """
    d = ldpc.utils.binaryproduct(tG, v)
    x = (-1) ** d

    return x

def array2d_2_nested_array(in_arr):
    y,x = in_arr.shape
    out_arr = []
    for i in range(y):
        out_arr.append(np.array(in_arr[i,:]))
    return out_arr

def generate_key_pair(size = 100, error_rate = 0.1):
    Shuffle.set_random_seed(4)
    correct_key = Key.create_random_key(size)
    noisy_key = correct_key.copy(error_rate, Key.ERROR_METHOD_EXACT)
    return str(correct_key), str(noisy_key)

def write_to_csv(csvPath):
    global fields_to_log
    if not os.path.isfile(csvPath):
        csvLog = open(csvPath, 'w', newline='')
        newCsvCreated = True
    else:
        csvLog = open(csvPath, 'a', newline='')
        newCsvCreated = False
    writer = csv.writer(csvLog)
    if newCsvCreated:
        writer.writerow(list(fields_to_log.keys()))
    writer.writerow(list(fields_to_log.values()))


def print_time(msg):
    global start_time
    global curr_time
    global prev_time
    curr_time = time.time()
    print("--- %s from start: %s seconds from prev: %s seconds---" % (msg,curr_time - start_time, curr_time - prev_time))
    prev_time = curr_time

def string_is_int(s):
    '''
    Check if a string is an integer
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False

def bincount2d(arr, bins=None):
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
    indexing = np.arange(len(arr))
    for col in arr.T:
        count[indexing, col] += 1
    return count
