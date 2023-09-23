import sys
import os
import csv
import importlib
import sw.csi_explorer.config as config
import nistrng
decoder = importlib.import_module(f'sw.csi_explorer.decoders.{config.decoder}') # This is also an import
import matplotlib.pyplot as plt
from math import log, e
import numpy as np
import scipy
from scipy.stats import entropy
import itertools
from sw.csi_explorer.decoders.interleaved import pilots,num_sc,nulls
from datetime import datetime
import time
import pandas as pd
import hashlib

######### Cascade ##############
sys.path.append("../cascade-python")
 
from cascade.key import Key
from cascade.mock_classical_channel import MockClassicalChannel
from cascade.reconciliation import Reconciliation
from cascade.algorithm import Algorithm
from cascade.shuffle import Shuffle

############################


NEGATED_GRAY_CODE = False #True
USE_SC_ALGO = False


############################



global start_time 
global curr_time
global prev_time
global fields_to_log
fields_to_log = {}


start_time = time.time()
prev_time = start_time

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

def plotAmpsVsPktNum(samples_dict,sc, show_quan_levels = True):
    
    fig, axs = plt.subplots(3)

    sc_idx = sc - base_sc #index..
    fig.suptitle('CSI amp vs PKT for subcarrier {} '.format(sc_idx))

    # These are also cleared with clear()
    axs[0].set_ylabel('Amplitude')
    axs[0].set_xlabel('Packet number')
    
    axs[1].set_ylabel('Quantized Amplitude')
    axs[1].set_xlabel('Packet number')
    
    axs[2].set_ylabel('RSSI')
    axs[2].set_xlabel('Packet number')
    
    
    shapes = ['-+','-x']
    quan_colors = ['r','b']
    i = 0
    
    for name,samples in samples_dict.items():
        
        axs[0].plot(range(len(samples.csi_amp[:,sc])), samples.csi_amp[:,sc],shapes[i],markevery = list(np.where(samples.mask[:,sc] == 0)[0]),label = name)
        if show_quan_levels:
            for j in range(samples.margins.shape[0]):
                axs[0].axhline(y=samples.margins[j,sc], color=quan_colors[i], linestyle='--')
            
        axs[1].plot(samples.csi_amp_quan[:,sc],shapes[i],label = name)
        axs[2].plot(samples.rssi,shapes[i],label = name)
        i+=1
            
    plt.legend()
    plt.draw()
    


def getCsiDbFromPcapFile(pcap_filename,quan_type = "percentile", q = 3,nsamples_max = 0, mask_dq = 0.03, cycle_mod = 1, sel_mod = 0):
    if '.pcap' not in pcap_filename:
        pcap_filename += '.pcap'
    pcap_filepath = '/'.join([config.pcap_fileroot, pcap_filename])
    try:
        samples = decoder.read_pcap(pcap_filepath = pcap_filepath, nsamples_max = nsamples_max,quan_type = quan_type, q = q,mask_dq = mask_dq, cycle_mod = cycle_mod, sel_mod = sel_mod)
    except FileNotFoundError:
        print(f'File {pcap_filepath} not found.')
        exit(-1)
    
    return samples


def plot_inter_pilots_error_rate(samples_dict):
        for name,samples in samples_dict.items():
            for pilot_sc1 in pilots[samples_dict['alice'].bandwidth]:
                for pilot_sc2 in pilots[samples_dict['alice'].bandwidth]:
                    if pilot_sc1!=pilot_sc2:
                        x1 = samples.csi_amp_quan[:,pilot_sc1]
                        x2 =  samples.csi_amp_quan[:,pilot_sc2]
                        internal_ber = 100*( x1 != x2 ).sum()/num_packets
                        pearson_val = abs(scipy.stats.pearsonr(x1, x2)[0])
                        print(f"internal {name} error rates between {pilot_sc1 - base_sc} and {pilot_sc2 - base_sc} is {internal_ber}, pearson test: {pearson_val}")
                

def get_best_sc_by_min_corr_old(samples, sc_list, num_required_sc=4):
    #TODO: use all, use worst as cost
    pearson_corr = abs(get_pearson_mat(samples, sc_list))
    #Find best (uncorrelated) two indexes:
    min_sc_index = np.argmin(pearson_corr[np.nonzero(pearson_corr)])
    best_sc_idx = list(np.unravel_index(min_sc_index, pearson_corr.shape))
    while len(best_sc_idx) < num_required_sc:
        success = False
        corr_th = 0
        while not success:
            corr_th += 0.1
            for sc_i in range(len(sc_list)):
                if sc_i in best_sc_idx:
                    continue
                success = True
                for j in range(len(best_sc_idx)):
                    if pearson_corr[sc_i][best_sc_idx[j]] > corr_th:
                        success = False
                        break
                if success:
                    best_sc_idx.append(sc_i)
                    break
    best_sc = []
    for k in best_sc_idx:
        best_sc.append(sc_list[k])
    return best_sc


def get_best_sc_by_min_corr(samples, sc_list, num_required_sc=4):
    #TODO: use all, use worst as cost
    pearson_corr = abs(get_pearson_mat(samples, sc_list))
    #Find best (uncorrelated) two indexes:
    min_sc_index = np.argmin(pearson_corr[np.nonzero(pearson_corr)])
    best_sc_idx = list(np.unravel_index(min_sc_index, pearson_corr.shape))
    while len(best_sc_idx) < num_required_sc:
        success = False
        max_pearson_val = np.zeros(len(sc_list))
        for sc_i in range(len(sc_list)):
            for j in range(len(best_sc_idx)):
                max_pearson_val[sc_i] = max(max_pearson_val[sc_i],pearson_corr[sc_i][best_sc_idx[j]])
        min_sc_index = np.argmin(max_pearson_val[np.nonzero(max_pearson_val)])
        best_sc_idx.append(min_sc_index)

    best_sc = []
    for k in best_sc_idx:
        best_sc.append(sc_list[k])
    return best_sc




def get_pearson_mat(samples, sc_list ):
    my_num_sc = len(sc_list)
    pearson_corr = np.zeros([my_num_sc, my_num_sc])
    for i1 in range(my_num_sc):
        sc1 = sc_list[i1]
        for i2 in range(my_num_sc):
            if i2 <= i1:
                sc2 = sc_list[i2]
                x1 = samples.csi_amp_quan[:,sc1]
                x2 =  samples.csi_amp_quan[:,sc2]
                temp_corr = scipy.stats.pearsonr(x1, x2)[0]
                pearson_corr[i2,i1] = temp_corr
                pearson_corr[i1,i2] = temp_corr
    return pearson_corr



def calc_subcarrieres_pearson_correlation(samples_dict, sc_list, plot = False):
    global fields_to_log  
    my_num_sc = len(sc_list)
    base_sc = int(samples_dict['alice'].bandwidth*3.2/2)
    pearson_corr_dict = {}
    for name,samples in samples_dict.items():
        pearson_corr_dict[name] = get_pearson_mat(samples,sc_list)
        fields_to_log["pearson_corr_{}".format(name)] = pearson_corr_dict[name]
        abs_pearson = np.abs(pearson_corr_dict[name])
        try:
            mask = np.ones(abs_pearson.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            max_pearson_val = abs_pearson[mask].max()
            # max_pearson_val = np.max(abs_pearson[abs_pearson<1])
        except:
            max_pearson_val = 1
            print("Fsiled to find max pearson over {}".format(abs_pearson))
        fields_to_log["max_pearson_corr_{}".format(name)] = max_pearson_val
        fields_to_log["sum_pearson_corr_{}".format(name)] = np.sum(abs_pearson[abs_pearson<1])/2 #/2 - remove duplicated i1 i2 (above\below diagonal)
        if plot and name == "alice":
            fig, axs = plt.subplots(1)
            # fig.suptitle(f'Sub carrier cross correlation')
            axs.set_xlabel("sub carrier index")
            axs.set_ylabel("sub carrier index")
            sc_list_norm = np.append(sc_list[0:my_num_sc:8],sc_list[-1])-base_sc
            x_positions = np.append(np.arange(0,my_num_sc,8),[len(sc_list)-1])  # pixel count at label position
            x_labels = sc_list_norm  # labels you want to see
            plt.imshow(abs(pearson_corr_dict[name]), origin='lower')
            plt.xticks(x_positions, x_labels)
            plt.yticks(x_positions, x_labels)
            plt.colorbar()
            plt.show()
            fig, axs = plt.subplots(1)
            # fig.suptitle(f'Sub carrier cross correlation')
            axs.set_xlabel("subcarrier index")
            axs.set_ylabel("correlation")
            for i1 in x_positions:
                plt.plot(sc_list,pearson_corr_dict[name][i1,:], label=f'sub-carrier {sc_list_norm[i1]}')
            plt.xticks(x_positions, x_labels)
            plt.legend()
            plt.show()
    
    return pearson_corr_dict


# def plot_all_subcarrieres_pearson_correlation(samples_dict, pilots_only = False):
#     if pilots_only:
#         sc_list = pilots[samples_dict['alice'].bandwidth]
#     else:
#         sc_list = list(range(my_num_sc))
#
    # calc_subcarrieres_pearson_correlation(samples_dict = samples_dict, sc_list = sc_list, plot = False)
    
def calculate_ber(samples_dict, alice_bob_mask):
    global fields_to_log
    alice_samples = samples_dict['alice']
    bob_samples = samples_dict['bob']
    my_num_sc = np.shape(alice_samples.csi_amp)[1]
    ber = np.zeros(my_num_sc) 
    masked_ber = np.zeros(my_num_sc)
    for sc in range(my_num_sc):
#         if sc not in alice_samples.masked_sc:
        
        ber[sc] = 100*(alice_samples.csi_amp_quan[:,sc] != bob_samples.csi_amp_quan[:,sc]).sum()/num_packets
        if sc in pilots[alice_samples.bandwidth]:
            print("Channel {} BER {}%, alice histogram: {}, bob histogram: {}".format(sc-my_num_sc/2,ber[sc],alice_samples.hist[sc],bob_samples.hist[sc]))
            
        
        if np.shape(alice_bob_mask) == ():
            unmasked_packets = num_packets
            curr_mask = np.zeros(masked_alice_quan[:,sc].shape,dtype=bool)
        else:
            unmasked_packets = np.count_nonzero(alice_bob_mask[:,sc]==0)
            curr_mask = alice_bob_mask[:,sc]
        
        
        masked_ber[sc] = 100*(masked_alice_quan[:,sc] != masked_bob_quan[:,sc]).sum()/unmasked_packets
         
        
        num_bins = alice_samples.margins.shape[0] +1 
        
        alice_hist = np.bincount(masked_alice_quan[~curr_mask,sc], minlength = num_bins)
        bob_hist = np.bincount(masked_bob_quan[~curr_mask,sc], minlength = num_bins)
        if sc in pilots[alice_samples.bandwidth]:
            print("Channel {} FBER {}%, masked alice hist {}, masked bob hist {},{}/{} packets remained ({}%),".format(sc-my_num_sc/2,masked_ber[sc],alice_hist,bob_hist,unmasked_packets,num_packets,unmasked_packets/num_packets*100))
        
        if sc in pilots[alice_samples.bandwidth]:
            fields_to_log["ber_sc_{}".format(sc-my_num_sc/2)]= ber[sc]
            fields_to_log["alice_quan_level_histogram_{}".format(sc-my_num_sc/2)]= alice_samples.hist[sc]
            fields_to_log["bob_quan_level_histogram_{}".format(sc-my_num_sc/2)]= bob_samples.hist[sc]
            fields_to_log["num_packets"] = num_packets
            fields_to_log["num_unmasked_packets"] = unmasked_packets
            fields_to_log["masked_ber"] = masked_ber[sc]
    plot_ber = False
    if plot_ber:        
        fig, axs = plt.subplots(1)
        fig.suptitle('Error rate per subcarrier')
        axs.set_xlabel("subcarrier")
        axs.set_ylabel("Error rate[%]")
        
        axs.plot(list(range(-base_sc,base_sc)),ber, "-gD",markevery = pilots[alice_samples.bandwidth],label = "Error Rate")
        axs.plot(list(range(-base_sc,base_sc)),masked_ber, "-rD",markevery = pilots[alice_samples.bandwidth], label = "masked Error Rate")
        
        plt.legend()
        plt.draw()    
        
    #     plotAmpsVsPktNum({'alice':alice_samples,'bob':bob_samples}, sc = 40)
        plt.show()    
    
    return ber,masked_ber


def gray_code(n,negated = False):
    def gray_code_recurse (g,n):
        k=len(g)
        if n<=0:
            return
        else:
            for i in range (k-1,-1,-1):
                char='1'+g[i]
                g.append(char)
            for i in range (k-1,-1,-1):
                g[i]='0'+g[i]
            gray_code_recurse (g,n-1)

    g=['0','1']
    gray_code_recurse(g,n-1)
    if negated:
        for i in range(len(g)):
            old = g[i]
            g[i] = g[i].replace("0","zero")
            g[i] = g[i].replace("1","0")
            g[i] = g[i].replace("zero","1")
            print(f"negated replaced {old} with {g[i]}")
    return g


def binary_string_to_array(str_in):
    return np.fromstring(str_in,'u1') - ord('0')


def array2BinaryString(array,nBits, base = 2,gray_code_enable = True, negated_gray = False):
    
    binStr = ''
    if gray_code_enable:
        num2gray = gray_code(nBits, negated=negated_gray)
    for i in range(len(array)):
        num = array[i]
        if not np.ma.is_masked(num):
            if gray_code_enable:
                num = num2gray[int(num)]
                binStr = binStr + num
            else:
                for j in range(nBits):
                    dig = num%base
                    binStr = binStr + str(dig)
                    num = int(num/base)
                if num != 0:
                    raise ValueError("Convert array2BinaryString failed, num = {}".format(num))
    return binStr
            
def calc_pre_ber_with_eve(alice_str, bob_str, eve_alice_str, eve_bob_str):
    my_keys = {}
    my_keys['alice'] = Key.create_static_key(alice_str)
    my_keys['bob'] = Key.create_static_key(bob_str)
    my_keys['eve_alice'] = Key.create_static_key(eve_alice_str)
    my_keys['eve_bob'] = Key.create_static_key(eve_bob_str)
    keys_combinations = itertools.combinations(my_keys.keys(),2)

    for comb in keys_combinations:
        key0 = my_keys[comb[0]]
        key1 = my_keys[comb[1]]
        ber = key0.difference(key1) / len(alice_str)
        result_name = f"ber_{comb[0]}_{comb[1]}"
        fields_to_log[result_name] = ber

def reconcile_keys(algorithm, origKeyStr,noisyKeyStr, stop_on_success = True, estimated_error_rate = -1 ):
    global fields_to_log
#     Shuffle.set_random_seed(4)
    correct_key = Key.create_static_key(origKeyStr)
    noisy_key = Key.create_static_key(noisyKeyStr)
    
    alice_hist = np.bincount(list(correct_key._bits.values()))
    zeros_to_one_ratio = alice_hist[0]/alice_hist[1]
    print("zeros to ones ratio= ",zeros_to_one_ratio)
    fields_to_log["zeros to ones ratio"] = zeros_to_one_ratio
#     noisy_key = correct_key.copy(error_rate, Key.ERROR_METHOD_EXACT)
    mock_classical_channel = MockClassicalChannel(correct_key)
    if estimated_error_rate == -1:
        estimated_error_rate = correct_key.difference(noisy_key)/len(origKeyStr)
    reconciliation = Reconciliation(algorithm, mock_classical_channel, noisy_key,estimated_error_rate )
    if stop_on_success:
        reconciliation.set_break_function(lambda x: correct_key.difference(x) == 0)
    reconciliation.set_correct_key(correct_key)
    pre_reconsile_errors = correct_key.difference(noisy_key)
    orig_key_len = len(origKeyStr)
    print("pre reconcilation difference: {}/{}, error rate:{}".format(pre_reconsile_errors,orig_key_len,pre_reconsile_errors/orig_key_len))
    reconciliation.reconcile()

    stats = reconciliation.stats

#     print (reconciliation.get_reconciled_key())
    stats.print_vals()
    
    fields_to_log["ask_parity_bits"] = stats.ask_parity_bits       
    fields_to_log["ask_parity_blocks"] = stats.ask_parity_blocks   
    fields_to_log["ask_parity_messages"] = stats.ask_parity_messages
    fields_to_log["biconf_iterations"] = stats.biconf_iterations
    fields_to_log["elapsed_process_time"] = stats.elapsed_process_time
    fields_to_log["elapsed_real_time"] = stats.elapsed_real_time
    fields_to_log["infer_parity_blocks"] = stats.infer_parity_blocks
    fields_to_log["normal_iterations"] = stats.normal_iterations
    fields_to_log["realistic_efficiency"] = stats.realistic_efficiency
    fields_to_log["reply_parity_bits"] = stats.reply_parity_bits
    fields_to_log["unrealistic_efficiency"] = stats.unrealistic_efficiency

    max_normal_iter = 10
    assert len(stats.good_blocks_pre_iter) == stats.normal_iterations
    for i in range(max_normal_iter):
        if i<stats.normal_iterations:
            fields_to_log[f"good_blocks_pre_iter_{i}"] = stats.good_blocks_pre_iter[i]
            fields_to_log[f"good_bits_pre_iter_{i}"] = stats.good_bits_pre_iter[i]
            fields_to_log[f"reply_parity_bits_pre_iter_{i}"] = stats.reply_parity_bits_pre_iter[i]
            fields_to_log[f"block_size_iter_{i}"] = stats.block_size_iter[i]
            fields_to_log[f"num_blocks_iter_{i}"] = stats.num_blocks_iter[i]
        else:
            fields_to_log[f"good_blocks_pre_iter_{i}"] = 0
            fields_to_log[f"good_bits_pre_iter_{i}"] = 0
            fields_to_log[f"reply_parity_bits_pre_iter_{i}"] = 0
            fields_to_log[f"block_size_iter_{i}"] = 0
            fields_to_log[f"num_blocks_iter_{i}"] = 0

    fields_to_log["original_key_length"] = orig_key_len
    fields_to_log["wrong_bits_pre_reconcilation"] = pre_reconsile_errors
    fields_to_log["final_key_len"] = orig_key_len - stats.reply_parity_bits
    fields_to_log["key_utilization[%]"] = (fields_to_log["final_key_len"] /fields_to_log["original_key_length"])*100  


#     print(reconciliation.get_noisy_key())
    if reconciliation.get_reconciled_key().__str__() == correct_key.__str__():
        print("success reconcilation!!!")
        fields_to_log["success"] = "True"
    else:
        print("Failed!!! number of failed bits: {}".format(correct_key.difference(reconciliation.get_reconciled_key())))
        fields_to_log["success"] = "False"
    
    return stats


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


def calc_entropy2(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def calc_entropy(labels, base=None, num_packets = 0):
    """ Computes entropy of label distribution. """
    
    n_labels = len(labels)
    
    if n_labels <= 1:
      return 0
    
    values,counts = np.unique(labels, return_counts=True)
    probs = 100*(counts / num_packets)
    num_unique_vals = np.count_nonzero(probs)
    num_common = min(30,num_unique_vals)
    sorted_ind = np.argsort(counts)[::-1]
    
#     hist = 100*np.bincount(labels.astype(int))/num_packets
    
    
    
    print("most common val =  ", values[sorted_ind[0]])
    print("most common val counts =  ", counts[sorted_ind[0]])
    print("most common val probs =  ", probs[sorted_ind[0]])

    
        
    common_val = values[sorted_ind[0:num_common]]
    common_vals_probs = probs[sorted_ind[0:num_common]]
    print(common_val)  # prints the 10 most frequent elements
    print(common_vals_probs)
    
    if num_unique_vals <= 1:
      return 0
    
    ent = 0.
    
    # Compute entropyr
    base = e if base is None else base
    for i in probs/100:
      ent -= i * log(i, base)
    
    return ent,num_unique_vals, common_val,common_vals_probs
    

def calc_entropy_stats(samples_dict,nBits=2,key_sc_list = []):
    global fields_to_log
#     alice_samples = samples_dict['alice']
#     bob_samples = samples_dict['bob']
    
    sc_list = pilots[samples_dict['alice'].bandwidth]
    
    my_num_sc = np.shape(key_sc_list)[0]
    key_sc_list_to_print = key_sc_list.copy()
    for sc_i in range(my_num_sc):
        key_sc_list_to_print[sc_i] = key_sc_list[sc_i] - samples_dict['alice'].zero_sc
   
    
    for samples_name,samples in samples_dict.items():
        num_packets = np.shape(samples.csi_amp)[0]
        labels = np.zeros(num_packets,dtype = np.uint)
        masked_labels = np.zeros(num_packets)
                    
            
        for pkt_num in range(num_packets):
            for sc_i in range(my_num_sc):
                new_val = (samples.csi_amp_quan[pkt_num,key_sc_list[sc_i]] << (nBits*sc_i))
                labels[pkt_num] = labels[pkt_num] + new_val
                
        plot_correlation_in_time = False
        
        if plot_correlation_in_time:
            fig, axs = plt.subplots(1)
            fig.suptitle(f'symbol vs pkt_num, nbits_per_packet = {my_num_sc*nBits}, num_packets = {num_packets}, sc_list = {key_sc_list_to_print}')
            axs.set_xlabel("pkt number")
            axs.set_ylabel("symbol")
            axs.plot(range(num_packets),labels,"-x")
            plt.show()
        
        (my_entropy, num_unique_vals, common_val,common_vals_prob) = calc_entropy(labels,base=2,num_packets = num_packets)
        max_p = max(common_vals_prob)/100
        min_entropy = -log(max_p, 2)
            
        fields_to_log[f'shannon_entropy_{samples_name}'] = my_entropy
        fields_to_log[f'min_entropy_{samples_name}'] = min_entropy
        fields_to_log[f'num_unique_vals_{samples_name}'] = num_unique_vals
        fields_to_log[f'common_val_{samples_name}'] = common_val
        fields_to_log[f'common_val_probabilities_{samples_name}'] = common_vals_prob
        fields_to_log[f'ideal_probabilitity'] = 1/(2**(my_num_sc*nBits)) #4 - num of pilots.. TODO: use parameter
        
        plot_symbols_histograms = False
    
        if plot_symbols_histograms:
            if nBits < 3:
                fig, axs = plt.subplots(1)
                fig.suptitle(f'Symbols histogram, nbits_per_packet = {my_num_sc*nBits}, num_packets = {num_packets}, sc_list = {key_sc_list_to_print}')
                axs.set_xlabel("symbol")
                axs.set_ylabel("percentage[%]")
                plt.hist(labels,range(2**(my_num_sc*nBits)))
                plt.show()
    
    
def parse_tests_csv(db_path):
    tests_db = pd.read_csv(db_path)
    return tests_db

# def run_error_analyzer():
#     nsub = int(bandwidth * 3.2)
#     x_amp = np.arange(-1 * nsub / 2, nsub / 2)
#     x_pha = np.arange(-1 * nsub / 2, nsub / 2)
#
#     for packet_num in range(masked_alice_quan[:,:].shape[0]):
#         csi_amp = {}
#         rssi = {}
#         # alice_bob_time_diff = eve_samples_bob.time_
#         for samples_name in samples_dict.keys():
#             csi_amp[samples_name] = samples_dict[samples_name].get_csi_amp_calibrated(packet_num)
#             rssi[samples_name] = samples_dict[samples_name].get_rssi(packet_num)


def nist_check(key, hashing = None):
    global fields_to_log
    key_bits = binary_string_to_array(key)
    if hashing is None:
        key_binary_sequence = key_bits
    else:
        if hashing == "SHA1":
            key_binary_buffer = hashlib.sha1(np.packbits(key_bits)).digest()
        elif hashing == "SHA256":
            key_binary_buffer = hashlib.sha256(np.packbits(key_bits)).digest()
        else:
            raise ValueError("Hash type is not supported")
        key_bytes_array = np.frombuffer(key_binary_buffer, dtype=np.uint8)
        key_binary_sequence = np.unpackbits(key_bytes_array)

    # key_binary_sequence = np.packbits(key_bits)
    eligible_battery: dict = nistrng.check_eligibility_all_battery(key_binary_sequence,
                                                                   nistrng.SP800_22R1A_BATTERY)
    # Print the eligible tests
    print("Eligible test from NIST-SP800-22r1a:")
    for name in eligible_battery.keys():
        print("-" + name)
    # Test the sequence on the eligible tests
    results = nistrng.run_all_battery(key_binary_sequence, eligible_battery, False)
    print("Test results:")
    results_dict = {}
    for result, elapsed_time in results:
        if result.passed:
            print("- PASSED - score: " + str(np.round(result.score,
                                                         3)) + " - " + result.name + " - elapsed time: " + str(
                elapsed_time) + " ms")
        else:
            print("- FAILED - score: " + str(np.round(result.score,
                                                         3)) + " - " + result.name + " - elapsed time: " + str(
                elapsed_time) + " ms")
        results_dict[f'{result.name}_status'] = result.passed
        results_dict[f'{result.name}_score']  = result.score
    return results_dict


def get_sc_list(bandwidth, majority=None):
    all_sc_list = []
    for i in range(num_sc[bandwidth]):
        if i not in nulls[bandwidth]:
            all_sc_list.append(i)
        #majority- have enough:
    if majority is None:
        return all_sc_list
    else:
        return all_sc_list[majority:-majority-1]

if __name__ == "__main__":

    eve_filenames = []

    #5000 samples:
#     alice_filenames= ['nexmonta_D20210518_T0221_alice']
#     bob_filenames= ['nexmonta_D20210518_T0221_bob']

#100 Dynamic out of 500, dt = 1:

#     alice_filenames= ['nexmonta_D20210518_T2357_alice']
#     bob_filenames= ['nexmonta_D20210518_T2357_bob']

#1000 dynamic, dt = 0.1:
     
#     alice_filenames= ['nexmonta_D20210607_T0148_alice']
#     bob_filenames= ['nexmonta_D20210607_T0148_bob']

     
#500 in chamber with CW interference pulse 5/15, 2450MHz:

#     alice_filenames= ['nexmonta_D20210607_T2108_chamber_alice_with_interference_2450']
#     bob_filenames= ['nexmonta_D20210607_T2107_chamber_bob_with_interference_2450']


#1000 in chamber, no packet loss:
#     alice_filenames= ['nexmonta_D20210607_T2140_chamber_alice']
#     bob_filenames= ['nexmonta_D20210607_T2136_chamber_bob']


#300, new house, 1:
#     alice_filenames= ['nexmonta_D20210803_T1927_alice']
#     bob_filenames= ['nexmonta_D20210803_T1927_bob']
    #

    
     #300, new house, 2:
#     alice_filenames= ['nexmonta_D20210804_T0138_alice']
#     bob_filenames= ['nexmonta_D20210804_T0138_bob']
    
    # #300, new house, 3:
#     alice_filenames= ['nexmonta_D20210804_T0145_alice']
#     bob_filenames= ['nexmonta_D20210804_T0145_bob']
    
    
    #All 300 on new house:
    # alice_filenames= ['nexmonta_D20210803_T1927_alice','nexmonta_D20210804_T0138_alice','nexmonta_D20210804_T0145_alice'] #1000: ,'nexmonta_D20210607_T0148_alice'
    # bob_filenames= ['nexmonta_D20210803_T1927_bob','nexmonta_D20210804_T0138_bob','nexmonta_D20210804_T0145_bob']     # 1000:,'nexmonta_D20210607_T0148_bob'

#8000 new house, no LOS, :
#     alice_filenames= ['nexmonta_D20220108_T1508_alice']
#     bob_filenames= ['nexmonta_D20220108_T1508_bob']
 
     
    ##################### 5G ###########################
    
    # 5G, 40MHz channel 36, 100 packets:
    # alice_filenames= ['nexmonta_D20210901_T0247_alice']
    # bob_filenames= ['nexmonta_D20210901_T0247_bob']
    
    
    # 5G, 40MHz channel 36, 300 packets:
    # alice_filenames= ['nexmonta_D20210901_T0306_alice']
    # bob_filenames= ['nexmonta_D20210901_T0306_bob']


    #5G, 40MHz channel 36, 300 packets,  static:
#     alice_filenames= ['nexmonta_D20210901_T0311_alice']
#     bob_filenames= ['nexmonta_D20210901_T0311_bob']
    

    # # 5G, 40MHz channel 36, 300 packets,  dynamic:
#     alice_filenames= ['nexmonta_D20210901_T0314_alice']
#     bob_filenames= ['nexmonta_D20210901_T0314_bob']
    #

    
    # 5G, 40MHz channel 124 (quiet!), 300 packets,  dynamic:
#     alice_filenames= ['nexmonta_D20210901_T0349_alice']
#     bob_filenames= ['nexmonta_D20210901_T0349_bob']


########## Eve ##########
    test_db = None


    test_db = parse_tests_csv(
         "../../data/db_files/dist_tests_dual_band_230129.csv")

    if test_db is None:
        # group_files_list = ['nexmonta_D20221012_T0202_','nexmonta_D20221012_T0214_', 'nexmonta_D20221012_T0237_','nexmonta_D20221012_T0242_static_5cm_']
        # test_name = ['eve_1m', 'eve_0.5m', 'eve_5cm', 'eve_5cm_static']


        ########################### 1m #################### 0.5m ########################  5cm ######################### static 5cm ####################
        # alice_filenames = []
        # bob_filenames = []
        # eve_filenames = []
        group_files_list = ['csi_ctrl_summary_D20221210_T2057_']

        test_db = pd.DataFrame()
        for i in group_files_list:
            test_db = test_db.append({'alice_filename': i+'alice',
                            'bob_filename': i + 'bob',
                            'eve_filename': i + 'eve'
                            }, ignore_index=True)

    #####################################################
        
    CSV_FILE_SUFFIX = "csi_compare.csv"
    logs_dir = "outputs"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    csvFileName = "{}_{}".format(timestamp_str,CSV_FILE_SUFFIX)
    csvPath = os.path.join(logs_dir, csvFileName)

    #####
    max_packets_list = [0]
    stop_on_success_list = [True]
    quan_type_list = ["percentile"]
    algorithm_list = ['original']
    plot_sc_correlation = False
    allowed_wifi_bw = [20,40]
    num_repetitions = 10
    ####
    use_best = True
    if use_best:
        q_list = [4]
        mask_dq_list = [0]
        majority_margin_list = [4]
        num_key_sc_list = {20: [4], 40: [8]}
        allScList = {20 : [x + 32 for x in [-25, -7, 11, 27]],
                     40 : [x + 64 for x in [-55, -43, -24, -5, 11, 24, 43, 55]]}
    else:
        q_list = [4]
        # q_list = [2, 4, 8, 16, 32]
        majority_margin_list = [4]
        majority_margin_list = [0,1,2,3,4,5,6,7,8,9,10,11]

        mask_dq_list = [0]
        #SC list:

        num_key_sc_list = {20: [4], 40: [8]}
        # allScList = {20: [x + 32 for x in [-25, -7, 11, 27]],
        #              40: [x + 64 for x in [-55, -43, -24, -5, 11, 24, 43, 55]]}
        # use_const_sc_list = True
        # num_key_sc_list = {20: [2, 3, 4, 5, 6, 7, 8, 9, 10], 40: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
        use_const_sc_list = False
        if use_const_sc_list:
             allScList = {20: [x + 32 for x in [-24,-19,-15,-11, -7, 4, 8, 11,16, 27]],
                          40: [x + 64 for x in [-55,-43,-35,-24,-16,-9,-5,5,11,16,24,30,36,43,55,]]}
        else:
            allScList = []


    for q in q_list:
        fields_to_log = {}
        fields_to_log["quan_levels"] = q
        nBits = int(q-1).bit_length()
        # num2gray = gray_code (n = nBits, negated = NEGATED_GRAY_CODE)
        fields_to_log["NEGATED_GRAY_CODE"] = NEGATED_GRAY_CODE
        for quan_type in quan_type_list:
            fields_to_log["quan_type"] = quan_type
            for test_num in range(len(test_db)):
                curr_test_db = test_db.iloc[test_num].to_dict()
                fields_to_log.update(curr_test_db)

                # if curr_test_db['wifi_bw'] not in allowed_wifi_bw:
                #     continue
                fields_to_log["alice_filename"] = os.path.basename(curr_test_db['alice_filename'])
                fields_to_log["bob_filename"] = os.path.basename(curr_test_db['bob_filename'])
                fields_to_log["eve_filename"] = os.path.basename(curr_test_db['eve_filename'])
                for algorithm in algorithm_list:
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    if isinstance(algorithm, str):
                        print(algorithm)
                        fields_to_log["algorithm"] = algorithm
                    else:
                        fields_to_log["algorithm"] = algorithm.name
                    for stop_on_success in stop_on_success_list:
                        fields_to_log["stop_on_success"] = stop_on_success
                        for max_packets in [0]:
                            for mask_dq in mask_dq_list:
    #                         for mask_dq in [0,0.01,0.02,0.03,0.04]:
                                fields_to_log["mask_dq"] = mask_dq
                                for gray_code_enable in [True]:
                                    fields_to_log["gray_code_enable"] = gray_code_enable
                                    for majority_margin in majority_margin_list:
                                        fields_to_log["majority_val"] = majority_margin
                                        print_time("pre-get-csi-from-pcap")
                                        alice_samples = getCsiDbFromPcapFile(fields_to_log["alice_filename"],quan_type=quan_type,q=q, nsamples_max=max_packets, mask_dq = mask_dq)
                                        print_time("post-get-csi-from-pcap-alice")
                                        bob_samples = getCsiDbFromPcapFile(fields_to_log["bob_filename"],quan_type=quan_type,q=q, nsamples_max=max_packets, mask_dq = mask_dq)
                                        print_time("post-get-csi-from-pcap-bob")
                                        if fields_to_log["eve_filename"]:
                                            eve_samples_alice = getCsiDbFromPcapFile(fields_to_log["eve_filename"],
                                                                                 quan_type=quan_type, q=q,
                                                                                 nsamples_max=max_packets, mask_dq=mask_dq,
                                                                                      cycle_mod = 2, sel_mod = 0)
                                            eve_samples_bob = getCsiDbFromPcapFile(fields_to_log["eve_filename"],
                                                                                      quan_type=quan_type, q=q,
                                                                                      nsamples_max=max_packets,
                                                                                      mask_dq=mask_dq,
                                                                                      cycle_mod=2, sel_mod=1)
                                        else:
                                            eve_samples_alice = None
                                            eve_samples_bob = None
                                        samples_dict = {'alice': alice_samples, 'bob': bob_samples, 'eve_alice': eve_samples_alice, 'eve_bob': eve_samples_bob}
                                        base_sc = int(alice_samples.bandwidth*3.2/2)
                                        calc_subcarrieres_pearson_correlation(samples_dict=samples_dict,
                                                                              sc_list=get_sc_list(
                                                                              bandwidth=alice_samples.bandwidth),
                                                                              plot=plot_sc_correlation)
                                        for num_key_sc in num_key_sc_list[alice_samples.bandwidth]:
                                            for i in range(num_repetitions):
                                                if allScList == [] or allScList[alice_samples.bandwidth] == []:
                                                    sc_range_list = get_sc_list(bandwidth=alice_samples.bandwidth, majority=majority_margin )
                                                else:
                                                    sc_range_list = allScList[alice_samples.bandwidth]

                                                if USE_SC_ALGO:
                                                    sc_range_list = get_best_sc_by_min_corr(alice_samples, get_sc_list(bandwidth=alice_samples.bandwidth, majority=majority_margin ), num_key_sc)



                                                calc_subcarrieres_pearson_correlation(samples_dict=samples_dict,
                                                                                      sc_list=get_sc_list(bandwidth=alice_samples.bandwidth, majority=majority_margin ), plot=False)
                                                key_sc_groups = itertools.combinations(sc_range_list, num_key_sc)
                                                for key_sc_list_tuple in key_sc_groups:
                                                    key_sc_list = list(key_sc_list_tuple)
                                                    pilots_only = True
                                                    myPilots = pilots[alice_samples.bandwidth]
                                                    for sc in key_sc_list:
                                                        if sc not in myPilots:
                                                            pilots_only = False
                                                    print(key_sc_list)
                                                    fields_to_log["key_sc"] = [x - samples_dict['alice'].zero_sc for x in key_sc_list]
                                                    fields_to_log["num_key_sc"] = len(key_sc_list)
                                                    fields_to_log["pilots_only"] = pilots_only

                                                #     mse = ((alice_samples.csi_amp - bob_samples.csi_amp)**2).mean(axis=0)
                                                #     print(mse)

                                                    # quan_type = "kmeans" #"const"
                                                    num_packets = np.shape(alice_samples.csi_amp)[0]
            #                                         num_sc = np.shape(alice_samples.csi_amp)[1]
                                                    num_misdecisions = np.count_nonzero(alice_samples.mask != bob_samples.mask)
                                                    alice_bob_mask = np.ma.mask_or(alice_samples.mask, bob_samples.mask)
                                                    masked_alice_quan = np.ma.array(alice_samples.csi_amp_quan, mask = alice_bob_mask)
                                                    masked_bob_quan = np.ma.array(bob_samples.csi_amp_quan, mask = alice_bob_mask)
                                                    masked_eve_alice_quan = np.ma.array(eve_samples_alice.csi_amp_quan, mask = alice_bob_mask)
                                                    masked_eve_bob_quan = np.ma.array(eve_samples_bob.csi_amp_quan,
                                                                                         mask=alice_bob_mask)

                                                    print_time("pre-pearson")
                                                    calc_subcarrieres_pearson_correlation(samples_dict = samples_dict, sc_list = key_sc_list, plot = False)
                                                    print_time("post-pearson")


                                                    #plot subcarriers pearson correlation:
                                                    if False:
                                                        plot_subcarrieres_pearson_correlation(samples_dict, pilots_only = False)

                                                    if 0:
                                                        plot_inter_pilots_error_rate(samples_dict)
                                                        for pilot_sc in pilots[alice_samples.bandwidth]:
                                                            plotAmpsVsPktNum({'alice':alice_samples,'bob':bob_samples}, sc = pilot_sc, show_quan_levels = True)

                                                    #### bit stream:
                                                    alice_key = ""
                                                    bob_key  = ""
                                                    eve_alice_key = ""
                                                    eve_bob_key = ""
                                                    key_db = pd.DataFrame()
                                                    for pilot_sc in key_sc_list:
                                                        if majority_margin > 0:
                                                            for packet_num in range(masked_alice_quan[:,pilot_sc].shape[0]):
                                                                masked_alice_quan[packet_num, pilot_sc] = np.bincount(masked_alice_quan[packet_num,pilot_sc-majority_margin:pilot_sc+majority_margin]).argmax()
                                                                masked_bob_quan[packet_num, pilot_sc] = np.bincount(masked_bob_quan[packet_num,pilot_sc-majority_margin:pilot_sc+majority_margin]).argmax()
                                                                masked_eve_alice_quan[packet_num, pilot_sc] = np.bincount(
                                                                    masked_eve_alice_quan[packet_num,
                                                                    pilot_sc - majority_margin:pilot_sc + majority_margin]).argmax()
                                                                masked_eve_bob_quan[packet_num, pilot_sc] = np.bincount(
                                                                    masked_eve_bob_quan[packet_num,
                                                                    pilot_sc - majority_margin:pilot_sc + majority_margin]).argmax()

                                                        alice_key = alice_key + array2BinaryString(masked_alice_quan[:,pilot_sc].flatten(), nBits=nBits,gray_code_enable=gray_code_enable, negated_gray=NEGATED_GRAY_CODE)
                                                        bob_key = bob_key + array2BinaryString(masked_bob_quan[:,pilot_sc].flatten(),nBits=nBits, gray_code_enable=gray_code_enable, negated_gray=NEGATED_GRAY_CODE)
                                                        eve_alice_key = eve_alice_key + array2BinaryString(masked_eve_alice_quan[:,pilot_sc].flatten(), nBits=nBits, gray_code_enable=gray_code_enable, negated_gray=NEGATED_GRAY_CODE)
                                                        eve_bob_key = eve_bob_key + array2BinaryString(
                                                            masked_eve_bob_quan[:, pilot_sc].flatten(), nBits=nBits,
                                                            gray_code_enable=gray_code_enable)

                                                        #Build key db:
                                                        # key_db['alice_val']
                                                    final_keys_str = {}
                                                    fields_to_log['final_key'] = alice_key
                                                    final_keys_str['alice'] =  alice_key
                                                    final_keys_str['bob'] = bob_key
                                                    final_keys_str['eve_alice'] = eve_alice_key
                                                    final_keys_str['eve_bob'] = eve_bob_key

                                                    for name,key in final_keys_str.items():
                                                        print(f'{name}_key : {key}')

                                                    calc_pre_ber_with_eve(alice_key, bob_key, eve_alice_key, eve_bob_key)
                                                    #TODO: error analyzer:
                                                    # run_error_analyzer()
                                                    ############
                                                    #TODO: returns different number of values for each test..
                                                    # for hashing in [None, 'SHA1', 'SHA256']:
                                                    #     nist_results = nist_check(alice_key,hashing=hashing)
                                                    #     prefix = ''
                                                    #     if hashing != None:
                                                    #         prefix = hashing
                                                    #     for key,val in nist_results.items():
                                                    #         fields_to_log["nist_"+prefix+"_"+key] = val


                                                    # binary_string_to_array(alice_key)

                                                    ########## Reconcilation ##########
                                                #     print(alice_key)
                                                #     print(bob_key)
                                                    print_time("pre-reconsile")
                                                    stats = reconcile_keys(algorithm = algorithm, origKeyStr = alice_key,noisyKeyStr = bob_key, stop_on_success=stop_on_success)
                                                    print_time("post-reconsile")
                                                    ##############################################
                                                    ###################
                                                    #nist check


                                                    ############

                                                    #calculate_ber(samples_dict,alice_bob_mask = alice_bob_mask)

                                                    print_time("pre-calc-entropy")
                                                    calc_entropy_stats(samples_dict, nBits=nBits, key_sc_list = key_sc_list)
                                                    print_time("post-calc-entropy")

                                                    #Calc real bits per packet:

                                                    samples_name = 'alice'
                                                    fields_to_log[f'total_secured_bits'] = fields_to_log[f'final_key_len']*fields_to_log[f'min_entropy_{samples_name}']/(nBits*num_key_sc)
                                                    fields_to_log[f'real_bits_per_packet'] = fields_to_log[f'final_key_len']*fields_to_log[f'min_entropy_{samples_name}']/(num_packets*nBits*num_key_sc)

                                                    #### calculate time:
                                                    # calculate_time_to_key()
                                                    cascade_side_channel_message_time = 0.2
                                                    fields_to_log['cascade_communication_penalty[s]'] = cascade_side_channel_message_time*fields_to_log['ask_parity_messages']
                                                    if 'failed_attempts_total' in fields_to_log.keys():
                                                        fields_to_log[
                                                            'estimated_retrial_penalty[s]'] = cascade_side_channel_message_time * \
                                                                                                  fields_to_log[
                                                                                                      'failed_attempts_total']
                                                        fields_to_log[
                                                            'time_to_all_bits[s]'] = fields_to_log['cascade_communication_penalty[s]'] + fields_to_log['collect_duration']

                                                    for requested_key_len in [128,256]:
                                                        try:
                                                            val = (fields_to_log[
                                                                                'cascade_communication_penalty[s]'] +
                                                                                fields_to_log['collect_duration']) * \
                                                                                (requested_key_len/
                                                                                 fields_to_log[f'total_secured_bits'])
                                                        except:
                                                            val = 9999999

                                                        fields_to_log[
                                                            f'time_to_{requested_key_len}_bits_key[s]'] = val

                                                        fields_to_log[
                                                            f'packets_to_{requested_key_len}_bits_key[s]'] = \
                                                            requested_key_len / fields_to_log[f'real_bits_per_packet']

                                                    ###################
                                                    print_time("pre-write-to-csv")
                                                    write_to_csv(csvPath=csvPath)
                                                    print_time("post-write-to-csv")

                        
    print("Done")
