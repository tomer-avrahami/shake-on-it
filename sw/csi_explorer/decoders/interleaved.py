'''
Interleaved
===========

Fast and efficient methods to extract
Interleaved CSI samples in PCAP files.

~230k samples per second.

Suitable for bcm43455c0 and bcm4339 chips.

Requires Numpy.

Usage
-----

import decoders.interleaved as decoder

samples = decoder.read_pcap('path_to_pcap_file')

Bandwidth is inferred from the pcap file, but
can also be explicitly set:
samples = decoder.read_pcap('path_to_pcap_file', bandwidth=40)
'''


__all__ = [
    'read_pcap'
]

import math
import os
import numpy as np
from math import sqrt
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
# import quantisizers.maxlloyd

# Indexes of Null and Pilot OFDM subcarriers
# https://www.oreilly.com/library/view/80211ac-a-survival/9781449357702/ch02.html

num_sc = {
    20 : 64,
    40 : 128,
    80 : 256,
    160 : 512
    }

nulls = {
    20: [x+32 for x in [
        -32, -31, -30, -29,
         #-28,28, #saw several packets with enormous CSI values
        #-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9, #added by Tomer,
              31,  30,  29, 0
    ]],

    40: [x+64 for x in [
        -64, -63, -62, -61, -60, -59, -1,
            -58,58,-4,-3,-2,2,3,4, #added by Tomer, saw several packets with enormous CSI values
              63,  62,  61,  60,  59,  1,  0
    ]],

    80: [x+128 for x in [
        -128, -127, -126, -125, -124, -123, -1,
               127,  126,  125,  124,  123,  1,  0
    ]],

    160: [x+256 for x in [
        -256, -255, -254, -253, -252, -251, -129, -128, -127, -5, -4, -3, -2, -1,
               255,  254,  253,  252,  251,  129,  128,  127,  5,  4,  3,  3,  1,  0 
    ]]
}

pilots = {
    20: [x+32 for x in [
        -21, -7,
         21,  7
    ]],

    40: [x+64 for x in [
        -53, -25, -11, 
         53,  25,  11
    ]],

    80: [x+128 for x in [
        -103, -75, -39, -11,
         103,  75,  39,  11
    ]],

    160: [x+256 for x in [
        -231, -203, -167, -139, -117, -89, -53, -25,
         231,  203,  167,  139,  117,  89,  53,  25
    ]]
}

############ move to utils ###############

import graycode

def np_gray_code(a,nbits=None):
    result = np.zeros_like(a)
    if nbits is None:
        nbits = np.max(a).bit_length()
    gray_code_table = graycode.gen_gray_codes(nbits)
    for i in range(len(gray_code_table)):
        result[a == i] = gray_code_table[i]
    return result

###############


class SampleSet(object):
    '''
        A helper class to contain data read
        from pcap files.
    '''


    def __init__(self, samples, bandwidth, quan_type = "percentile", q = 4, mask_dq = 0.03,ts_sec= None,ts_usec = None, time_diff_usec = None,gray_coding = False):
        self.rssi,self.frame_ctl,self.mac, self.seq, self.css, self.csi = samples
        self.q = q
        self.mask_dq = mask_dq
        self.nsamples = self.csi.shape[0]
        self.bandwidth = bandwidth
        self.num_sc = num_sc[self.bandwidth]
        self.zero_sc = self.num_sc/2
        self.csi_amp = None
        self.quan_type = quan_type
        self.gray_coding = gray_coding
        self.mask = np.zeros(np.shape(self.csi))
        self.generate_quan_filter()
        self.time_diff_usec = time_diff_usec
        self.ts_sec = ts_sec
        self.ts_usec = ts_usec
        self.majority_margin = 0
        self.valid_sc = [True] * self.num_sc
        for sc in nulls[self.bandwidth]:
            self.valid_sc[sc] = False


#         self.generate_mahalanobis_distance()

    def get_nbits(self):
        return int(math.log2(self.q))

    def generate_key(self):
        self.generate_full_csi_amp()
        self.quantize_samples()
        self.majority_filter()
        self.generate_gray_code_array()

    def get_kmeans_vals(self,data):
        data = data.reshape(-1,1)
        k = self.q
#         kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10, algorithm = "full")
        init_centers = np.zeros(self.q)
        
#         const_per = [5,35, 65, 95]
        for i in range(k):
            per = (i+1)*100/(k+1)
            print(per)
            init_centers[i] = np.percentile(a = data,q = per ,axis = 0)
             
#         init_centers = np.linspace(np.percentile(data, int(100/k),0),np.percentile(data, 95,0),k)
        init_centers = init_centers.reshape(-1,1)
        print(init_centers)
        kmeans = KMeans(init=init_centers, n_clusters=k, n_init=1)
        
        
        kmeans.fit(data)
        return [kmeans.cluster_centers_,kmeans.predict(data)]
    
    
    
    def get_vor_vals(self):
        return Voronoi(points) 
    
        
        
    
#     def kmeans_samples(self):
#         q = 5
#         self.kmeans_quan = np.zeros_like(self.csi_amp,'int')
#         for cs in range(np.shape(self.kmeans_quan)[1]):
# #                 print(margins[:,cs])
#                 self.kmeans_quan[:,cs] =  self.get_kmeans_vals(self.csi_amp[:,cs])
#                 
#                 
    def set_quan_params(self,quan_type,q,mask_dq=0):
        self.quan_type = quan_type
        self.q = q
        self.mask_dq = mask_dq

    def set_majority_margin_val(self,majority_margin):
        self.majority_margin = majority_margin

    def quantize_samples(self):
        q = self.q
        quan_type = self.quan_type
        self.csi_amp_quan = np.zeros_like(self.csi_amp,'ushort')
        
        
        if quan_type == "percentile":
            margins = np.zeros([q-1,np.shape(self.csi_amp_quan)[1]])
            margins = np.percentile(self.csi_amp, np.linspace(100/q,100-100/q,q-1),0)
            # if q==4:            
            #     margins = np.percentile(self.csi_amp, [25,50,75],0)
            # elif q==3:            
            #     margins = np.percentile(self.csi_amp, [33,66],0)
            # elif q==2:
            #     margins = np.percentile(self.csi_amp, [50],0)
            # else:
            #     raise ValueError(f"q {q} is not supported")
        elif quan_type == "fixed_percentile":
            margins = np.zeros([q-1,np.shape(self.csi_amp_quan)[1]])
            med = np.percentile(self.csi_amp, [50],0)
            if q==5:       
                margins[0] = np.max(med-0.2,0)
                margins[1] = np.max(med-0.1,0)
                margins[2] = np.max(med+0.1,0)
                margins[3] = np.max(med+0.2,0)
            elif q==4:            
                margins[0] = np.max(med-0.4,0)
                margins[1] = np.max(med,0)
                margins[2] = np.max(med+0.4,0)
            elif q==3:
                margins[0] = np.max(med-0.1,0)
                margins[1] = np.max(med+0.1,0)
            else:
                raise ValueError(f"q {q} is not supported for fixed_percentile")
        elif quan_type == "mean":
            margins = np.zeros([q+1,np.shape(self.csi_amp_quan)[1]])
            min = np.min(self.csi_amp,0)
            max = np.max(self.csi_amp,0)
            mean = np.mean(self.csi_amp,0)
            dq = 0.2
            real_q = (max-min)/dq
            margins[0] = min
            margins[q] = max
            for m in range(1,q):
                margins[m] = mean + (m/2-m)*dq
        elif quan_type == "a2d":
            margins = np.linspace(np.percentile(self.csi_amp, 5,0),np.percentile(self.csi_amp, 95,0),q+2)
        elif quan_type == "lloyd":
            pass
        elif quan_type in ["kmeans", "dbscan","SpectralClustering"] :
            margins = np.zeros([q,np.shape(self.csi_amp_quan)[1]])
        else:
            raise ValueError("Quantization of type {} is not supported".format(quan_type))
            
        num_bins = margins.shape[0] + 1
        self.hist = np.zeros([self.csi.shape[1],num_bins])
        
        for cs in range(np.shape(self.csi_amp_quan)[1]):
            if quan_type == "kmeans":
                cs_margin,cs_amp_quan = self.get_kmeans_vals(self.csi_amp[:,cs])
                margins[:,cs] =  cs_margin.squeeze()
                self.csi_amp_quan[:,cs] = cs_amp_quan
#                 print(margins[:,cs])
            elif quan_type == "dbscan":
                clustering = DBSCAN(eps=0.05, min_samples=100).fit(self.csi_amp[:,cs].reshape(-1,1))
                cs_amp_quan = clustering.labels_
                self.csi_amp_quan[:,cs] = cs_amp_quan
                n_clusters_ = len(set(cs_amp_quan)) - (1 if -1 in cs_amp_quan else 0)
                margins[:,cs] = np.ones(q)*n_clusters_
                
#                 margins[:,cs] = 
#                 n_clusters_ = len(set(cs_amp_quan)) - (1 if -1 in labels else 0)
#                 n_noise_ = list(labels).count(-1)
#                 a = self.get_voronoi_vals(self.csi_amp[:,cs],k=q)
#                 cs_margin,cs_amp_quan = self.get_kmeans_vals(self.csi_amp[:,cs],k=q)
#                 margins[:,cs] =  cs_margin.squeeze()
#                 self.csi_amp_quan[:,cs] = cs_amp_quan
            elif quan_type == "SpectralClustering":
                clustering = SpectralClustering(n_clusters = q, random_state = 456).fit(self.csi_amp[:,cs].reshape(-1,1))
                cs_amp_quan = clustering.labels_
                self.csi_amp_quan[:,cs] = cs_amp_quan
                n_clusters_ = len(set(cs_amp_quan)) - (1 if -1 in cs_amp_quan else 0)
                margins[:,cs] = np.ones(q)*n_clusters_
                
            
            else:
                #if np.count_nonzero(margins[:,cs]) != 0 and : #Why?? changed for percentile 2:
                if np.count_nonzero(margins[:,cs]) != 0:
                    if margins[-1,cs] > margins[0,cs] or margins.shape[0] == 1: 
                        self.csi_amp_quan[:,cs] = np.digitize(self.csi_amp[:,cs],margins[:,cs])
            self.hist[cs] = np.bincount(self.csi_amp_quan[:,cs], minlength = num_bins)
        self.margins = margins


    def generate_gray_code_array(self):
        self.csi_amp_quan_grayed = np.zeros_like(self.csi_quan_with_major)
        for cs in range(np.shape(self.csi_quan_with_major)[1]):
            self.csi_amp_quan_grayed[:, cs] = np_gray_code(self.csi_quan_with_major[:, cs], self.get_nbits())


#         print(self.hist)
            
    def majority_filter(self):
        majority_margin = self.majority_margin
        self.csi_quan_with_major = np.zeros_like(self.csi_amp, 'short')
        for pilot_sc in range(self.csi_amp.shape[1]):
            self.valid_sc[pilot_sc] = True
            min_pilot_sc = pilot_sc - majority_margin
            max_pilot_sc = pilot_sc + majority_margin + 1
            if min_pilot_sc < 0 or min_pilot_sc > self.csi_amp.shape[1]:
                self.valid_sc[pilot_sc] = False
            else:
                for null_sc in nulls:
                    if null_sc in list(range(min_pilot_sc, max_pilot_sc)):
                        self.valid_sc[pilot_sc] = False
            if self.valid_sc[pilot_sc]:
                for packet_num in range(self.csi_amp_quan[:, pilot_sc].shape[0]):
                    self.csi_quan_with_major[packet_num, pilot_sc] = np.bincount(
                        self.csi_amp_quan[packet_num, :pilot_sc + majority_margin+1]).argmax()


    def generate_quan_filter(self):

        if self.mask_dq > 0:
            for sc in range(self.csi_amp.shape[1]):
                for margin_idx in range(self.margins.shape[0]):
                    sc_mask_idx = None
                    sc_mask_idx = abs(self.csi_amp[:,sc]-self.margins[margin_idx,sc]) < self.mask_dq
                    self.mask[sc_mask_idx,sc] = 1
                         
        
    def generate_mahalanobis_distance(self):
        mean = np.mean(self.csi_amp,0)
        cov = np.cov(self.csi_amp.T)
        icov = np.linalg.inv(cov)
        num_packets = self.csi_amp.shape[0]
#         maha_dis = zeros
        for n_pkt in range(num_packets):
            maha_dis[n_pkt] = distance.mahalanobis(self.csi_amp[n_pkt, :], mean, icov)
        
        
    def generate_full_csi_amp(self, rm_nulls=True, rm_pilots=False):
        self.csi_amp = np.abs(self.csi)
        if rm_nulls:
            self.csi_amp[:,nulls[self.bandwidth]] = 0
        if rm_pilots:
            self.csi_amp[:,pilots[self.bandwidth]] = 0
            
        self.csi_amp = np.abs(self.csi_amp)
        csi_amp_sum = np.sum(np.square(self.csi_amp),1)
        # s = 10**((np.abs(self.rssi))/10)/csi_amp_sum
        s = np.abs(self.rssi)/ csi_amp_sum
        # s = 1 / csi_amp_sum
        s = np.sqrt(s)
        # s = s/s
        self.csi_amp = self.csi_amp*s[:,None] 
#         self.csi_amp = np.diff(self.csi_amp,1,0)
            
        
        
    def get_mac(self, index):
        return self.mac[index*6: (index+1)*6]

    def get_seq(self, index):
        sc = int.from_bytes( #uint16: SC
            self.seq[index*2: (index+1)*2],
            byteorder = 'little',
            signed = False
        )
        fn = sc % 16 # Fragment Number
        sc = int((sc - fn)/16) # Sequence Number

        return (sc, fn)
    
    def get_css(self, index):
        return self.css[index*2: (index+1)*2]

    def get_csi(self, index, rm_nulls=False, rm_pilots=False):
        csi = self.csi[index].copy()
        if rm_nulls:
            csi[nulls[self.bandwidth]] = 0
        if rm_pilots:
            csi[pilots[self.bandwidth]] = 0

        return csi
    
    def get_rssi(self,index):
        return self.rssi[index: (index+1)]
    
    def get_frame_ctl(self,index):
        return self.frame_ctl[index: (index+1)]
    
    
    def get_csi_amp_calibrated(self,index, rm_nulls=False, rm_pilots=False):
        csi = self.get_csi(index, rm_nulls=rm_nulls, rm_pilots=rm_pilots)
        rssi = self.get_rssi(index)
        csi_amp = np.abs(csi)
        csi_amp_sum = np.sum(np.square(csi_amp))
        s= sqrt(rssi/csi_amp_sum)
        csi_amp = (csi_amp*s)
        return csi_amp

    def get_time_usec_from_prev_packet(self,index):
        return self.time_diff_usec[index]

    def get_time_usec_from_first_packet(self, index):
        return (self.ts_sec[index] - self.ts_sec[0])*10**6 \
                                           + self.time_diff_usec[index]-self.time_diff_usec[0]

    def print(self,index):
        print(self.to_str(index))

    def to_str(self, index):
        # Mac ID
        macid = self.get_mac(index).hex()
        macid = ':'.join([macid[i:i+2] for i in range(0, len(macid), 2)])

        # Sequence control
        sc, fn = self.get_seq(index)
        sc0, _ = self.get_seq(0)
        if sc0 < 0:
            sc0 += 4096

        # Core and Spatial Stream
        css = self.get_css(index).hex()
        rssi = self.get_rssi(index).hex()
        frame_ctl = self.get_frame_ctl(index).hex()
        time_from_first_packet = self.get_time_usec_from_first_packet(index)
        time_from_prev_packet = self.get_time_usec_from_prev_packet(index)


        return f'''
    Sample #{index}\n \
    ---------------\n \
    Source Mac ID: {macid}\n \
    Sequence: {sc}\n \
    sequence index from first packet: {sc - sc0}\n \
    Frame number:{fn}\n \
    Core and Spatial Stream: 0x{css}\n \
    RSSI: 0x{rssi}\n \
    frame control: {frame_ctl}\n \
    time from first packet: {time_from_first_packet}\n \
    time from prev packet: {time_from_prev_packet}\n \
    '''


def __find_bandwidth(incl_len):
    '''
        Determines bandwidth
        from length of packets.
        
        incl_len is the 4 bytes
        indicating the length of the
        packet in packet header
        https://wiki.wireshark.org/Development/LibpcapFileFormat/

        This function is immune to small
        changes in packet lengths.
    '''

    pkt_len = int.from_bytes(
        incl_len,
        byteorder='little',
        signed=False
    )

    # The number of bytes before we
    # have CSI data is 60. By adding
    # 128-60 to frame_len, bandwidth
    # will be calculated correctly even
    # if frame_len changes +/- 128
    # Some packets have zero padding.
    # 128 = 20 * 3.2 * 4
    nbytes_before_csi = 60
    pkt_len += (128 - nbytes_before_csi)

    bandwidth = 20 * int(
        pkt_len // (20 * 3.2 * 4)
    )

    return bandwidth



def __find_nsamples_max(pcap_filesize, nsub,nsamples_max):
    '''
        Returns an estimate for the maximum possible number
        of samples in the pcap file.

        The size of the pcap file is divided by the size of
        a packet to calculate the number of samples. However,
        some packets have a padding of a few bytes, so the value
        returned is slightly higher than the actual number of
        samples in the pcap file.
    '''

    # PCAP global header is 24 bytes
    # PCAP packet header is 12 bytes
    # Ethernet + IP + UDP headers are 46 bytes
    # Nexmon metadata is 18 bytes
    # CSI is nsub*4 bytes long
    #
    # So each packet is 12 + 46 + 18 + nsub*4 bytes long
    nsamples_max_int = int(
        (pcap_filesize - 24) / (
            12 + 46 + 18 + (nsub*4)
        )
    )
    if nsamples_max == 0:
        return nsamples_max_int
    else:
        return min(nsamples_max_int,nsamples_max)

def read_pcap(pcap_filepath, bandwidth=0, nsamples_max=0, quan_type = "percentile", q = 4, mask_dq = 0, cycle_mod = 1, sel_mod = 0):
    '''
        Reads CSI samples from
        a pcap file. A SampleSet
        object is returned.

        Bandwidth and maximum samples
        are inferred from the pcap file by
        default, but you can also set them explicitly.
    '''

    pcap_filesize = os.stat(pcap_filepath).st_size
    with open(pcap_filepath, 'rb') as pcapfile:
        fc = pcapfile.read()
    
    if bandwidth == 0:
        bandwidth = __find_bandwidth(
            # 32-36 is where the incl_len
            # bytes for the first frame are
            # located.
            # https://wiki.wireshark.org/Development/LibpcapFileFormat/
            fc[32:36]
        )
    # Number of OFDM sub-carriers
    # bandwidth = 160
    nsub = int(bandwidth * 3.2)

    
    nsamples_max = __find_nsamples_max(pcap_filesize, nsub,nsamples_max*cycle_mod)
    nsamples_max= int(nsamples_max/cycle_mod)

    # Preallocating memory
    ts_sec = np.empty(nsamples_max)
    ts_usec = np.empty(nsamples_max)
    time_diff_usec = np.empty(nsamples_max)
    # ts_usec = bytearray(nsamples_max * 4)
    rssi = bytearray(nsamples_max * 1)
    frame_ctl   = bytearray(nsamples_max * 1)
    mac  = bytearray(nsamples_max * 6)
    seq  = bytearray(nsamples_max * 2)
    css  = bytearray(nsamples_max * 2)
    csi  = bytearray(nsamples_max * nsub * 4)

    # Pointer to current location in file.
    # This is faster than using file.tell()
    # =24 to skip pcap global header
    ptr = 24

    nsamples = 0
    pkt_ptr = 0
    while ptr < pcap_filesize and (nsamples_max == 0 or nsamples<nsamples_max):
        # Read frame header
        # Skip over Eth, IP, UDP
        curr_ts_sec = int.from_bytes(
            fc[ptr: ptr+4],
            byteorder='little',
            signed=False
        )
        ptr += 4
        curr_ts_usec = int.from_bytes(
            fc[ptr: ptr + 4],
            byteorder='little',
            signed=False
        )
        ptr += 4
        frame_len = int.from_bytes(
            fc[ptr: ptr+4],
            byteorder='little',
            signed=False
        )

        orig_packet_len = int.from_bytes(
            fc[ptr+4: ptr+8],
            byteorder='little',
            signed=False
        )
        ptr += 50

        # 2 bytes: Magic Bytes               @ 0 - 1
        # 1 bytes: RSSI                      @2
        # 1 bytes: frame control             @3
        # 6 bytes: Source Mac ID             @ 4 - 9
        # 2 bytes: Sequence Number           @ 10 - 11
        # 2 bytes: Core and Spatial Stream   @ 12 - 13
        # 2 bytes: ChanSpec                  @ 14 - 15
        # 2 bytes: Chip Version              @ 16 - 17
        # nsub*4 bytes: CSI Data             @ 18 - 18 + nsub*4

        if (pkt_ptr % cycle_mod) == sel_mod:
            ts_sec[nsamples] = curr_ts_sec
            ts_usec[nsamples] = curr_ts_usec
            if nsamples == 0:
                time_diff_usec[nsamples] = 0
            else:
                time_diff_usec[nsamples] = (ts_sec[nsamples] - ts_sec[nsamples-1])*10**6 \
                                           + ts_usec[nsamples]-ts_usec[nsamples-1]
            rssi[nsamples*1:(nsamples+1)*1] = fc[ptr+2: ptr+3]
            frame_ctl[nsamples*1:  (nsamples+1)*1] = fc[ptr+3: ptr+4]
            mac[nsamples*6: (nsamples+1)*6] = fc[ptr+4: ptr+10]
            seq[nsamples*2: (nsamples+1)*2] = fc[ptr+10: ptr+12]
            css[nsamples*2: (nsamples+1)*2] = fc[ptr+12: ptr+14]
            csi[nsamples*(nsub*4): (nsamples+1)*(nsub*4)] = fc[ptr+18: ptr+18 + nsub*4]
            nsamples += 1

        ptr += (frame_len - 42)
        pkt_ptr += 1

    # Convert CSI bytes to numpy array
    csi_np = np.frombuffer(
        csi,
        dtype = np.int16,
        count = nsub * 2 * nsamples
    )

    # Cast numpy 1-d array to matrix
    csi_np = csi_np.reshape((nsamples, nsub * 2))

    # Convert csi into complex numbers
    csi_cmplx = np.fft.fftshift(
            csi_np[:nsamples, ::2] + 1.j * csi_np[:nsamples, 1::2], axes=(1,)
    )
    samples = SampleSet(
        (rssi,
        frame_ctl,
        mac,
        seq,
        css,
        csi_cmplx),
        bandwidth,
        quan_type,
        q,
        mask_dq,
        ts_sec,
        ts_usec,
        time_diff_usec
    )
    samples.generate_key()
    return samples

if __name__ == "__main__":
    samples = read_pcap('pcap_files/output-40.pcap')
