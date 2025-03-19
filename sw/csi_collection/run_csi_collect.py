import argparse
from scapy.all import *
from socket import socket
import socket
import Pyro5.api
import os

# known working channels: 6/20, 64/40, 36/40, 36/80 (157/80???)
setup_params = {
    "alice_x": 0,
    "alice_y": 0,
    "alice_z": 0,
    "bob_x": 0,
    "bob_y": 300,
    "bob_z": 100,
    "eve_x": 5,
    "eve_y": 0,
    "eve_z": 0,
    "alice_start_delay": 0.08,
    "bob_resp_delay": 0.06,
    "rotating_member": "None",
    "wifi_ch": 6,
    "wifi_bw": 20,
    "bob_eve_rx_timeout": 1,
    "failed_attempts_limit": 10
}

EVE_EN = True
PC_DEBUG_MODE = False
FAST_MODE = False

OUTPUT_FILE_PREFIX = 'csi_ctrl_summary'

if PC_DEBUG_MODE:
    FILTER = None
    IFACE = None
    SEND_PACKET_COMMAND = 'echo send packet'
    GET_CONFIG_COMMAND = "echo get config"
else:
    FILTER = 'dst port 5500'
    IFACE = 'wlan0'
    SEND_PACKET_COMMAND = "nexutil -Iwlan0 -s510"
    GET_CONFIG_COMMAND = "nexutil -k"

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-sa', '--stand_alone', dest='stand_alone', required=False, action = "store_true",
                            default=False, help='choose pyro or stand alone mode')
        parser.add_argument('-a', '--alice_mode', dest='alice_mode', required=False, action="store_true",
                            default=False, help='choose alice or bob mode.')
        parser.add_argument('-e', '--eve_mode', dest='eve_mode', required=False, action="store_true",
                            default=False, help='choose eve mode.')
        parser.add_argument('-b', '--bob_mode', dest='bob_mode', required=False, action="store_true",
                            default=False, help='choose bob mode.')
        parser.add_argument('-c', '--ctrl_mode', dest='ctrl_mode', required=False, action="store_true",
                             default = False, help='choose ctrl_mode')
        parser.add_argument('-n', '--num_packets', dest='num_packets', required=False, type=int,
                            default=300, help='num of packets to send')
        parser.add_argument('-man_ip', '--man_ip', dest='man_ip', required=False,
                            default=None, action="store_true", help='if set- use manual IP addresses from file')

        # self.parser.add_argument('-s', '--socket_dis', dest='socket_dis', required=False, action = "store_true",
        #                     default = False, help='choose alice or bob mode')
        
        return parser.parse_args()


def get_my_ip():
    if PC_DEBUG_MODE:
        my_ip = "127.0.0.1"
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        my_ip = s.getsockname()[0]
        print(my_ip)
        s.close()
    return my_ip


def get_my_hostname():
    return socket.gethostname()


def dict_to_csv(dict_inst,csv_filepath):
    import pandas as pd
    df = pd.DataFrame(dict_inst, index=[0])
    df.to_csv(csv_filepath, header=True)


@Pyro5.api.expose
class rkgCsiTa(object):
    def __init__(self,args):
        # reset counters:
        self.txPktCnt = 0
        self.rxPktCnt = 0
        self.rxPkt = None
        self.args = args
        self.device_mode_string = None
        self.opened_socket = None
        self.good_packets = PacketList()
        self.fast_mode = False

    def set_fast_mode(self, mode=False):
        self.fast_mode = mode

    def getSendAck(self):
        self.rxPkt = None
        try:
            self.rxPkt = sniff(iface = IFACE, filter = FILTER, count=1, prn=lambda x: x.summary(), timeout = self.bob_eve_rx_timeout+0.005)
        except:
            return 0
        if len(self.rxPkt.res) > 0:
            self.sendPkt()
            self.parsePacket(self.rxPkt)
            return 1
        else:
            print("Sniff timeout")

    def bobLoop(self, num_packets):
        while (self.rxPktCnt <= num_packets and self.txPktCnt <= 2 * num_packets):  # allow PER of 20%
            if self.getSendAck() == 0:
                break

    def try_connect(self):
        if self.opened_socket is None:
            self.prn = lambda x: x.summary()
            self.iface = resolve_iface(IFACE)
            self.L2socket = self.iface.l2listen()
            self.opened_socket = self.L2socket(type=ETH_P_ALL, iface=IFACE, filter=FILTER)


    def close_server(self):
        if self.opened_socket:
            self.opened_socket.close()
            self.opened_socket = None

    @Pyro5.server.oneway
    def onewayGetSendAck(self,count = 1):
        self.rxPkt = None
        if self.fast_mode:
            self.rxPkt = sniff(filter = FILTER, prn=self.prn, count=count, timeout = self.bob_eve_rx_timeout, opened_socket = self.opened_socket)
        else:
            self.rxPkt = sniff(iface=IFACE, filter=FILTER, count=count, prn=lambda x: x.summary(), timeout=self.bob_eve_rx_timeout)
        if self.rxPkt is None or len(self.rxPkt.res) != count:
            print(f"onewayGetSendAck got {len(self.rxPkt.res)} pakcets instead of {count}")
            return False
        return True

    def get_num_rx_packets(self):
        if self.rxPkt is None:
            return 0
        else:
            return len(self.rxPkt.res)

    def clean_buffer(self):
        t = AsyncSniffer(filter=FILTER, prn=lambda x: x.summary(), opened_socket=self.opened_socket)
        t.start()
        time.sleep(0.1)
        t.stop()
        self.rxPkt = t.results

    def sendGetAck(self):
        self.rxPkt = None
        if self.fast_mode:
            t = AsyncSniffer(filter = FILTER,prn=lambda x: x.summary(), opened_socket=self.opened_socket)
        else:
            t = AsyncSniffer(iface=IFACE, filter=FILTER, prn=lambda x: x.summary())
        t.start()
        time.sleep(self.alice_start_delay)
        self.sendPkt()
        time.sleep(self.bob_resp_delay)
        t.stop()
        self.rxPkt = t.results
        if self.rxPkt is None or len(self.rxPkt) == 0:
            print("TX packet number {} - no ack".format(self.txPktCnt))
            return False
        else:
            return True
            
    def parseRxPackets(self, max_packets = 1):
        return self.parsePacket(self.rxPkt, max_packets=max_packets)
    
    def aliceLoop(self,num_packets):
        while (self.rxPktCnt <= num_packets and self.txPktCnt <= 2*num_packets): #allow PER of 20% 
             self.sendGetAck()

    def check_packets_delay(self,max_delay):
        if len(self.rxPkt) != 2:
            raise ValueError("not enouth packets to check diff")
        else:
            return True
            # print(self.rxPkt)
            # print(self.rxPkt[0])
            # print(self.rxPkt[1])
            # print(self.rxPkt[1].locals())



    def parsePacket(self,packets, max_packets=1):
        if packets is None:
            return False
        pktNum = len(packets)
        if  pktNum > max_packets:
            raise ValueError("Got {} packets, which is above max limit {}".format(pktNum,max_packets))
        if pktNum == 0:
            return False
        else:
            print("Got {} packets to parse".format(pktNum))
            self.rxPktCnt += pktNum

            for i in range(len(packets)):
                # self.good_packets.append(packets[i])
                wrpcap(self.outfilename, packets[i], append=True)
                print("wrote packet number {}".format(self.rxPktCnt))
            # wrpcap(self.outfilename, packets[pktNum-1], append=True)
            return True


    def sendPkt(self):
        self.txPktCnt+=1
        print("Sending packet number {}".format(self.txPktCnt))
        subprocess.run(SEND_PACKET_COMMAND,shell=True)

    def get_phy_config(self):
        output = subprocess.run(GET_CONFIG_COMMAND, shell=True, capture_output=True)
        return output.stdout.decode("utf-8")

    def nexmon_phy_config(self, channel=6, bandwidth=20, mac = "00:11:22:33:44:55"):
        cmd = f"~pi/nexmon/nexmon/patches/bcm43455c0/7_45_189/nexmon_csi/utils/makecsiparams/makecsiparams -c {channel}/{bandwidth} -C 1 -N 1 -m {mac}"
        output = subprocess.run(cmd, shell=True, capture_output=True)
        print(output)
        nexmon_param = str(output.stdout.decode("utf-8")).strip()
        cmd = f"nexutil -Iwlan0 -s500 -b -l34 -v {nexmon_param}"
        output = subprocess.run(cmd, shell=True, capture_output=True)
        print(output)
        cmd = f"nexutil -Iwlan0 -s510 -b -l34 -v {nexmon_param}"
        output = subprocess.run(cmd, shell=True, capture_output=True)
        print(output)

    def config(self, alice_start_delay=None, bob_resp_delay=None,start_time_str=None, bob_eve_rx_timeout=None):
        if alice_start_delay is None:
            self.alice_start_delay = setup_params['alice_start_delay']
        else:
            self.alice_start_delay = alice_start_delay

        if bob_resp_delay is None:
            self.bob_resp_delay = setup_params['bob_resp_delay']
        else:
            self.bob_resp_delay = bob_resp_delay

        if bob_eve_rx_timeout is None:
            self.bob_eve_rx_timeout = setup_params['bob_eve_rx_timeout']
        else:
            self.bob_eve_rx_timeout = bob_eve_rx_timeout

        if start_time_str is None:
            self.start_time_str = datetime.now().strftime("D%Y%m%d_T%H%M")
        else: #sync time for filenames..
            self.start_time_str = start_time_str
        if self.args.alice_mode:
            self.device_mode_string = 'alice'
        elif args.bob_mode:
            self.device_mode_string = 'bob'
        elif args.eve_mode:
            self.device_mode_string = 'eve'
        else:
            raise ValueError("run mode was not set! please set one of alice or bob or eve mode")
        self.outDir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"outputs")
        self.outfilename = os.path.join(self.outDir,"{}_{}_{}.pcap".format(OUTPUT_FILE_PREFIX,self.start_time_str,
                                                                           self.device_mode_string))

    def get_server_params(self):
        my_params = {}
        my_params[self.device_mode_string + "_hostname"] = get_my_hostname()
        my_params[self.device_mode_string + "_ip"] = get_my_ip()
        my_params[self.device_mode_string + "_filename"] = self.outfilename
        my_params[self.device_mode_string + "_chan_spec"] = self.get_phy_config()
        return my_params

    def runStandAlone(self):
        print("Stand alone mode is not supported anymore, use only if you know what you are doing !!!!!!!!!!!!!!")
        self.config(alice_start_delay=setup_params['alice_start_delay'], bob_resp_delay=setup_params['bob_resp_delay'],
                    bob_eve_rx_timeout=setup_params['bob_eve_rx_timeout'] )
        if self.args.alice_mode:
            self.aliceLoop(self.args.num_packets)
        else:
            self.bobLoop(self.args.num_packets)

    def write_to_pcap(self):
        for i in range(len(self.good_packets)):
            wrpcap(self.outfilename, self.good_packets[i], append=True)

def connect_to_server(server):
    return server.get_num_rx_packets()

def runController(num_packets):
    from threading import Thread
    setup_params.update({"req_num_packets": num_packets})
    all_servers = []
    # call config in order to create new PCAP file:
    start_time = time.time()
    start_time_str = datetime.now().strftime("D%Y%m%d_T%H%M")
    outDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs")
    outfilename = os.path.join(outDir, "{}_{}_{}.csv".format(OUTPUT_FILE_PREFIX, start_time_str, "ctrl"))
    aliceServer = Pyro5.api.Proxy("PYRONAME:alicerkgCsiTa")
    all_servers.append(aliceServer)
    bobServer = Pyro5.api.Proxy("PYRONAME:bobrkgCsiTa")
    all_servers.append(bobServer)
    if EVE_EN:
        eveServer = Pyro5.api.Proxy("PYRONAME:everkgCsiTa")
        all_servers.append(eveServer)
    else:
        eveServer = None

    #configure servers:
    for server in all_servers:
        server.config(start_time_str=start_time_str, alice_start_delay=setup_params['alice_start_delay'],
                      bob_resp_delay=setup_params['bob_resp_delay'])
        server.nexmon_phy_config(channel=setup_params['wifi_ch'], bandwidth=setup_params['wifi_bw'], mac = "00:11:22:33:44:55")
        server_params = server.get_server_params()
        setup_params.update(server_params)
        server.set_fast_mode(FAST_MODE)
        if FAST_MODE:
            server.try_connect()

    for server in all_servers:
        if FAST_MODE:
            #After setting up nexmon.. prevent spare packets:
            server.clean_buffer()

    pktNum = 0
    failed_attempts = 0
    failed_attempts_total = 0
    #Run loop:
    try:
        start_collect_time = time.time()
        start_collect_time_str = datetime.now().strftime("D%Y%m%d_T%H%M")
        while (pktNum<num_packets):
            bobServer.onewayGetSendAck()
            if eveServer:
                eveServer.onewayGetSendAck(count=2) #Listen only...
            try:
                success = aliceServer.sendGetAck() \
                          and aliceServer.get_num_rx_packets() == bobServer.get_num_rx_packets() == 1
                if eveServer:
                    success = success and eveServer.get_num_rx_packets() == 2 and eveServer.check_packets_delay(0.5)
            except Exception as e:
                print("Exception occoured " + str(e))
                success = 0
            if success:
                failed_attempts_total += failed_attempts
                failed_attempts = 0
                pktNum += 1
                print("packet number {} succeed".format(pktNum))
                if not bobServer.parseRxPackets(max_packets=1):
                    raise ValueError("Bob failed to parse packet nummber {}".format(pktNum))
                if not aliceServer.parseRxPackets(max_packets=1):
                    raise ValueError("Alice failed to parse packet nummber {}".format(pktNum))
                if eveServer:
                    if not eveServer.parseRxPackets(max_packets=2):
                        print("Eve failed to parse packet nummber {}".format(pktNum))
            else:
                failed_attempts += 1
                print("packet number {} failed, current failed attempt {}, total failed attempts till now {} "
                      .format(pktNum+1, failed_attempts, failed_attempts_total))
                time.sleep(0.1)
                if failed_attempts > setup_params['failed_attempts_limit']:
                    raise ValueError(f"Reached maximum failed attempts of {setup_params['failed_attempts_limit']}")
    except Exception as e:
        print(f"Run loop got the following exception {e}")
    finally:
        end_time = time.time()
        collect_duration = end_time - start_collect_time
        total_duration = end_time - start_time
        end_time_str = datetime.now().strftime("D%Y%m%d_T%H%M")
        setup_params.update({"num_packets": pktNum})
        setup_params.update({"failed_attempts_total": failed_attempts_total})
        setup_params.update({"start_time": start_time_str})
        setup_params.update({"start_collect_time": start_collect_time_str})
        setup_params.update({"end_time": end_time_str})
        setup_params.update({"collect_duration": collect_duration})
        setup_params.update({"total_duration": total_duration})
        dict_to_csv(setup_params, outfilename)
        for server in all_servers:
            # After setting up nexmon.. prevent spare packets:
            server.write_to_pcap()
            server.close_server()
        print("CSI collection ended. Summary:")
        print(setup_params)


if __name__ == "__main__":
    
    args = parse_args()
    ns = Pyro5.api.locate_ns()             # find the name server
    
    if args.ctrl_mode:
        runController(args.num_packets)
        
    else:
        rkgCsiTaInst = rkgCsiTa(args)
        if args.stand_alone:
            rkgCsiTa.run()
        else:
            if args.man_ip is not None:
                myIp = args.man_ip
            else:
                myIp = get_my_ip()
            if args.alice_mode:
                instName = "alice"
            elif args.bob_mode:
                instName = "bob"
            elif args.eve_mode:
                instName = "eve"
            else:
                instName = ""
                raise ValueError("Server mode not selected! Exiting.")
        daemon = Pyro5.server.Daemon(host=myIp)         # make a Pyro daemon
        uri = daemon.register(rkgCsiTaInst)   # register the greeting maker as a Pyro object
        ns.register("{}rkgCsiTa".format(instName), uri)   # register the object with a name in the name server
    
        print("Ready. uri = {}".format(uri))
        daemon.requestLoop()                   # start the event loop of the server to wait for calls
    
    
        