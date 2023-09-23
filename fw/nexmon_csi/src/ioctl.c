/***************************************************************************
 *                                                                         *
 *          ###########   ###########   ##########    ##########           *
 *         ############  ############  ############  ############          *
 *         ##            ##            ##   ##   ##  ##        ##          *
 *         ##            ##            ##   ##   ##  ##        ##          *
 *         ###########   ####  ######  ##   ##   ##  ##    ######          *
 *          ###########  ####  #       ##   ##   ##  ##    #    #          *
 *                   ##  ##    ######  ##   ##   ##  ##    #    #          *
 *                   ##  ##    #       ##   ##   ##  ##    #    #          *
 *         ############  ##### ######  ##   ##   ##  ##### ######          *
 *         ###########    ###########  ##   ##   ##   ##########           *
 *                                                                         *
 *            S E C U R E   M O B I L E   N E T W O R K I N G              *
 *                                                                         *
 * Copyright (c) 2019 Matthias Schulz                                      *
 *                                                                         *
 * Permission is hereby granted, free of charge, to any person obtaining a *
 * copy of this software and associated documentation files (the           *
 * "Software"), to deal in the Software without restriction, including     *
 * without limitation the rights to use, copy, modify, merge, publish,     *
 * distribute, sublicense, and/or sell copies of the Software, and to      *
 * permit persons to whom the Software is furnished to do so, subject to   *
 * the following conditions:                                               *
 *                                                                         *
 * 1. The above copyright notice and this permission notice shall be       *
 *    include in all copies or substantial portions of the Software.       *
 *                                                                         *
 * 2. Any use of the Software which results in an academic publication or  *
 *    other publication which includes a bibliography must include         *
 *    citations to the nexmon project a) and the paper cited under b):     *
 *                                                                         *
 *    a) "Matthias Schulz, Daniel Wegemer and Matthias Hollick. Nexmon:    *
 *        The C-based Firmware Patching Framework. https://nexmon.org"     *
 *                                                                         *
 *    b) "Francesco Gringoli, Matthias Schulz, Jakob Link, and Matthias    *
 *        Hollick. Free Your CSI: A Channel State Information Extraction   *
 *        Platform For Modern Wi-Fi Chipsets. Accepted to appear in        *
 *        Proceedings of the 13th Workshop on Wireless Network Testbeds,   *
 *        Experimental evaluation & CHaracterization (WiNTECH 2019),       *
 *        October 2019."                                                   *
 *                                                                         *
 * 3. The Software is not used by, in cooperation with, or on behalf of    *
 *    any armed forces, intelligence agencies, reconnaissance agencies,    *
 *    defense agencies, offense agencies or any supplier, contractor, or   *
 *    research associated.                                                 *
 *                                                                         *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS *
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF              *
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  *
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY    *
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,    *
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE       *
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                  *
 *                                                                         *
 **************************************************************************/

#pragma NEXMON targetregion "patch"

#include <firmware_version.h>   // definition of firmware version macros
#include <debug.h>              // contains macros to access the debug hardware
#include <wrapper.h>            // wrapper definitions for functions that already exist in the firmware
#include <structs.h>            // structures that are used by the code in the firmware
#include <helper.h>             // useful helper functions
#include <patcher.h>            // macros used to craete patches such as BLPatch, BPatch, ...
#include <rates.h>              // rates used to build the ratespec for frame injection
#include <nexioctls.h>          // ioctls added in the nexmon patch
#include <version.h>            // version information
#include <argprintf.h>          // allows to execute argprintf to print into the arg buffer
#include <objmem.h>
#include <sendframe.h>
#include <channels.h>
#if NEXMON_CHIP == CHIP_VER_BCM4366c0
#define SHM_CSI_COLLECT         0xB80
#define NSSMASK                 0xB81
#define COREMASK                0xB82
#define APPLY_PKT_FILTER        0xB83
#define PKT_FILTER_BYTE         0xB84
#define N_CMP_SRC_MAC           0xB85
#define CMP_SRC_MAC_0_0         0xB86
#define CMP_SRC_MAC_0_1         0xB87
#define CMP_SRC_MAC_0_2         0xB88
#define CMP_SRC_MAC_1_0         0xB89
#define CMP_SRC_MAC_1_1         0xB8A
#define CMP_SRC_MAC_1_2         0xB8B
#define CMP_SRC_MAC_2_0         0xB8C
#define CMP_SRC_MAC_2_1         0xB8D
#define CMP_SRC_MAC_2_2         0xB8E
#define CMP_SRC_MAC_3_0         0xB8F
#define CMP_SRC_MAC_3_1         0xB90
#define CMP_SRC_MAC_3_2         0xB91
#define FORCEDEAF               0xB92
#define CLEANDEAF               0xB93
#define FIFODELAY               0xB94
#else
#define N_CMP_SRC_MAC           0x888
#define CMP_SRC_MAC_0_0         0x889
#define CMP_SRC_MAC_0_1         0x88A
#define CMP_SRC_MAC_0_2         0x88B
#define CMP_SRC_MAC_1_0         0x88C
#define CMP_SRC_MAC_1_1         0x88D
#define CMP_SRC_MAC_1_2         0x88E
#define CMP_SRC_MAC_2_0         0x88F
#define CMP_SRC_MAC_2_1         0x890
#define CMP_SRC_MAC_2_2         0x891
#define CMP_SRC_MAC_3_0         0x892
#define CMP_SRC_MAC_3_1         0x893
#define CMP_SRC_MAC_3_2         0x894
#define APPLY_PKT_FILTER        0x898
#define PKT_FILTER_BYTE         0x899
#define FIFODELAY               0x89e
#define SHM_CSI_COLLECT         0x8b0
#define	CLEANDEAF               0x8a3
#define	FORCEDEAF               0x8a4
#define NSSMASK                 0x8a6
#define COREMASK                0x8a7
#endif

struct udpstream {
	uint8 id;
	uint8 power;
	uint16 fps;
	uint16 destPort;
	uint8 modulation;
	uint8 rate;
	uint8 bandwidth;
	uint8 ldpc;
} __attribute__((packed));

//struct jamming_settings {
//    uint16 idftSize;
//    uint16 port;
//    int16 numActiveSubcarriers;
//    uint8 jammingType;
//    uint16 jammingSignalRepetitions;
//    int8 power;
//    cint16ap freqDomSamps[];
//} __attribute__((packed));

static struct tx_task *tx_task_list[10] = {0};

struct wlandata_header {
	uint8 ver_type_subtype;
	uint8 flags;
	uint16 duration;
	uint8 dst_addr[6];
	uint8 src_addr[6];
	uint8 bssid[6];
	uint16 frag_seq_num;
	uint16 qos_ctrl;
	uint8 llc_dsap;
	uint8 llc_ssap;
	uint8 llc_ctrl;
	uint8 llc_org_code[3];
	uint16 llc_type;
} __attribute__((packed));

struct wlandata_ipv4_udp_header {
	struct wlandata_header wlan;
	struct ip_header ip;
	struct udp_header udp;
	uint8 payload[];
} __attribute__((packed));

struct wlandata_ipv4_udp_header wlandata_ipv4_udp_header = {
		.wlan = {
				.ver_type_subtype = 0x88,
				.flags = 0x00,
				.duration = 0x013a,
				.dst_addr = { 'N', 'E', 'X', 'M', 'O', 'N' },
				.src_addr = { 'J', 'A', 'M', 'M', 'E', 'R' },
				.bssid = { 'D', 'E', 'M', 'O', 0, 0 },
				.frag_seq_num = 0x0000,
				.qos_ctrl = 0x0000,
				.llc_dsap = 0xaa,
				.llc_ssap = 0xaa,
				.llc_ctrl = 0x03,
				.llc_org_code = { 0x00, 0x00, 0x00 },
				.llc_type = 0x0008,
		},
		.ip = {
				.version_ihl = 0x45,
				.dscp_ecn = 0x00,
				.total_length = 0x0000,
				.identification = 0x0100,
				.flags_fragment_offset = 0x0000,
				.ttl = 0x01,
				.protocol = 0x11,
				.header_checksum = 0x0000,
				.src_ip.array = { 10, 10, 10, 10 },
				.dst_ip.array = { 255, 255, 255, 255 }
		},
		.udp = {
				.src_port = HTONS(5500),
				.dst_port = HTONS(5500),
				.len_chk_cov.length = 0x0000,
				.checksum = 0x0000
		},
};


/**
 * Calculates the IPv4 header checksum given the total IPv4 packet length.
 *
 * This checksum is specific to the packet format above. This is not a full
 * implementation of the checksum algorithm. Instead, as much as possible is
 * precalculated to reduce the amount of computation needed. This calculation
 * is accurate for total lengths up to 42457.
 */
static inline uint16_t
calc_checksum(uint16_t total_len)
{
	return ~(23078 + total_len);
}

//void
//prepend_wlandata_ipv4_udp_header(struct sk_buff *p, uint16 destPort)
//{
//    wlandata_ipv4_udp_header.ip.total_length = htons(p->len + sizeof(struct ip_header) + sizeof(struct udp_header));
//    wlandata_ipv4_udp_header.ip.header_checksum = htons(calc_checksum(p->len + sizeof(struct ip_header) + sizeof(struct udp_header)));
//    wlandata_ipv4_udp_header.udp.len_chk_cov.length = htons(p->len + sizeof(struct udp_header));
//    wlandata_ipv4_udp_header.udp.src_port = htons(destPort);
//    wlandata_ipv4_udp_header.udp.dst_port = htons(destPort);
//
//    skb_push(p, sizeof(wlandata_ipv4_udp_header));
//    memcpy(p->data, &wlandata_ipv4_udp_header, sizeof(wlandata_ipv4_udp_header));
//}

static void
exp_set_gains_by_index_change_bbmult(struct phy_info *pi, int8 index, int8 bbmultmult)
{
	ac_txgain_setting_t gains = { 0 };
	wlc_phy_txpwrctrl_enable_acphy(pi, 0);
	wlc_phy_get_txgain_settings_by_index_acphy(pi, &gains, index);
	gains.bbmult *= bbmultmult;
	wlc_phy_txcal_txgain_cleanup_acphy(pi, &gains);
}

static void
set_power_index(struct phy_info *pi, int8 index) {
	if (index < 0) {
		wlc_phy_txpwrctrl_enable_acphy(pi, 1);
	} else {
		exp_set_gains_by_index_change_bbmult(pi, index, 1);
	}
}

/*Tomer - removed due to compilation errors
static uint8 *orig_etheraddr = 0;

static void
change_etheraddr(struct wlc_info *wlc, uint8 *newaddr) {
    if (orig_etheraddr == 0) {
        orig_etheraddr = malloc(4, 0);
        memcpy(orig_etheraddr, wlc->pub->cur_etheraddr, 6);
    }
    wlc_iovar_op(wlc, "cur_etheraddr", 0, 0, newaddr, 6, 1, 0);
}

static uint8 *
get_orig_etheraddr(struct wlc_info *wlc) {
    if (orig_etheraddr == 0) {
        orig_etheraddr = malloc(4, 0);
        memcpy(orig_etheraddr, wlc->pub->cur_etheraddr, 6);
    }

    return orig_etheraddr;
}
 */

int 
wlc_ioctl_hook(struct wlc_info *wlc, int cmd, char *arg, int len, void *wlc_if)
{
	int ret = IOCTL_ERROR;
	struct osl_info *osh = wlc->osh;
	struct phy_info *pi = wlc->hw->band->pi;
	argprintf_init(arg, len);

	switch(cmd) {
	case 500:   // set csi_collect
	{
		struct params {
			uint16 chanspec;            // chanspec to tune to
			uint8  csi_collect;         // trigger csi collect (1: on, 0: off)
			uint8  core_nss_mask;       // coremask and spatialstream mask
			uint8  use_pkt_filter;      // trigger first packet byte filter (1: on, 0: off)
			uint8  first_pkt_byte;      // first packet byte to filter for
			uint16 n_mac_addr;          // number of mac addresses to filter for (0: off, 1-4: on,use src_mac_0-3)
			uint16 cmp_src_mac_0_0;     // filter src mac 0
			uint16 cmp_src_mac_0_1;
			uint16 cmp_src_mac_0_2;
			uint16 cmp_src_mac_1_0;     // filter src mac 1
			uint16 cmp_src_mac_1_1;
			uint16 cmp_src_mac_1_2;
			uint16 cmp_src_mac_2_0;     // filter src mac 2
			uint16 cmp_src_mac_2_1;
			uint16 cmp_src_mac_2_2;
			uint16 cmp_src_mac_3_0;     // filter src mac 3
			uint16 cmp_src_mac_3_1;
			uint16 cmp_src_mac_3_2;
			uint16 delay;               // delay between extractions in us
		};
		struct params *params = (struct params *) arg;
		// deactivate scanning
		set_scansuppress(wlc, 1);
		// deactivate minimum power consumption
		set_mpc(wlc, 0);
		// set the channel
		set_chanspec(wlc, params->chanspec);
		//Set retransmission:
		set_intioctl(wlc, WLC_SET_LRL, 1);
		set_intioctl(wlc, WLC_SET_SRL, 1);

		// write shared memory
		if (wlc->hw->up && len > 1) {
			wlc_bmac_write_shm(wlc->hw, SHM_CSI_COLLECT * 2, params->csi_collect);
			wlc_bmac_write_shm(wlc->hw, NSSMASK * 2, ((params->core_nss_mask)&0xf0)>>4);
			wlc_bmac_write_shm(wlc->hw, COREMASK * 2, (params->core_nss_mask)&0x0f);
			wlc_bmac_write_shm(wlc->hw, N_CMP_SRC_MAC * 2, params->n_mac_addr);
			wlc_bmac_write_shm(wlc->hw, APPLY_PKT_FILTER * 2, params->use_pkt_filter);
			wlc_bmac_write_shm(wlc->hw, PKT_FILTER_BYTE * 2, params->first_pkt_byte);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_0_0 * 2, params->cmp_src_mac_0_0);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_0_1 * 2, params->cmp_src_mac_0_1);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_0_2 * 2, params->cmp_src_mac_0_2);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_1_0 * 2, params->cmp_src_mac_1_0);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_1_1 * 2, params->cmp_src_mac_1_1);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_1_2 * 2, params->cmp_src_mac_1_2);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_2_0 * 2, params->cmp_src_mac_2_0);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_2_1 * 2, params->cmp_src_mac_2_1);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_2_2 * 2, params->cmp_src_mac_2_2);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_3_0 * 2, params->cmp_src_mac_3_0);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_3_1 * 2, params->cmp_src_mac_3_1);
			wlc_bmac_write_shm(wlc->hw, CMP_SRC_MAC_3_2 * 2, params->cmp_src_mac_3_2);
			wlc_bmac_write_shm(wlc->hw, FIFODELAY * 2, params->delay);
			ret = IOCTL_SUCCESS;
		}
		break;
	}
	case 501:   // get csi collect
	{
		if (wlc->hw->up && len > 1) {
			*(uint16 *) arg = wlc_bmac_read_shm(wlc->hw, SHM_CSI_COLLECT * 2);
			ret = IOCTL_SUCCESS;
		}
		break;
	}
	case 502:	// force deaf mode
	{
		if (wlc->hw->up && len > 1) {
			wlc_bmac_write_shm(wlc->hw, FORCEDEAF * 2, 1);
			ret = IOCTL_SUCCESS;
		}
		break;
	}
	case 503:	// clean deaf mode
	{
		if (wlc->hw->up && len > 1) {
			wlc_bmac_write_shm(wlc->hw, CLEANDEAF * 2, 1);
			ret = IOCTL_SUCCESS;
		}
		break;
	}
	case NEX_READ_OBJMEM:
	{
		set_mpc(wlc, 0);
		if (wlc->hw->up && len >= 4) {
			int addr = ((int *) arg)[0];
			int i = 0;
			for (i = 0; i < len / 4; i++) {
				wlc_bmac_read_objmem32_objaddr(wlc->hw, addr + i, &((unsigned int *) arg)[i]);
			}
			ret = IOCTL_SUCCESS;
		}
		break;
	}

	case 510: // start stream according to parameters
	{

//		struct params {
//			uint16 chanspec;            // chanspec to tune to
//			uint8  csi_collect;         // trigger csi collect (1: on, 0: off)
//			uint8  core_nss_mask;       // coremask and spatialstream mask
//			uint8  use_pkt_filter;      // trigger first packet byte filter (1: on, 0: off)
//			uint8  first_pkt_byte;      // first packet byte to filter for
//			uint16 n_mac_addr;          // number of mac addresses to filter for (0: off, 1-4: on,use src_mac_0-3)
//			uint16 cmp_src_mac_0_0;     // filter src mac 0
//			uint16 cmp_src_mac_0_1;
//			uint16 cmp_src_mac_0_2;
//			uint16 cmp_src_mac_1_0;     // filter src mac 1
//			uint16 cmp_src_mac_1_1;
//			uint16 cmp_src_mac_1_2;
//			uint16 cmp_src_mac_2_0;     // filter src mac 2
//			uint16 cmp_src_mac_2_1;
//			uint16 cmp_src_mac_2_2;
//			uint16 cmp_src_mac_3_0;     // filter src mac 3
//			uint16 cmp_src_mac_3_1;
//			uint16 cmp_src_mac_3_2;
//			uint16 delay;               // delay between extractions in us
//		};
//		struct params *params = (struct params *) arg;
//		// deactivate scanning
//		set_scansuppress(wlc, 1);
//		// deactivate minimum power consumption
//		set_mpc(wlc, 0);
//		// set the channel
//		set_chanspec(wlc, params->chanspec);
//
//		// set the retransmission settings
//		set_intioctl(wlc, WLC_SET_LRL, 1);
//		set_intioctl(wlc, WLC_SET_SRL, 1);
//
//		struct udpstream upstream_local;
//		upstream_local.id = 0;
//		upstream_local.power=0; //not in use..
//		upstream_local.fps = 1;
//		upstream_local.destPort = 5555;
//		upstream_local.modulation = 2;
//		upstream_local.rate = 0;
//		upstream_local.bandwidth = 20;
//		upstream_local.ldpc = 0;


			//                        struct udpstream *config = (struct udpstream *) arg;
//			struct udpstream *config = (struct udpstream *) &upstream_local;
			//                        if (config->id >= ARRAYSIZE(tx_task_list)) {
			//                            printf("too many udp streams");
			//                            break;
			//                        }
			//
			//                        if (tx_task_list[config->id] != 0) {
			//                            // end the task automatically at its next execution
			//                            tx_task_list[config->id]->txrepetitions = 0;
			//                            tx_task_list[config->id] = 0;
			//                        }

			//                        set_chanspec(wlc, 0x1006);
			//                        set_scansuppress(wlc, 1);
			//                        set_mpc(wlc, 0);

			// deactivate the transmission of ampdus
			//                        wlc_ampdu_tx_set(wlc->ampdu_tx, 0); //TomerA- removed, compilation error



			// set mac address to "JAMMER"
			//wlc_iovar_op(wlc, "cur_etheraddr", 0, 0, "JAMMER", 6, 1, 0);

			//                        change_etheraddr(wlc, (uint8 *) "JAMMER"); //TOMER- removed due to compilation error

//			unsigned int fifo = 0;

//			int txdelay = 0;
//			int txrepetitions = -1;
//			int txperiodicity = 1000 / config->fps;

//			unsigned int rate = RATES_RATE_6M | RATES_BW_80MHZ;
//			// setting the rate spec here allows to activate LDPC in 802.11n frames
//			wlc->band->rspec_override = rate;
//
////			unsigned int rate = 0;
//			
//			upstream_local.rate = rate;
//			rate = RATES_RATE_6M;
//			switch (upstream_local.bandwidth)
//			{
//			case (20):
//                                			{
//				rate |= RATES_BW_20MHZ;
//                                			}
//			break;
//
//			case (40):
//                                			{
//				rate |= RATES_BW_40MHZ;
//                                			}
//			break;
//
//			case (80):
//                                			{
//				rate |= RATES_BW_80MHZ;
//                                			}
//			case (160):
//			                                			{
//					rate |= RATES_BW_160MHZ;
//			                                			}
//			break;
//			}


//			uint16 payload_length = 1000;

		//	struct sk_buff *p = pkt_buf_get_skb(osh, sizeof(wlandata_ipv4_udp_header) + payload_length + 202);
//			if (!p) break;

			// pull to have space for d11txhdrs
//			skb_pull(p, 202);

			// pull as prepend_wlandata_ipv4_udp_header pushes
//			skb_pull(p, sizeof(wlandata_ipv4_udp_header));

			//memset(p->data, 0x23, payload_length);
			//                        snprintf(p->data, payload_length,
			//                            "This frame is part of the \"Demonstrating Smartphone-based Jammers\" demo presented at ACM WiSec 2017. "
			//                            "It was transmitted by a Nexus 5 smartphone using Nexmon, the C-based firmware patching framework (https://nexmon.org).");


			//                        uint8 *macaddr = get_orig_etheraddr(wlc); //TOMER- removed due to compilation errors
			//                        memcpy(&wlandata_ipv4_udp_header.wlan.bssid[4], &macaddr[4], 2); //TOMER- removed due to compilation errors
			//                        prepend_wlandata_ipv4_udp_header(p, config->destPort); //TOMER- removed due to linker errors


//			wlc->band->rspec_override = ;
			//                        wlc_d11hdrs_ext(wlc, p, wlc->band->hwrs_scb, 0, 0, 1, 1, 0, 0, 0 /* data_rate */, 0); //TOMER- removed due to linker errors + probably not needed... See:
			//https://github.com/seemoo-lab/nexmon/pull/94
			//                        p->scb = wlc->band->hwrs_scb; //TOMER- removed due to compilation errors

			//                        tx_task_list[config->id] = sendframe_with_timer(wlc, p, fifo, rate, txdelay, txrepetitions, txperiodicity);
			send_packet(wlc);

			printf("%s: starting stream\n", __FUNCTION__);

			ret = IOCTL_SUCCESS;
		break;
	}

	default:
		ret = wlc_ioctl(wlc, cmd, arg, len, wlc_if);
	}

	return ret;
}





__attribute__((at(0x1F3488, "", CHIP_VER_BCM4339, FW_VER_6_37_32_RC23_34_43_r639704)))
__attribute__((at(0x20CD80, "", CHIP_VER_BCM43455c0, FW_VER_7_45_189)))
__attribute__((at(0x1F3230, "", CHIP_VER_BCM4358, FW_VER_7_112_300_14)))
__attribute__((at(0x2F0CF8, "", CHIP_VER_BCM4366c0, FW_VER_10_10_122_20)))
GenericPatch4(wlc_ioctl_hook, wlc_ioctl_hook + 1);
