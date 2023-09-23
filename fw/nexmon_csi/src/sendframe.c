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
 * This file is part of NexMon.                                            *
 *                                                                         *
 * Copyright (c) 2016 NexMon Team                                          *
 *                                                                         *
 * NexMon is free software: you can redistribute it and/or modify          *
 * it under the terms of the GNU General Public License as published by    *
 * the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                     *
 *                                                                         *
 * NexMon is distributed in the hope that it will be useful,               *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 * GNU General Public License for more details.                            *
 *                                                                         *
 * You should have received a copy of the GNU General Public License       *
 * along with NexMon. If not, see <http://www.gnu.org/licenses/>.          *
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
#include <capabilities.h>       // capabilities included in a nexmon patch
#include <sendframe.h>
#include <channels.h>

char packet_bytes[] = {
		0x88, 0x42, 0x2c, 0x00,
		0x00, 0x11, 0x22, 0x33, 0x44, 0x55, // dest
		0x00, 0x11, 0x22, 0x33, 0x44, 0x55, // transmitter
		0x00, 0x11, 0x22, 0x33, 0x44, 0x55, // src
		0x20, 0x21, 0x00, 0x00, 0x46, 0x09, 0x00, 0x20, 0x00, 0x00,
		0x00, 0x00,
		0x32, 0xf5, 0x22, 0xdf, 0xf1, 0xb2, 0xf5, 0x9b,
		0x19, 0x20, 0x0e, 0x56, 0x9e, 0x27, 0xac, 0x7c,
		0x6c, 0xb0, 0xca, 0x4b, 0x56, 0x10, 0x10, 0x51,
		0x8e, 0xe2, 0x19, 0x75, 0x4f, 0x80, 0x44, 0x7d,
		0x87, 0x73, 0xc1, 0x0e, 0x2f, 0xf5, 0x2e, 0x7c,
		0xdc, 0x05, 0xba, 0x91, 0x3e, 0xe0, 0x94, 0xd3,
		0x82, 0x2a, 0x25, 0x3c, 0xe1, 0xbb, 0xb4, 0xef,
		0x83, 0x60, 0xef, 0x3e, 0xf0, 0x79
};

char ack_packet_bytes[] = {
		0xd4, 0x00, 0x00, 0x00, 0x00, 0x11, 0x22, 0x33,
		0xee, 0xee
};

void
send_packet(struct wlc_info *wlc)
{
	int len = sizeof(packet_bytes);
	
	unsigned int rate = RATES_OVERRIDE_MODE | RATES_LDPC_CODING;
	
	unsigned short chanspec = get_chanspec(wlc);
	
	switch (chanspec & WL_CHANSPEC_BW_MASK) 
	{
				case (WL_CHANSPEC_BW_20):
	                                			{
					rate |= RATES_BW_20MHZ | RATES_ENCODE_HT | RATES_HT_MCS(0);
	                                			}
				break;
	
				case (WL_CHANSPEC_BW_40):
	                                			{
					rate |= RATES_BW_40MHZ | RATES_ENCODE_HT | RATES_HT_MCS(0);
	                                			}
				break;
	
				case (WL_CHANSPEC_BW_80):
	                                			{
					rate |= RATES_BW_80MHZ | RATES_ENCODE_VHT | RATES_VHT_MCS(0) | RATES_VHT_NSS(1) | RATES_RATE_6M ;
	                                			}
				case (WL_CHANSPEC_BW_160):
				                                			{
					rate |= RATES_BW_160MHZ | RATES_ENCODE_VHT | RATES_VHT_MCS(0) | RATES_VHT_NSS(1) | RATES_RATE_6M;
				                                			}
				break;
				}
			
	

	//Can it be located here in order to support Slave as well?
	set_intioctl(wlc, WLC_SET_LRL, 1);
	set_intioctl(wlc, WLC_SET_SRL, 1);

	//20MHz:
//	unsigned int rate = RATES_OVERRIDE_MODE | RATES_ENCODE_HT | RATES_BW_40MHZ | RATES_HT_MCS(0) | RATES_LDPC_CODING;
	//80MHz:
//	unsigned int rate = TOMER_RATE;

	sk_buff *p = pkt_buf_get_skb(wlc->osh, len + 202);
	char *packet_skb;
	packet_skb = (char *) skb_pull(p, 202);
	memcpy(packet_skb, &packet_bytes, len);



	packet_skb[38] = wlc->osh->pktalloced & 0xFF;
	packet_skb[37] = (wlc->osh->pktalloced >>8) & 0xFF;
	packet_skb[36] = (wlc->osh->pktalloced >> 16) & 0xFF;

	sendframe(wlc, p, 1, rate);
//	uint32 rate = RATES_RATE_6M;
	//    sendframe_with_timer(wlc, p, 1, rate, 0, 10, 1000);
}


char
sendframe(struct wlc_info *wlc, struct sk_buff *p, unsigned int fifo, unsigned int rate)
{
	char ret;

	 // this unmutes the currently used channel and allows to send on "quiet/passive" channels
	 wlc_bmac_mute(wlc->hw, 0, 0);
	    
	if (wlc->band->bandtype == WLC_BAND_5G && rate < RATES_RATE_6M) {
		rate = RATES_RATE_6M;
	}

	if (wlc->hw->up) {
		ret = wlc_sendctl(wlc, p, wlc->active_queue, wlc->band->hwrs_scb, fifo, rate, 0);
	} else {
		ret = wlc_sendctl(wlc, p, wlc->active_queue, wlc->band->hwrs_scb, fifo, rate, 1);
		printf("ERR: wlc down\n");
	}

	return ret;
}




static void
sendframe_copy(struct tx_task *task)
{
	// first, we create a copy copy of the frame that should be transmitted
	struct sk_buff *p_copy = pkt_buf_get_skb(task->wlc->osh, task->p->len + 202);
	if (!p_copy) return;

	skb_pull(p_copy, 202);
	memcpy(p_copy->data, task->p->data, task->p->len);
	p_copy->flags = task->p->flags;
	//    p_copy->scb = task->p->scb; //Tomer - removed due to compilation errors

	sendframe(task->wlc, p_copy, task->fifo, task->rate);

	if (task->txrepetitions > 0) {
		task->txrepetitions--;
	}
}

static void
sendframe_timer_handler(struct hndrte_timer *t)
{
	struct tx_task *task = (struct tx_task *) t->data;

	if (task->txrepetitions == 0) {
		// there must have been a mistake, just delete the frame task and timer
		pkt_buf_free_skb(task->wlc->osh, task->p, 0);
		goto free_timer_and_task;
	} else if (task->txrepetitions == 1) {
		// transmit the last frame
		sendframe(task->wlc, task->p, task->fifo, task->rate);
		free_timer_and_task:
		//hndrte_del_timer(t); //Tomer - removed due to linker error
		hndrte_free_timer(t);
		free(task);
	} else {
		sendframe_copy(task);
	}
}

static void
sendframe_repeatedly(struct tx_task *task)
{
	struct hndrte_timer *t;

	sendframe_copy(task);
	if (task->txrepetitions == 0)
		return;

	t = hndrte_init_timer(sendframe_repeatedly, task, sendframe_timer_handler, 0);

	if (!t) {
		free(task);
		return;
	}

	if (!hndrte_add_timer(t, task->txperiodicity, 1)) {
		hndrte_free_timer(t);
		free(task);

		printf("ERR: could not add timer");
	}
}

/**
 *  Is scheduled to transmit a frame after a delay
 */
static void
sendframe_task_handler(struct hndrte_timer *t)
{
	struct tx_task *task = (struct tx_task *) t->data;

	if (task->txrepetitions != 0 && task->txperiodicity > 0) {
		sendframe_repeatedly(task);
	} else {
		sendframe(task->wlc, task->p, task->fifo, task->rate);
		free(task);
	}
}

struct tx_task *
sendframe_with_timer(struct wlc_info *wlc, struct sk_buff *p, unsigned int fifo, unsigned int rate, int txdelay, int txrepetitions, int txperiodicity)
{
	struct tx_task *task = 0;

	// if we need to send the frame with a delay or repeatedly, we create a task
	if (txdelay > 0 || (txrepetitions != 0 && txperiodicity > 0)) {
		task = (struct tx_task *) malloc(sizeof(struct tx_task), 0);
		memset(task, 0, sizeof(struct tx_task)); // will be freed after finishing the task
		task->wlc = wlc;
		task->p = p;
		task->fifo = fifo;
		task->rate = rate;
		task->txrepetitions = txrepetitions;
		task->txperiodicity = txperiodicity;
	}

	if (txdelay > 0) {
		hndrte_schedule_work(sendframe_with_timer, task, sendframe_task_handler, txdelay);
	} else if (txrepetitions != 0 && txperiodicity > 0) {
		sendframe_repeatedly(task);
	} else {
		sendframe(wlc, p, fifo, rate);
	}

	return task;
}

