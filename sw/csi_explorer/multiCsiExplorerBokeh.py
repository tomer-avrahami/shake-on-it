import os
import config
import time
import decoders.interleaved as decoder
from bokeh.plotting import figure, output_file, show, ColumnDataSource, save
from bokeh.models import Range1d, HoverTool, Panel, Tabs,  Div, RangeSlider, Spinner
from bokeh.layouts import column, row, layout
from bokeh.models import CustomJS, Slider, Div, Title
import numpy as np


DEFAULT_SC = -7
DEFAULT_PACKET = 99

def string_is_int(s):
    '''
    Check if a string is an integer
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False


def init_bokeh(bandwidth):
    global text_boxes
    global samples_dict
    global one_packet_figs
    global x_amp
    global x_packet
    global packet_multi_fig
    global color_bars
    global one_sc_figs
    global csi_val_vs_pkt_num_figs
    global error_quan_vs_pkt_num_figs
    global alice_bob_time_diff_vs_pkt_num_fig
    global x_range_all_packets
    global x_range_all_subs
    global y_range_csi_amp
    global euc_error_vs_pkt_num_figs
    global ham_error_vs_pkt_num_figs

    total_width = 1200
    one_width = int(total_width/4)
    height = 500
    text_boxes = {}
    one_packet_figs = {}
    one_sc_figs = {}
    error_quan_vs_pkt_num_figs = {}
    euc_error_vs_pkt_num_figs = {}
    ham_error_vs_pkt_num_figs = {}
    console_output_lines = 10
    nsub = int(bandwidth * 3.2)
    x_amp = np.arange(-1 * nsub / 2, nsub / 2)
    x_packet = np.arange(0, 300)
    style_dict = {'overflow-y': 'scroll', 'height': '{0}px'.format(20 * console_output_lines)}
    packet_multi_fig = figure(title=f"CSI vals of selected packet, all subcarriers", x_axis_label="Sub-carrier index", y_axis_label="CSI amplitude",plot_width=total_width, plot_height=height)
    packet_multi_fig.add_tools(HoverTool(tooltips=[("(x,y)", "$x{0.2f}, $y{0.2f}")]))
    csi_val_vs_pkt_num_figs = figure(title=f"CSI amplitude vs packet number", x_axis_label="packet number", y_axis_label="CSI amplitude", plot_width=total_width,
                              plot_height=height)
    csi_val_vs_pkt_num_figs.add_tools(HoverTool(tooltips=[("(x,y)", "$x{0.2f}, $y{0.2f}")]))
    csi_val_vs_pkt_num_figs.legend.click_policy="hide"



    alice_bob_time_diff_vs_pkt_num_fig = figure(title=f"time diff usec", x_axis_label="packet number", y_axis_label="time diff[us]", plot_width=total_width,
                           plot_height=height)
    alice_bob_time_diff_vs_pkt_num_fig.add_tools(HoverTool(tooltips=[("(x,y)", "$x{0.2f}, $y{0.2f}")]))
    alice_bob_time_diff_vs_pkt_num_fig.legend.click_policy = "hide"


    color_bars = {}
    color_bars['alice'] = 'green'
    color_bars['bob'] = 'blue'
    color_bars['eve_alice'] = 'orange'
    color_bars['eve_bob'] = 'red'
    color_bars[0] = 'green'
    color_bars[1] = 'blue'
    color_bars[2] = 'orange'
    color_bars[3] = 'red'

    x_range_all_subs = Range1d(-1 * nsub / 2, nsub / 2)
    x_range_all_packets = Range1d(0, 300)
    y_range_csi_amp = Range1d(0, 3.5)
    y_range_csi_amp_error = Range1d(-2, 2)
    y_range_csi_amp_quan = Range1d(0, 4)


    for name in ['alice','bob','eve_alice', 'eve_bob']:
        text_boxes[name] = Div(text=name, width=one_width, height=int(height/2), style=style_dict)
        #one packet- all subcarriers
        one_packet_figs[name] = figure(title=f"{name} CSI vals\n Selected packet, all subcarreiers ", x_axis_label="sc", y_axis_label="CSI",
                                   plot_width=one_width, plot_height=height)
        one_packet_figs[name].add_tools(HoverTool(tooltips=[("(x,y)", "$x{0.2f}, $y{0.2f}")]))
        one_packet_figs[name].x_range = x_range_all_subs
        one_packet_figs[name].y_range = y_range_csi_amp

        # one packet- all subcarriers, different figure per unit
        one_sc_figs[name] = figure(title=f"{name} sc amplitude", x_axis_label="packet_num", y_axis_label="CSI",
                                       plot_width=one_width, plot_height=height)
        one_sc_figs[name].add_tools(HoverTool(tooltips=[("(x,y)", "$x{0.2f}, $y{0.2f}")]))

        one_sc_figs[name].x_range = x_range_all_packets
        one_sc_figs[name].y_range = y_range_csi_amp


        #Euclidean error:
        euc_error_vs_pkt_num_figs[name] = figure(title=f"{name} error (euclidian distance)", x_axis_label="packet_num",
                                                  y_axis_label="CSI error- euclidean",
                                                  plot_width=one_width, plot_height=height)
        euc_error_vs_pkt_num_figs[name].x_range = x_range_all_packets
        euc_error_vs_pkt_num_figs[name].y_range = y_range_csi_amp_error

        euc_error_vs_pkt_num_figs[name].add_tools(HoverTool(tooltips=[("(x,y)", "$x{0.2f}, $y{0.2f}")]))

        #Quantisized error:
        error_quan_vs_pkt_num_figs[name] = figure(title=f"{name} euclidean error",
                                                  x_axis_label="packet_num",
                                                  y_axis_label="CSI quantized absolute error",
                                   plot_width=one_width, plot_height=height)
        error_quan_vs_pkt_num_figs[name].x_range = x_range_all_packets
        error_quan_vs_pkt_num_figs[name].y_range = y_range_csi_amp_quan

        error_quan_vs_pkt_num_figs[name].add_tools(HoverTool(tooltips=[("(x,y)", "$x{0.2f}, $y{0.2f}")]))

        # Hamming distance:
        ham_error_vs_pkt_num_figs[name] = figure(title=f"{name} hamming distance error", x_axis_label="packet_num",
                                             y_axis_label="CSI quantized Hamming distance error",
                                             plot_width=one_width, plot_height=height)
        ham_error_vs_pkt_num_figs[name].x_range = x_range_all_packets
        ham_error_vs_pkt_num_figs[name].y_range = y_range_csi_amp_quan

        ham_error_vs_pkt_num_figs[name].add_tools(HoverTool(tooltips=[("(x,y)", "$x{0.2f}, $y{0.2f}")]))

        range_val = 60
        # p.y_range = Range1d(0, range_val)
        # for margin in samples_dict[name].margins:


        #     one_packet_figs[name].line(x_amp, 64*[margin], line_width=3, color='gray', line_dash='dashed')

    packet_num_spinner = Spinner(
        title="packet number",  # a string to display above the widget
        low=0,  # the lowest possible number to pick
        high=100,  # the highest possible number to pick
        step=1,  # the increments by which the number can be adjusted
        value=DEFAULT_PACKET,  # the initial value to display in the widget
        width=total_width,  # the width of the widget in pixels
    )

    sc_num_spinner = Spinner(
        title="sc number",  # a string to display above the widget
        low=-base_sc,  # the lowest possible number to pick
        high=base_sc,  # the highest possible number to pick
        step=1,  # the increments by which the number can be adjusted
        value=DEFAULT_SC,  # the initial value to display in the widget
        width=total_width,  # the width of the widget in pixels
    )

    global my_layout
    my_layout = layout([
        [packet_num_spinner],
        [packet_multi_fig],
        [one_packet_figs['alice'], one_packet_figs['bob'], one_packet_figs['eve_alice'], one_packet_figs['eve_bob']],
        [text_boxes['alice'], text_boxes['bob'], text_boxes['eve_alice'], text_boxes['eve_bob']],
        [sc_num_spinner],
        [csi_val_vs_pkt_num_figs],
        [one_sc_figs['alice'], one_sc_figs['bob'], one_sc_figs['eve_alice'], one_sc_figs['eve_bob']],

        [euc_error_vs_pkt_num_figs['alice'], euc_error_vs_pkt_num_figs['bob'],
         euc_error_vs_pkt_num_figs['eve_alice'], euc_error_vs_pkt_num_figs['eve_bob']],
        [ham_error_vs_pkt_num_figs['alice'], ham_error_vs_pkt_num_figs['bob'],
         ham_error_vs_pkt_num_figs['eve_alice'], ham_error_vs_pkt_num_figs['eve_bob']],
        [error_quan_vs_pkt_num_figs['alice'], error_quan_vs_pkt_num_figs['bob'],
         error_quan_vs_pkt_num_figs['eve_alice'],
         error_quan_vs_pkt_num_figs['eve_bob']],
        [alice_bob_time_diff_vs_pkt_num_fig]
                        ])
    plot_one_index(DEFAULT_PACKET)


def calc_hamming_distance(a,b):
    levels = int(max(np.max(a),np.max(b))).bit_length()
    xored = np.bitwise_xor(a, b)
    ham_dist = np.zeros_like(a)
    bit_loc = levels
    while bit_loc != 0:
        ham_dist = ham_dist + np.bitwise_and(xored, 1)
        xored = np.right_shift(xored, 1)
        bit_loc -= 1
    return ham_dist




def plot_one_index(my_index, sc_index=DEFAULT_SC):
    global text_boxes
    global samples_dict
    global one_packet_figs
    global x_amp
    global x_packet
    global packet_multi_fig
    global color_bars
    global one_sc_figs
    global csi_val_vs_pkt_num_figs
    global error_quan_vs_pkt_num_figs
    global euc_error_vs_pkt_num_figs
    global ham_error_vs_pkt_num_figs
    global sc_data


    for name, text_box in text_boxes.items():
        print(name)
        print(samples_dict[name].to_str(my_index))
        text_box.text = samples_dict[name].to_str(my_index)
        csi_data = samples_dict[name].get_csi_amp_calibrated(my_index, rm_nulls=True, rm_pilots=False)
        if "eve" in name:
            line_dash = "dashed"
        else:
            line_dash = "solid"
        one_packet_figs[name].line(x_amp,csi_data, legend_label= name, line_width=3, color=color_bars[name],line_dash=line_dash)
        one_packet_figs[name].legend.click_policy="hide"
        packet_multi_fig.line(x_amp, csi_data, legend_label=name, line_width=3, color=color_bars[name], line_dash=line_dash)
        packet_multi_fig.legend.click_policy = "hide"
        packet_multi_fig.legend.label_text_font_size = '20pt'
    sc = sc_index+base_sc
    quan_key = samples_dict['alice'].csi_quan_with_major[:, sc]
    for name, samples in samples_dict.items():
        if "eve" in name:
            line_dash = "dashed"
        else:
            line_dash = "solid"
        csi_data = samples_dict[name].csi_amp[:, sc]
        csi_quan_data = samples_dict[name].csi_quan_with_major[:, sc]
        csi_quan_data_gray = samples_dict[name].csi_amp_quan_grayed[:, sc]

        euc_errors =  samples_dict['alice'].csi_amp[:, sc] - csi_data
        ham_errors = calc_hamming_distance(csi_quan_data_gray, samples_dict["alice"].csi_amp_quan_grayed[:, sc])
        quan_errors = np.absolute(csi_quan_data - quan_key)
        error_colors = [color_bars[x] for x in quan_errors]
        one_sc_figs[name].scatter(x_packet, csi_data, legend_label=name, line_width=3, color=color_bars[name])
        one_sc_figs[name].legend.click_policy = "hide"
        for margin_val in samples_dict[name].margins[:,sc]:
            one_sc_figs[name].line(x_packet, margin_val, legend_label=name, line_width=3, color='gray')

        error_quan_vs_pkt_num_figs[name].plus(x_packet, quan_errors , legend_label=name+"_error", line_width=3,
                                              color=color_bars[name])
        error_quan_vs_pkt_num_figs[name].legend.click_policy = "hide"
        euc_error_vs_pkt_num_figs[name].plus(x_packet, euc_errors, legend_label=name + "_error", line_width=3,
                                              color=color_bars[name])
        ham_error_vs_pkt_num_figs[name].plus(x_packet, ham_errors, legend_label=name + "_error", line_width=3,
                                             color=color_bars[name])

        csi_val_vs_pkt_num_figs.line(x_packet, csi_data, legend_label=name, line_width=3, color=color_bars[name],line_dash=line_dash)
        csi_val_vs_pkt_num_figs.legend.click_policy = "hide"
        csi_val_vs_pkt_num_figs.legend.label_text_font_size = '20pt'
        if name == 'bob':
            alice_bob_time_diff_vs_pkt_num_fig.scatter(x_packet, sc_data['time_diff_usec'], color = error_colors)

    show(my_layout)


if __name__ == "__main__":
    pcap_paths = {}
    global samples_dict
    samples_dict = {}


    bandwidth = 20
    # base_sc = 64
    q = 4
    nsub = int(bandwidth * 3.2)
    base_sc = int(nsub / 2)

    # pcap_filename = 'csi_ctrl_summary_D20221130_T0959_'
    # pcap_filename = 'csi_ctrl_summary_D20230208_T1231_' #Static
    #pcap_filename = 'csi_ctrl_summary_D20230208_T1245_' #static 20 ***
    pcap_filename = 'csi_ctrl_summary_D20230129_T0701_' #Good 20 ***
    # pcap_filename = 'csi_ctrl_summary_D20221211_T0034_'  # Good 40

    # pcap_filename ='csi_ctrl_summary_D20221023_T0724_'
    # pcap_filename = input('Pcap file name: ')
    if '.pcap' not in pcap_filename:
        if pcap_filename.endswith("_"):
            for name in ['alice', 'bob']:
                cycle_mod = 1
                sel_mod = 0
                pcap_paths[name] = os.path.join(config.pcap_fileroot,pcap_filename + f"{name}.pcap")
                samples_dict[name] = decoder.read_pcap(pcap_filepath=pcap_paths[name], bandwidth=bandwidth, nsamples_max=0,
                                                       quan_type="percentile", q=q, mask_dq=0,
                                                       cycle_mod=cycle_mod, sel_mod=sel_mod)

            cycle_mod = 2
            sel_mod = 0
            samples_dict['eve_alice'] = decoder.read_pcap(pcap_filepath=os.path.join(config.pcap_fileroot,pcap_filename + "eve.pcap"),
                                                        bandwidth=bandwidth, nsamples_max=0,
                                                          quan_type="percentile", q=q, mask_dq=0,
                                                          cycle_mod=cycle_mod, sel_mod=sel_mod)
            cycle_mod = 2
            sel_mod = 1
            samples_dict['eve_bob'] = decoder.read_pcap(pcap_filepath=os.path.join(config.pcap_fileroot,pcap_filename + "eve.pcap"), bandwidth=bandwidth, nsamples_max=0,
                                                        quan_type="percentile", q=q, mask_dq=0,
                                                        cycle_mod=cycle_mod, sel_mod=sel_mod)

        else:
            raise ValueError("unsupported input")

    samples = samples_dict[name]
    #Generate keys:
    for name,samples in samples_dict.items():
        samples.set_quan_params(quan_type = "percentile", q=q, mask_dq=0)
        samples.set_majority_margin_val(2)
        samples.generate_key()

    time_diff_usec = (samples_dict['eve_bob'].ts_sec - samples_dict['eve_alice'].ts_sec) * 10 ** 6 \
                           + samples_dict['eve_bob'].ts_usec - samples_dict['eve_alice'].ts_usec

    rssi_diff = np.abs(samples_dict['bob'].rssi).astype(int) #- np.abs(samples_dict['alice'].rssi).astype(int)

    global sc_data
    sc_data = {}
    sc_data['time_diff_usec'] = rssi_diff
    for name,samples in samples_dict.items():
        for sc in range(samples.num_sc):
            sc_data[f'{name}_sc_{sc-base_sc}_amp'] = samples.csi_amp[:, sc]
            sc_data[f'{name}_sc_{sc - base_sc}_amp_quan'] = samples.csi_quan_with_major[:, sc]
            sc_data[f'{name}_sc_{sc - base_sc}_amp_quan_error'] = samples.csi_quan_with_major[:, sc] - \
                                                             samples_dict['alice'].csi_quan_with_major[:, sc]
        for packet_num in range(samples.nsamples):
            sc_data[f'{name}_pkt_{packet_num}_amp'] = samples.csi_amp[packet_num, :]
            sc_data[f'{name}_pkt_{packet_num}_amp_quan'] = samples.csi_amp_quan[packet_num, :]

        sc_data[f'{name}_one_sc_fig_data'] = sc_data[f'{name}_sc_{base_sc+DEFAULT_SC}_amp']
        sc_data[f'{name}one_packet_figs'] = sc_data[f'{name}_pkt_0_amp_quan']





    init_bokeh(bandwidth=samples.bandwidth)


    while True:
        command = input('> ')

        if 'help' in command:
            print(config.help_str)
        
        elif 'exit' in command:
            break

        elif ('-' in command) and \
            string_is_int(command.split('-')[0]) and \
            string_is_int(command.split('-')[1]):

            start = int(command.split('-')[0])
            end = int(command.split('-')[1])

            for index in range(start, end+1):
                plot_one_index(index)
                time.sleep(config.plot_animation_delay_s)

        elif string_is_int(command):
            index = int(command)
            plot_one_index(index)
        else:
            print('Unknown command. Type help.')