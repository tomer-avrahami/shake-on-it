import time
import os
import importlib
import config
from plotters.AmpPhaPlotter import Plotter # Amplitude and Phase plotter
from plotters.CalibratedAmpPlotter import CalibratedAmpPlotter # Amplitude plotter
decoder = importlib.import_module(f'decoders.{config.decoder}') # This is also an import

def string_is_int(s):
    '''
    Check if a string is an integer
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False

def plot_one_index(my_index):
    if config.print_samples:
        for samples in samples_list:
            samples.print(my_index)
    if config.plot_samples:
        csi_list = []
        for samples in samples_list:
            csi_list.append(
                samples.get_csi(
                    my_index,
                    config.remove_null_subcarriers,
                    config.remove_pilot_subcarriers
                )
            )
        plotter.update(csi_list)

    if config.plot_samples_calibrated_amp:
        for samples in samples_list:
            csi_list.append(
                samples.get_csi_amp_calibrated(
                    my_index,
                    config.remove_null_subcarriers,
                    config.remove_pilot_subcarriers
                )
            )
        CalibratedAmpPlotter.update(csi_list)

if __name__ == "__main__":
    pcap_paths = []
    samples_list = []
    pcap_filename = input('Pcap file name: ')
    if '.pcap' not in pcap_filename:
        if pcap_filename.endswith("_"):
            for name in ['alice','bob']:
                pcap_paths.append(os.path.join(config.pcap_fileroot,pcap_filename + f"{name}.pcap"))
        else:
            pcap_paths.append(pcap_filepath = os.path.join(config.pcap_fileroot, pcap_filename))



        for pcap_filepath in pcap_paths:
            try:
                samples_list.append(decoder.read_pcap(pcap_filepath))
            except FileNotFoundError:
                print(f'File {pcap_filepath} not found.')
                exit(-1)

    samples = samples_list[0]
    if config.plot_samples:
        plotter = Plotter(samples.bandwidth)
    if config.plot_samples_calibrated_amp:
        CalibratedAmpPlotter = CalibratedAmpPlotter(samples.bandwidth)

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

            if config.print_samples:
                samples.print(index)
            if config.plot_samples:
                    csi = samples.get_csi(
                        index,
                        config.remove_null_subcarriers,
                        config.remove_pilot_subcarriers
                    )
                    plotter.update(csi)
        else:
            print('Unknown command. Type help.')