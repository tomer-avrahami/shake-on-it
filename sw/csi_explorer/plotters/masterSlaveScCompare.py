import numpy as np
import matplotlib.pyplot as plt

'''
Amplitude and Phase plotter
---------------------------

Plot Amplitude and Phase of CSI samples
and update the plots in the same window.

Initiate Plotter with bandwidth, and
call update with CSI value.
'''

__all__ = [
    'masterSlaveScCompare'
]

class masterSlaveScCompare():
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth

        nsub = int(bandwidth * 3.2)
        self.x_amp = np.arange(-1 * nsub/2, nsub/2)

        self.fig, axs = plt.subplots(1)

        self.ax_amp = axs

        self.fig.suptitle('Subcarrier compare')

        plt.ion()
        plt.show() 
    
    def update(self, csi_amp):
        self.ax_amp.clear()

        # These are also cleared with clear()
        self.ax_amp.set_ylabel('Amplitude')
        self.ax_amp.set_xlabel('Subcarrier')

        try:
            self.ax_amp.plot(self.x_amp, csi_amp)

        except ValueError as err:
            print(
                f'A ValueError occurred. Is the bandwidth {self.bandwidth} MHz correct?\nError: ', err
            )
            exit(-1)
        plt.draw()
        plt.pause(0.001)
    
    def __del__(self):
        pass
