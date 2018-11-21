#!/usr/bin/python3.5
#http://www.signalogic.com/index.pl?page=codec_samples

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('wav/male.wav')
print(sample_rate)
print(len(samples))

frequencies, times, spectrogram = signal.spectrogram(samples[:10000], sample_rate)

f, ax = plt.subplots()
ax.pcolormesh(times, frequencies, spectrogram, cmap='viridis')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]');
plt.show()
