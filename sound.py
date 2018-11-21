#!/usr/bin/python3.5
#http://www.signalogic.com/index.pl?page=codec_samples

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('wav/male.wav')
print(sample_rate)
print(len(samples))

frequencies, times, spectrogram = signal.spectrogram(samples[:10000], sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
