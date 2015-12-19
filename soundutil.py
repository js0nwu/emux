from itertools import groupby
from operator import itemgetter

from scipy.fftpack import fft, dct
from scipy.signal import hamming
import numpy
import itertools

def mel(hertz):
    return 1125 * numpy.log(1 + hertz/ 700)
def mel_inv(mels):
    return 700 * (numpy.exp(mels / 1125) - 1)
def filterbanks(low, high, num):
    return [mel_inv(x) for x in list(range(int(mel(low)), int(mel(high)), int((mel(high) - mel(low)) / (num - 1))))]
def hf(f, m, k):
    if k < f[max(0, m - 1)]:
        return 0
    elif k >= f[m - 1] and k < f[m]:
        return (k - f[m - 1]) / (f[m] - f[m - 1])
    elif k >= f[m] and k < f[m + 1]:
        return (f[m + 1] - k) / (f[m + 1] - f[m])
    elif k >= f[min(len(f) - 1, m + 1)]:
        return 0
    else:
        return 0
def frame_power_transform(frame):
    n = 256
    window = hamming(frame.size)
    frame_dft = numpy.multiply(fft(window * frame, n, axis=0), (1 / numpy.sqrt(n)))
    frame_power = numpy.abs(numpy.square(frame_dft))
    return frame_power
    
def signal_energy(frame):
    return numpy.sum(frame_power_transform(frame))

def mfcc(frame, frequency):
    frame_power = frame_power_transform(frame)
    # code to auto determine filter bank frequency range based on input audio sample rate
    frame_banks = [int(((frame_power.size + 1) * x) / frequency) for x in filterbanks(200, int(frequency / 2), 12)]
    max_bank = numpy.max(numpy.asarray(frame_banks))
    frame_mels = list(itertools.starmap(lambda x, y : hf(frame_banks, x, y) * frame_power[y], itertools.product(range(len(frame_banks)), range(max_bank))))
    frame_ampl = [numpy.log(numpy.sum(x)) if x != 0 else 0 for x in frame_mels]
    return dct(frame_ampl, axis=0)
def zero_cross_rate(frame):
    return numpy.sum(numpy.abs([frame[x] + frame[x + 1] for x in range(frame.size - 1)])) / frame.size
def cep_matrix(subframes, frequency):
    return numpy.asarray([mfcc(x, frequency) for x in subframes])
def group_consecutive(vals, step):
    consecs = []
    for k, g in groupby(enumerate(vals), lambda ix: ix[0] - ix[1]):
        consecs.append(list(map(itemgetter(step), g)))
    return consecs
def combine_sigs(signals):
    return numpy.concatenate(signals)
def sound_split_frames(values, frequency, duration):
    return numpy.array_split(values, int(values.size / (frequency * duration)))
