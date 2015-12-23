import numpy

from scipy.signal import hamming

import itertools
from scipy.fftpack import dct, fft

from audiofiles import utility

from moviepy.editor import *

import blender

PATH_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

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

def mfcc_distance(r1, s1, r2, s2):
    a = mfcc(s1, r1)
    b = mfcc(s2, r2)
    return numpy.sqrt(numpy.sum(numpy.square(b - a)))

def cost1(t1, t2, av, bv):
    return blender.face_distance(av.get_frame(t1), bv.get_frame(t2))

def cost2(t1, t2, aa, ba):
    return mfcc_distance(stereo_to_mono(aa.get_frame(t1)), stereo_to_mono(ba.get_frame(t2)))

def get_cost(t1, t2, av, bv, aa, ba):
    return cost1(t1, t2, av, bv) + cost2(t1, t2, aa, ba)

def remainder(ta, tb, tastep):
    return 0

def stereo_to_mono(s):
    return utility.float2pcm(numpy.mean(s, axis = 1))

def sync_clips(a, b, t):
    b_audio = stereo_to_mono(b.audio.to_soundarray())
    b_r = b.audio.fps
    time = a.duration
    clips = []
    clip_start = 0
    while time > 0:
        clip_length = min(time, t)
        a_clip = a.subclip(clip_start, clip_start + clip_length)
        a_audio = stereo_to_mono(a_clip.audio.to_soundarray())

        clip_start += clip_length
        time -= clip_length
    synced = concatenate(clips)
    return synced