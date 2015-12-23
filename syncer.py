import scipy.signal as signal
import scipy.io.wavfile as wav

import numpy
import cv2

import itertools
from scipy.fftpack import dct

from audiofiles import utility

from moviepy.editor import *

import blender

PATH_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

def mel(hertz):
    return 1125 * numpy.log(1 + hertz/ 700)
def mel_inv(mels):
    return 700 * (numpy.exp(mels / 1125) - 1)
def filterbanks(low, high, num):
    return [mel_inv(x) for x in range(int(mel(low)), int(mel(high)), int((mel(high) - mel(low)) / (num - 1)))]
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

def cost1(t1, t2, av, bv):
    return 0

def cost2(t1, t2, aa, ba):
    return 0

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