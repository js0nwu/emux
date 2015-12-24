import numpy

from scipy.signal import hamming
import scipy.io.wavfile as wav

import itertools
from scipy.fftpack import dct, fft

from audiofiles import utility

from moviepy.editor import *

import blender

import queue

import librosa.effects as re

import numpy.linalg

import os

PATH_STEPS = [1/2, 2/3, 3/4, 1, 4/3, 3/2, 2/1]

FRAME_LENGTH = 0.5


def mel(hertz):
    return 1125 * numpy.log(1 + hertz / 700)


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
    frame_mels = list(itertools.starmap(lambda x, y: hf(frame_banks, x, y) * frame_power[y],
                                        itertools.product(range(len(frame_banks)), range(max_bank))))
    frame_ampl = [numpy.log(numpy.sum(x)) if x != 0 else 0 for x in frame_mels]
    return dct(frame_ampl, axis=0)


def mfcc_distance(r1, s1, r2, s2):
    a = mfcc(s1, r1)
    b = mfcc(s2, r2)
    return numpy.linalg.norm(b - a)

def extract_frame(r, s, t, l=FRAME_LENGTH):
    index_start = r * t
    index_end = r * (t + l)
    return s[index_start:index_end]


def cost1(t1, t2, av, bv):
    return blender.face_distance(av.get_frame(t1), bv.get_frame(t2))


def cost2(t1, t2, ar, aa, br, ba):
    af = extract_frame(ar, aa, t1)
    bf = extract_frame(br, ba, t2)
    return mfcc_distance(ar, af, br, bf)


def get_cost(t1, t2, av, bv, ar, aa, br, ba, landmarks=False):
    if landmarks:
        return cost1(t1, t2, av, bv) + cost2(t1, t2, ar, aa, br, ba)
    else:
        return cost2(t1, t2, ar, aa, br, ba)


def stereo_to_mono(s):
    return utility.float2pcm(numpy.mean(s, axis=1))


def sync_clips(a, b):
    a_audio = stereo_to_mono(a.audio.to_soundarray())
    a_r = a.audio.fps
    b_audio = stereo_to_mono(b.audio.to_soundarray())
    b_r = b.audio.fps
    pq = queue.PriorityQueue()
    for e in PATH_STEPS:
        ew = e * FRAME_LENGTH
        cost = get_cost(FRAME_LENGTH, ew, a, b, a_r, a_audio, b_r, b_audio)
        pq.put((cost, FRAME_LENGTH, ew, [ew]))
    visited = []
    paths = queue.PriorityQueue()
    minpath = None
    while pq.empty() == False:
        sc, sa, sb, sp = pq.get()
        # pruning heuristic for min time left on sa and visited?
        stepsleft = (a.duration - sa) / FRAME_LENGTH
        if sb + stepsleft * numpy.min(PATH_STEPS) >= b.duration:
            continue
        if sb + (stepsleft + 1) * numpy.max(PATH_STEPS) < b.duration:
            continue
        if minpath is not None and sc >= minpath:
            continue
        print(sa)
        key = (sa, sb)
        if key in visited:
            continue
        for n in PATH_STEPS:
            nw = n * FRAME_LENGTH
            nat = sa + FRAME_LENGTH
            nbt = sb + nw
            if nbt >= b.duration:
                continue
            if nat >= a.duration:
                b_fact = ((a.duration - sa) / FRAME_LENGTH) * nw
                ncost = sc + get_cost(a.duration, sb + b_fact, a, b, a_r, a_audio, b_r, b_audio)
                paths.put((ncost, sp + [nw]))
                if minpath is None or sc < minpath:
                    minpath = sc
                continue
            ncost = sc + get_cost(nat, nbt, a, b, a_r, a_audio, b_r, b_audio)
            pq.put((ncost, nat, nbt, sp + [nw]))

        visited.append(key)

    scost, spath = paths.get()
    clips = []
    cstart = 0
    for seg in spath:
        clip = b.subclip(cstart, cstart + seg)
        stretch_factor = seg / FRAME_LENGTH
        a_array = re.time_stretch(utility.pcm2float(stereo_to_mono(clip.audio.to_soundarray())), stretch_factor)
        wav.write("tmp.wav", clip.audio.fps, a_array)
        clip_audio = AudioFileClip("tmp.wav")
        os.remove("tmp.wav")
        clip = clip.speedx(stretch_factor).set_audio(clip_audio)
        clips.append(clip)
        cstart += seg

    return concatenate(clips).subclip(0, a.duration)

