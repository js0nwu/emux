import numpy

from scipy.signal import hamming

import itertools
from scipy.fftpack import dct, fft

from audiofiles import utility

from moviepy.editor import *

import blender

import queue

PATH_STEPS = [0.5, 1.0, 1.5, 2.0]

FRAME_LENGTH = 3


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

def extract_frame(r, s, t, l =FRAME_LENGTH):
    index_start = r * t
    index_end = r * (t + l)
    return s[index_start:index_end]

def cost1(t1, t2, av, bv):
    return blender.face_distance(av.get_frame(t1), bv.get_frame(t2))

def cost2(t1, t2, ar, aa, br, ba):
    af = extract_frame(ar, aa, t1)
    bf = extract_frame(br, ba, t2)
    return mfcc_distance(ar, af, br, bf)

def get_cost(t1, t2, av, bv, ar, aa, br, ba):
    return cost1(t1, t2, av, bv) + cost2(t1, t2, ar, aa, br, ba)

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
    visited = {}
    paths = []
    while pq.empty() == False:
        sc, sa, sb, sp = pq.get()
        print(sa)
        key = (sa, sb)
        if key not in visited or visited[key] > sc:
            # pruning heuristic for min time left on sa and visited?
            for n in PATH_STEPS:
                nw = n * FRAME_LENGTH
                nat = sa + FRAME_LENGTH
                nbt = sb + nw
                if nat < a.duration and nbt < b.duration:
                    ncost = sc + get_cost(nat, nbt, a, b,  a_r, a_audio, b_r, b_audio)
                    pq.put((ncost, nat, nbt, sp + [nw]))
                elif nat >= a.duration:
                    paths.append((sc, sp))
            visited[key] = sc

    print(paths)
    scost, spath = paths[0]
    clips = []
    cstart = 0
    for seg in spath:
        clips.append(b.subclip(cstart, cstart+ seg).speedx(seg / FRAME_LENGTH))
        cstart += seg

    remainder = b.subclip(cstart, b.duration).speedx((b.duration - cstart) / (a.duration % FRAME_LENGTH))
    clips.append(remainder)
    return concatenate(clips)

a = VideoFileClip("angry.mp4")
b = VideoFileClip("sad.mp4")
synced = sync_clips(a, b)
synced.set_audio(a.audio).write_videofile("happysync.mp4")


