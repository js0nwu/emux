import blender
import cv2
from moviepy.editor import *
from moviepy.audio.AudioClip import *
import syncer
import numpy
from multiprocessing import Pool

VIDEO_A = "sadsmall.mp4"
VIDEO_B = "angrysmall.mp4"

OUTPUT_FPS = 10

a = VideoFileClip(VIDEO_A)
b = VideoFileClip(VIDEO_B)

synced_b = syncer.sync_clips(a, b)

print("done syncing")

time = 0
tstep = 1 / OUTPUT_FPS


def get_factor2(t):
    t = numpy.mean(t)
    if t > 0.4 * a.duration and t < 0.55 * a.duration:
        tdelta = t - 0.4 * a.duration
        ttotal = 0.15 * a.duration
        return 0.1 + (tdelta / ttotal) * 0.9
    elif t < 0.4 * a.duration:
        return (t / (0.4 * a.duration)) * 0.1
    else:
        return 1


def get_factor(t):
    t = numpy.mean(t)
    return t / a.duration


def process_frame(framenum, a, b, factor):
    print("frame at " + str(framenum))
    frame_a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    frame_b = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
    frame_c = cv2.cvtColor(blender.generate_midframe(frame_a, frame_b, factor), cv2.COLOR_BGR2RGB)
    return (framenum, frame_c)


def process_frame_helper(args):
    return process_frame(*args)


if __name__ == '__main__':
    pool = Pool(processes=8)
    poolargs = [(t, a.get_frame(t), synced_b.get_frame(t), get_factor(t)) for t in numpy.arange(0, a.duration, tstep)]
    poolout = pool.map(process_frame_helper, poolargs)
    poolout = [f[1] for f in sorted(poolout, key=lambda x: x[0])]
    images = ImageSequenceClip(poolout, fps=OUTPUT_FPS)
    a_audio = a.audio
    b_audio = synced_b.audio
    make_frame = lambda t: (1 - get_factor(t)) * a_audio.get_frame(t) + get_factor(t) * b_audio.get_frame(t)
    audios = AudioClip(make_frame, duration=a_audio.duration)
    images = images.set_audio(audios)

    synced_b.to_videofile("syncedb.mp4")
    images.to_videofile("images.mp4")
