import blender
import cv2

from moviepy.editor import *
from moviepy.audio.AudioClip import *

import syncer
import numpy
from multiprocessing import Pool

VIDEO_A = "scared.mp4"
VIDEO_B = "neutral.mp4"

OUTPUT_FPS = 10

a = VideoFileClip(VIDEO_A)
b = VideoFileClip(VIDEO_B)

synced_b = syncer.sync_clips(a, b)

print("done syncing")

time = 0
tstep = 1 / OUTPUT_FPS

def get_factor(t):
    return 0.8


def process_frame(framenum, a, b, factor):
    print("frame at " + str(framenum))
    frame_a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    frame_b = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
    frame_c = cv2.cvtColor(blender.generate_midframe(frame_a, frame_b, factor), cv2.COLOR_BGR2RGB)
    return (framenum, frame_c)


def process_frame_helper(args):
    return process_frame(*args)


pool = Pool(processes=8)
poolargs = [(t, a.get_frame(t), synced_b.get_frame(t), get_factor(t)) for t in numpy.arange(0, a.duration, tstep)]
poolout = pool.map(process_frame_helper, poolargs)
print([p[0] for p in poolout])
exit(0)
images = ImageSequenceClip(poolout, fps=OUTPUT_FPS)
a_audio = a.audio
b_audio = synced_b.audio
make_frame = lambda t: (1 - get_factor(t)) * a_audio.get_frame(t) + get_factor(t) * b_audio.get_frame(t)
audios = AudioClip(make_frame, duration = a_audio.duration)
images = images.set_audio(audios)
images.to_videofile("glorious.mp4")
