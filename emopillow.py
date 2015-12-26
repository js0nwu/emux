import blender
import cv2

import numpy

from moviepy.editor import *
from moviepy.audio.AudioClip import *

import syncer

VIDEO_A = "angry.mp4"
VIDEO_B = "sad.mp4"

OUTPUT_FPS = 10

a = VideoFileClip(VIDEO_A)
b = VideoFileClip(VIDEO_B)

synced_b = syncer.sync_clips(a, b)

print("done syncing")

exit(0)

synced_b.to_videofile("happysync2.mp4")

time = 0
tstep = 1 / OUTPUT_FPS

frames = []


def get_factor(t):
    return 0.8


def generate_audio(a, b, d):
    t = 0
    track = []
    while t < d:
        a_seg = a.subclip(t, min(d, t + tstep))
        b_seg = b.subclip(t, min(d, t + tstep))
        seg = CompositeAudioClip([a_seg.volumex(1 - get_factor(t)), b_seg.volumex(get_factor(t))])
        track.append(seg)
        t += tstep
    return concatenate_audioclips(track)

def mix_audio_frame(a, b, factor):
    if factor == 0:
        return a
    if factor == 1.0:
        return b
    if a.shape != b.shape:
        return a
    return (1.0 - factor) * a + factor * b


def process_frame(a, b, factor):
    frame_a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    frame_b = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
    frame_c = cv2.cvtColor(blender.generate_midframe(frame_a, frame_b, factor), cv2.COLOR_BGR2RGB)
    frames.append(frame_c)

while time < a.duration:
    print("frame at " + str(time))
    process_frame(a.get_frame(time), synced_b.get_frame(time), get_factor(time))
    time += tstep

images = ImageSequenceClip(frames, fps=OUTPUT_FPS)
a_audio = a.audio
b_audio = synced_b.audio
audios = generate_audio(a_audio, b_audio, a.audio.duration)
audios.to_audiofile("badaudio.wav")
images = images.set_audio(audios)
images.to_videofile("glorious.mp4")
