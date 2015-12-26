import blender
import cv2

from moviepy.editor import *
from moviepy.audio.AudioClip import *

import syncer

from threading import Thread

VIDEO_A = "angry.mp4"
VIDEO_B = "sad.mp4"

OUTPUT_FPS = 10

a = VideoFileClip(VIDEO_A)
b = VideoFileClip(VIDEO_B)

synced_b = syncer.sync_clips(a, b)

print("done syncing")

time = 0
tstep = 1 / OUTPUT_FPS

frames = []


def get_factor(t):
    return 0.5


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
a_audio = a.audio.to_soundarray()
a_audio = blender.stereo_to_mono(a_audio)
# a = a.set_audio(AudioArrayClip([a_audio], a.audio.fps))
make_frame = lambda t: mix_audio_frame(a.audio.get_frame(t), synced_b.audio.get_frame(t), get_factor(t))
audios = AudioClip(make_frame, duration=a.audio.duration)
# images = images.set_audio(audios)
images.set_audio(a.audio)
images.to_videofile("glorious.mp4")
