import blender
import cv2

from moviepy.editor import *
from moviepy.audio.AudioClip import *

import syncer

PICTURE = "obama.jpg"
REPLACE = "obama2.jpg"

a = cv2.imread(PICTURE)
b = cv2.imread(REPLACE)

c = blender.generate_midframe(a, b, 0.5)

blender.cv_display_image("c", c)

cv2.imwrite("obama3.jpg", c)
exit(0)

VIDEO_A = "angry.mp4"
VIDEO_B = "sad.mp4"

OUTPUT_FPS = 10

a = VideoFileClip(VIDEO_A)
b = VideoFileClip(VIDEO_B)

synced_b = syncer.sync_clips(a, b)

print("done syncing")

synced_b.to_videofile("happysync2.mp4")

time = 0
tstep = 1 / OUTPUT_FPS

frames = []


def get_factor(t):
    return 0.8

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
make_frame = lambda t: (1 - get_factor(t)) * a_audio.get_frame(t) + get_factor(t) * b_audio.get_frame(t)
audios = AudioClip(make_frame, duration=a_audio.duration)
audios.to_audiofile("goodaudio.wav")
images = images.set_audio(audios)
images.to_videofile("glorious.mp4")
