import blender
import cv2

from moviepy.editor import *

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
frame = 0

def process_frame(a, b, time, factor, frame):
    frame_a = cv2.cvtColor(a.get_frame(time), cv2.COLOR_RGB2BGR)
    frame_b = cv2.cvtColor(synced_b.get_frame(time), cv2.COLOR_RGB2BGR)
    frame_c = blender.generate_midframe(frame_a, frame_b, factor)
    cv2.imwrite("frame" + str(frame) + ".jpg", frame_c)

# TODO multithread this
while time < a.duration:
    print("frame at " + str(time))
    process_frame(a, b, time, 0.85, frame)
    frame += 1
    time += tstep