import blender
import cv2

from moviepy.editor import *

import syncer

VIDEO_A = "angry.mp4"
VIDEO_B = "sad.mp4"

OUTPUT_FPS = 10

a = VideoFileClip(VIDEO_A)
b = VideoFileClip(VIDEO_B)

synced_b = syncer.sync_clips(a, b)

time = 0
tstep = 1 / OUTPUT_FPS
frame = 0
while time < a.duration:
    frame_a = a.get_frame(time)
    frame_b = b.get_frame(time)
    frame_c = blender.generate_midframe(frame_a, frame_b, 0.5)
    cv2.imwrite("frame" + str(frame) + ".jpg", frame_c)
    frame += 1
    time += tstep