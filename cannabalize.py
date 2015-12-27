from moviepy.editor import *
import moviepy.video.fx.all as vfx

TARGET_FILE = "facedirector.mp4"

x1 = 130
y1 = 30
x2 = 1150
y2 = 605

# times for neutral

# t1 = 156
# t2 = 169

t1 = 145
t2 = 155

source = VideoFileClip(TARGET_FILE)

edit = source.subclip(t1, t2).fx(vfx.crop, x1=x1, y1=y1, x2=x2, y2=y2)
edit.to_videofile("output.mp4")
