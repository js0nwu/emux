import blender
import cv2
import sys

from moviepy.editor import *

OUTPUT_PREFIX = "out_"
OUTPUT_EXTENSION = ".jpg"

FPS = 10

picture_path = sys.argv[1]
replace_path = sys.argv[2]
num = int(sys.argv[3])

if len(sys.argv) == 5:
    FPS = int(sys.argv[4])

a = cv2.imread(picture_path)
b = cv2.imread(replace_path)

for i in range(1, num + 1):
    c = blender.generate_midframe(a, b, float(i) / float(num))
    cv2.imwrite(OUTPUT_PREFIX + str(i) + OUTPUT_EXTENSION, c)
    print("processed " + str(i))

clip = ImageSequenceClip([OUTPUT_PREFIX + str(i + 1) + OUTPUT_EXTENSION for i in range(num)], fps=FPS)
clip.write_gif(OUTPUT_PREFIX + "anim.gif", fps=FPS)
print("done!")
