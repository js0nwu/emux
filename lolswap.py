import blender
import cv2
import sys

OUTPUT_PREFIX = "out_"
OUTPUT_EXTENSION = ".jpg"

picture_path = sys.argv[1]
replace_path = sys.argv[2]
num = int(sys.argv[3])

a = cv2.imread(picture_path)
b = cv2.imread(replace_path)

for i in range(1, num + 1):
    c = blender.generate_midframe(a, b, float(i) / float(num))
    cv2.imwrite(OUTPUT_PREFIX + str(i) + OUTPUT_EXTENSION, c)
    print("processed " + str(i))

print("done!")
