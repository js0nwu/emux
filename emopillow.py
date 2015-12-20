import blender
import cv2

PICTURE_PATH = 'picture.jpg'
REPLACE_PATH = 'replace.jpg'

mat_picture = cv2.imread(PICTURE_PATH)
mat_replace = cv2.imread(REPLACE_PATH)

output = blender.generate_midframe(mat_picture, mat_replace, 0.5)

# blender.cv_display_image('output', output)
