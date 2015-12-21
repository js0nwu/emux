# based on code and ideas from
# https://github.com/matthewearl/faceswap/blob/master/faceswap.py
# http://stackoverflow.com/a/10374811

import cv2
import dlib
import numpy

UPSAMPLE = 1

FACE_CASCADE_PATH = 'train/haarcascade_frontalface_default.xml'

FACE_PREDICTOR_PATH = 'train/shape_predictor_68_face_landmarks.dat'

HOG_DETECT = False

FACE_SCALE = 1.3
FACE_NEIGHBOR = 5

FEATHER_AMOUNT = 17

COLOR_CORRECT_BLUR = 0.6

LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

face_detector_hog = dlib.get_frontal_face_detector()
face_detector_haar = cv2.CascadeClassifier(FACE_CASCADE_PATH)
face_predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)


def find_faces(picture, hog=True):
    if hog:
        return find_objects_hog(picture, face_detector_hog)
    else:
        return find_objects_cascade(picture, face_detector_haar)


def find_objects_cascade(picture, detector):
    faces = detector.detectMultiScale(picture, FACE_SCALE, FACE_NEIGHBOR)
    return [dlib.rectangle(left=int(f[0]), top=int(f[1]), right=int(f[0] + f[2]), bottom=int(f[1] + f[3])) for f in
            faces]


def find_objects_hog(picture, detector):
    return detector(picture, UPSAMPLE)


def get_landmarks(picture, bounding, predictor):
    return numpy.matrix([[p.x, p.y] for p in predictor(picture, bounding).parts()])


def col_string(color):
    if isinstance(color, str):
        rgb_color = (0, 0, 0)
        if color.lower() == 'red':
            rgb_color = (0, 0, 255)
        elif color.lower() == 'blue':
            rgb_color = (255, 0, 0)
        elif color.lower() == 'green':
            rgb_color = (0, 255, 0)
        elif color.lower() == 'black':
            rgb_color = (0, 0, 0)
        elif color.lower() == 'white':
            rgb_color = (255, 255, 255)
        return rgb_color
    else:
        return color


class Box(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @classmethod
    def from_dlib_rect(cls, b):
        return cls(b.center().x - b.width() // 2, b.center().y - b.height() // 2, b.width(), b.height())


def draw_box(picture, box, color='red', thick=2):
    rgb_color = col_string(color)
    cv2.rectangle(picture, (box.x, box.y), (box.x + box.w, box.y + box.h), rgb_color, thick)


class Line(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def start(self):
        return Point(self.x1, self.y1)

    def end(self):
        return Point(self.x2, self.y2)


class Region(object):
    def __init__(self, picture, x, y, w, h):
        self.picture = picture
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @classmethod
    def from_box(cls, picture, b):
        return cls(picture, b.x, b.y, b.w, b.h)

    @classmethod
    def from_picture(cls, picture):
        return cls(picture, 0, 0, picture.shape[1], picture.shape[0])

    def get_image(self):
        return self.picture[self.y: self.y + self.h, self.x: self.x + self.w]


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def from_arr(cls, a):
        return cls(a[0], a[1])


def draw_line(picture, line, color='red', thick=2):
    rgb_color = col_string(color)
    cv2.line(picture, (line.x1, line.y1), (line.x2, line.y2), rgb_color, thick)


def connect_dots(points):
    return [Line(points[si, 0], points[si, 1], points[si + 1, 0], points[si + 1, 1]) for si in range(len(points) - 1)]


def procrustes(p1, p2):
    p1 = p1.astype(numpy.float32)
    p2 = p2.astype(numpy.float32)
    c1 = numpy.mean(p1, axis=0)
    c2 = numpy.mean(p2, axis=0)
    p1 -= c1
    p2 -= c2
    s1 = numpy.std(p1)
    s2 = numpy.std(p2)
    p1 /= s1
    p2 /= s2
    U, S, Vt = numpy.linalg.svd(p1.T * p2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1])])


def warp_picture(picture, matrix, shape):
    output = numpy.zeros(shape, dtype=picture.dtype)
    cv2.warpAffine(picture, matrix[:2], (shape[1], shape[0]), dst=output, borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output


def cv_display_image(title, picture):
    cv2.imshow(title, picture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_convex_hull(picture, points, color):
    cv2.fillConvexPoly(picture, cv2.convexHull(points), color=color)


def generate_mask(picture, points, grouped=True):
    mask = numpy.zeros(picture.shape[:2], dtype=numpy.float64)
    if grouped:
        for group in OVERLAY_POINTS:
            draw_convex_hull(mask, points[group], 1)
    else:
        draw_convex_hull(mask, points, 1)
    mask = numpy.array([mask, mask, mask]).transpose((1, 2, 0))
    return mask


def feather_image(picture, amount):
    picture = (cv2.GaussianBlur(picture, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    picture = cv2.GaussianBlur(picture, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return picture


def linear_blend_mask(a, b, mask):
    return numpy.array(a * (1.0 - mask) + b * mask, dtype=numpy.uint8)


def linear_blend(a, b, f):
    return numpy.array(a * (1.0 - f) + b * f, dtype=numpy.uint8)


def combine_masks(a, b):
    return numpy.max([a, b], axis=0)


def generate_combined_mask(a, la, b, lb):
    return combine_masks(generate_mask(a, la), feather_image(generate_mask(b, lb), FEATHER_AMOUNT))


def color_correct(a, b, la):
    blur_amount = COLOR_CORRECT_BLUR * numpy.linalg.norm(
        numpy.mean(la[LEFT_EYE_POINTS], axis=0) - numpy.mean(la[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    a_blur = cv2.GaussianBlur(a, (blur_amount, blur_amount), 0)
    b_blur = cv2.GaussianBlur(b, (blur_amount, blur_amount), 0)

    b_blur += ((b_blur <= 1.0) * 128).astype(b_blur.dtype)

    return numpy.clip((b.astype(numpy.float64) * a_blur.astype(numpy.float64) / b_blur.astype(numpy.float64)), 0, 255)


def face_swap(mat_picture, mat_replace, poisson = False):
    p_gray = cv2.cvtColor(mat_picture, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(mat_replace, cv2.COLOR_BGR2GRAY)
    f_faces = find_faces(p_gray, HOG_DETECT)
    f = f_faces[0]
    f_landmarks = get_landmarks(mat_picture, f, face_predictor)
    r_faces = find_faces(r_gray, HOG_DETECT)
    r = r_faces[0]
    r_landmarks = get_landmarks(mat_replace, r, face_predictor)
    transform = procrustes(f_landmarks, r_landmarks)
    mat_replace = warp_picture(mat_replace, transform, mat_picture.shape)
    r_landmarks = get_landmarks(mat_replace, r, face_predictor)
    mask = generate_combined_mask(mat_picture, f_landmarks, mat_replace, r_landmarks)
    return (linear_blend_mask(mat_picture, color_correct(mat_picture, mat_replace, f_landmarks), mask), f_landmarks,
            r_landmarks, mask)


def projection_points(l):
    return numpy.float32([l[36], l[45], l[55], l[59]])
    # return numpy.float32([numpy.mean(l[LEFT_BROW_POINTS], axis = 0), numpy.mean(l[RIGHT_BROW_POINTS], axis = 0), numpy.mean(l[LEFT_EYE_POINTS], axis = 0), numpy.mean(l[RIGHT_EYE_POINTS], axis = 0)])


def warp_picture_landmarks(a, la, b, lb, f):
    ldiff = lb - la
    ldiff *= f
    src = la
    dst = la + ldiff
    M = cv2.getPerspectiveTransform(projection_points(src), projection_points(dst))
    return cv2.warpPerspective(a, M, (b.shape[1], b.shape[0]))


def morph_picture(a, la, b, lb, mask, f, warp=True):
    if not warp:
        return linear_blend(a, b, f)
    warped_a = warp_picture_landmarks(a, la, b, lb, f)
    warped_b = warp_picture_landmarks(b, lb, a, la, (1.0 - f))
    return linear_blend(linear_blend_mask(a, warped_a, mask), linear_blend_mask(a, warped_b, mask), f)


def generate_midframe(a, b, f):
    output_picture, input_l, output_l, mask = face_swap(a, b)
    return morph_picture(a, input_l, output_picture, output_l, mask, f)
