import scipy.signal as signal
import soundutil

import scipy.io.wavfile as wav

import numpy
import cv2

MIN_MATCHES = 4

FLANN_INDEX_KDTREE = 1

SPEC_MAX = 20
SPEC_MIN = -36

signal_detector = cv2.xfeatures2d.SIFT_create()
signal_matcher = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=4), {})


class SignalFinder(object):
    def __init__(self, r, s):
        self.r = r
        self.s = s
        self.fingers = None

    @staticmethod
    def get_finger(r, s):
        spec = numpy.clip(numpy.log(signal.spectrogram(s, fs=r)[2]), SPEC_MIN, SPEC_MAX)
        spec -= SPEC_MIN
        spec /= (SPEC_MAX - SPEC_MIN)
        spec *= 255
        spec = spec.astype(dtype=numpy.uint8)
        return (signal_detector.detectAndCompute(spec, None), spec.shape)

    def train_fingers(self):
        self.fingers = SignalFinder.get_finger(self.r, self.s)

    def find_signal(self, r, s):
        if self.fingers is None:
            self.train_fingers()
        query = SignalFinder.get_finger(r, s)
        matches = signal_matcher.knnMatch(query[0][1], self.fingers[0][1], k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)
        print(len(good))
        if len(good) >= MIN_MATCHES:
            src_pts = numpy.float32([query[0][0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = numpy.float32([self.fingers[0][0][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = query[1]
            pts = numpy.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            dst_sort = dst[dst[:,0].argsort()]
            print(dst_sort)
            return (dst_sort[0][0], dst_sort[-1][0] - dst_sort[0][0])
        else:
            return (0, 0)


def sync_signals(r1, s1, r2, s2, n):
    a_duration = s1.size // r1
    b_duration = s2.size // r2
    step_duration = a_duration / n
    steps = soundutil.sound_split_frames(s1, step_duration, r1)

r1, s1 = wav.read('angry.wav')
r2, s2 = wav.read('sad.wav')
s1 = s1[15000:23000]
sm = SignalFinder(r2, s2)
sm.train_fingers()
print(sm.find_signal(r1, s1))