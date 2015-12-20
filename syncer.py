import scipy.signal as signal
import soundutil

import numpy
import cv2

FLANN_INDEX_KDTREE = 1

SPEC_MAX = 20
SPEC_MIN = -36

signal_detector = cv2.xfeatures2d.SIFT_create()
signal_matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_KDTREE, trees = 4), {})

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
        return signal_detector.detectAndCompute(spec, None)
    def train_fingers(self):
        self.fingers = SignalFinder.get_finger(self.r, self.s)
    def find_signal(self, r, s):
        if self.fingers is None:
            self.train_fingers()
        pattern = SignalFinder.get_finger(r, s)
        return (0, 0)


def sync_signals(r1, s1, r2, s2, n):
    a_duration = s1.size // r1
    b_duration = s2.size // r2
    step_duration = a_duration / n
    steps = soundutil.sound_split_frames(s1, step_duration, r1)
