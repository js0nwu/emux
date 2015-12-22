import scipy.signal as signal

import scipy.io.wavfile as wav

import numpy
import cv2

import soundutil

FIT_THRESHOLD = 1.0

MAX_MATCH_NUM = 10

SPEC_MAX = 20
SPEC_MIN = -36

detector = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


class SignalFinder(object):
    def __init__(self, r, s):
        self.r = r
        self.s = s
        self.fingers = None

    @staticmethod
    def generate_spectrogram(r, s):
        specgram = signal.spectrogram(s, fs=r)
        spec = numpy.log(specgram[2])
        spec[spec == -numpy.inf] = 0
        spec = numpy.clip(spec, SPEC_MIN, SPEC_MAX)
        spec -= SPEC_MIN
        spec /= (SPEC_MAX - SPEC_MIN)
        spec *= 255
        spec = spec.astype(dtype=numpy.uint8)
        return (specgram[0], specgram[1], spec)

    @staticmethod
    def get_finger(r, s):
        specgram = SignalFinder.generate_spectrogram(r, s)
        # time and keypoints
        return (specgram[0], specgram[1], detector.detectAndCompute(specgram[2], None))

    def train_fingers(self):
        self.fingers = SignalFinder.get_finger(self.r, self.s)

    def find_signal(self, r, s):
        if self.fingers is None:
            self.train_fingers()
        query = SignalFinder.get_finger(r, s)
        f1 = query[0]
        t1 = query[1]
        kp1 = query[2][0]
        des1 = query[2][1]
        f2 = self.fingers[0]
        t2 = self.fingers[1]
        kp2 = self.fingers[2][0]
        des2 = self.fingers[2][1]
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:MAX_MATCH_NUM]
        src_pts = numpy.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = numpy.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, FIT_THRESHOLD)
        h = f1.size
        w = t1.size
        pts = numpy.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        times = [d[0][0] for d in dst].sort()
        timestart = times[0]
        timelength = times[-1] - times[0]
        return (timestart, timelength)


def sync_signals(r1, s1, r2, s2, n):
    a_duration = s1.size // r1
    b_duration = s2.size // r2
    step_duration = a_duration / n
    steps = soundutil.sound_split_frames(s1, step_duration, r1)


r1, s1 = wav.read('angry.wav')
r2, s2 = wav.read('sad.wav')
cv2.imwrite('1.jpg', SignalFinder.generate_spectrogram(r1, s1)[2])
cv2.imwrite('2.jpg', SignalFinder.generate_spectrogram(r2, s2)[2])
s1 = s1[15000:23000]
sm = SignalFinder(r2, s2)
sm.train_fingers()
print(sm.find_signal(r1, s1))
