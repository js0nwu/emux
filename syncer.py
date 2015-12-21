import scipy.signal as signal
import soundutil

import scipy.io.wavfile as wav

import numpy
import cv2

import scipy.ndimage.morphology as morph
import scipy.ndimage.filters as filters

from operator import itemgetter

MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

PEAK_SORT = True

FAN_VALUE = 15

FLANN_INDEX_KDTREE = 1

SPEC_MAX = 20
SPEC_MIN = -36

PEAK_NEIGHBORHOOD_SIZE = 20

AMP_MIN = 10

# method based from https://github.com/worldveil/dejavu

class SignalFinder(object):
    def __init__(self, r, s):
        self.r = r
        self.s = s
        self.fingers = None

    @staticmethod
    def generate_spectrogram(r, s):
        spec = numpy.log(signal.spectrogram(s, fs=r)[2])
        spec[spec == -numpy.inf] = 0
        spec = numpy.clip(spec, SPEC_MIN, SPEC_MAX)
        spec -= SPEC_MIN
        spec /= (SPEC_MAX - SPEC_MIN)
        spec *= 255
        spec = spec.astype(dtype=numpy.uint8)
        return spec
    @staticmethod
    def get_peaks(arr2D):
        struct = morph.generate_binary_structure(2, 1)
        neighborhood = morph.iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)
        local_max = filters.maximum_filter(arr2D, footprint=neighborhood) == arr2D
        background = (arr2D == 0)
        eroded_background = morph.binary_erosion(background, structure = neighborhood, border_value = 1)
        detected_peaks = local_max - eroded_background
        amps = arr2D[detected_peaks]
        j, i = numpy.where(detected_peaks)
        amps = amps.flatten()
        peaks = zip(i, j, amps)
        peaks_filtered = [x for x in peaks if x[2] > AMP_MIN]
        frequency_idx = [x[1] for x in peaks_filtered]
        time_idx = [x[0] for x in peaks_filtered]
        return zip(frequency_idx, time_idx)
    @staticmethod
    def get_finger(r, s):
        peaks = SignalFinder.get_peaks(SignalFinder.generate_spectrogram(r, s))
        if PEAK_SORT:
            peaks.sort(key = itemgetter(1))
        for i in range(len(peaks)):
            for j in range(1, FAN_VALUE):
                if (i + j) < len(peaks):
                    freq1 = peaks[i][0]
                    freq2 = peaks[i + j][i]
                    t1 = peaks[i][1]
                    t2 = peaks[i + j][1]
                    t_delta = t2 - t1
                    if t_delta >= MIN_HASH_TIME_DELTA and t_delta <= MAX_HASH_TIME_DELTA:


    def train_fingers(self):
        self.fingers = SignalFinder.get_finger(self.r, self.s)

    def find_signal(self, r, s):
        if self.fingers is None:
            self.train_fingers()
        query = SignalFinder.get_finger(r, s)


def sync_signals(r1, s1, r2, s2, n):
    a_duration = s1.size // r1
    b_duration = s2.size // r2
    step_duration = a_duration / n
    steps = soundutil.sound_split_frames(s1, step_duration, r1)

r1, s1 = wav.read('angry.wav')
r2, s2 = wav.read('sad.wav')
cv2.imwrite('1.jpg', SignalFinder.generate_spectrogram(r1, s1))
cv2.imwrite('2.jpg', SignalFinder.generate_spectrogram(r2, s2))
s1 = s1[15000:23000]
sm = SignalFinder(r2, s2)
sm.train_fingers()
print(sm.find_signal(r1, s1))