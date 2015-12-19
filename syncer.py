import scipy.signal as signal

import soundutil

def sync_signals(r1, s1, r2, s2, n):
    a_duration = s1.size // r1
    b_duration = s2.size // r2
    step_duration = a_duration / n
    steps = soundutil.sound_split_frames(s1, step_duration, r1)
