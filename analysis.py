import scipy.io.wavfile as wav
import scipy.signal as signal
import pathlib

from .audiofiles.utility import *

from .soundutil import *

def clip_info(wavefile):
    (r, s) = wav.read(wavefile)
    return (r, s.size / r)

def normalize_read(wavefile):
    resample_rate = 22000
    (r, s) = wav.read(wavefile)
    return (resample_rate, signal.resample(pcm2float(s), s.size * (resample_rate / r)))

def train_template():
    frame_time = 0.1
    subframe_time = 0.02
    breath_paths = [str(f) for f in pathlib.Path('./breaths').iterdir() if f.is_file() and f.name.startswith('breath')]
    if len(breath_paths) == 0:
        return None
    breath_signals = []
    rate = 0;
    for b in breath_paths:
        (r, s) = normalize_read(b)
        if rate == 0:
            rate = r
        breath_signals.append(s)
    breaths_all = combine_sigs(breath_signals)
    breath_frames = sound_split_frames(breaths_all, rate, frame_time)
    breath_matrices = [cep_matrix(sound_split_frames(x, rate, subframe_time), rate) for x in breath_frames]
    template_cep_matrix = numpy.zeros(breath_matrices[0].shape)
    for bm in breath_matrices:
        template_cep_matrix = numpy.add(template_cep_matrix, bm)
    template_cep_matrix = numpy.divide(template_cep_matrix, len(breath_matrices))
    return template_cep_matrix

def find_breaths(read_sig, read_rate, template_cep_matrix):
    frame_time = 0.1
    subframe_time = 0.02
    breath_min = 4
    read_frames = sound_split_frames(read_sig, read_rate, frame_time)
    # refine these with machine learning
    zcr_limit = -0.5
    cep_limit = -0.2
    zcrs = [zero_cross_rate(r) for r in read_frames]
    zcr_threshold = numpy.mean(zcrs) + zcr_limit * numpy.std(zcrs)
    unvoiced_frames = [i for i in range(len(read_frames)) if zcrs[i] < zcr_threshold]
    read_matrices = [cep_matrix(sound_split_frames(read_frames[i], read_rate, subframe_time), read_rate) for i in unvoiced_frames]
    # for some reason linalg norm is slower than doing it manually, can assume everything is real
    # http://stackoverflow.com/a/10674608
    # norms = [linalg.norm(numpy.subtract(f, template_cep_matrix)) for f in unvoiced_frames]
    diffs = numpy.asarray([numpy.subtract(f, template_cep_matrix) for f in unvoiced_frames])
    norms = numpy.sqrt((diffs * diffs).sum(axis = 1).sum(axis = 1))
    cep_threshold = numpy.mean(norms) + cep_limit * numpy.std(norms)
    guess_frames = [unvoiced_frames[i] for i in range(len(unvoiced_frames)) if norms[i] < cep_threshold]
    guess_breaths = group_consecutive(guess_frames, 1)
    breaths = [b for b in guess_breaths if len(b) >= breath_min]
    breath_lengths = [float(b[-1] - b[0] + 1) * frame_time for b in breaths]
    breath_energies = [numpy.sum([signal_energy(read_frames[f]) for f in b]) for b in breaths]
    return list(zip(breath_lengths, breath_energies))
