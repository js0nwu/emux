from .analysis import *

import os
import pathlib

import time

import numpy

# should probably put the training model in a secure database
# try and process stream instead of file
def process_clip_file(filepath):
    train_dir = './train'
    # use python exceptions
    if os.path.isfile(filepath) == False:
        print("file not found")
        return None
    if os.path.isdir(train_dir) == False:
        os.mkdir(train_dir)
    train_paths = [str(f) for f in pathlib.Path(train_dir).iterdir() if f.is_file and str(f).endswith('.npy')]
    template_cep_matrix = None
    if len(train_paths) > 0:
        train_file = train_paths[0]
        template_cep_matrix = numpy.load(train_file)
    else:
        template_cep_matrix = train_template()
        if template_cep_matrix is None:
            return None
        template_name = str(time.time())
        numpy.save(train_dir + '/' + template_name, template_cep_matrix)
    if template_cep_matrix is None:
        print("template cep matrix failed to generate or load")
        return None
    (read_rate, read_sig) = normalize_read(filepath)
    return find_breaths(read_sig, read_rate, template_cep_matrix)
