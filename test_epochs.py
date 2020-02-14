import sys
import time
from multiprocessing import Pool, cpu_count
import numpy as np
from torch import nn, manual_seed

sys.path.insert(0, 'graph_enc_dec')
from graph_enc_dec import data_sets as ds
from graph_enc_dec import graph_clustering as gc
from graph_enc_dec.architecture import GraphEncoderDecoder
from graph_enc_dec.model import Model, LinearModel
from graph_enc_dec.standard_architectures import ConvAutoencoder, FCAutoencoder
from graph_enc_dec import utils


SEED = 15
N_CPUS = cpu_count()
VERBOSE = False
SAVE = True
SAVE_PATH = './results/node_pert'
EVAL_F = 5

MAX_EPOCHS = 500

EXPS = [
        {'type': 'Enc_Dec',  # 4000
         'f_enc': [1, 20, 20, 30, 30, 30],
         'n_enc': [256, 64, 32, 16, 8, 4],
         'f_dec': [30, 30, 30, 30, 30, 30],
         'n_dec': [4, 8, 16, 32, 64, None],
         'f_conv': [30, 30, 1],
         'ups': gc.WEI,
         'downs': gc.WEI,
         'fmt': ['o-', 'o--']},
        {'type': 'Enc_Dec',
         'f_enc': [1, 30, 30, 30, 30, 30],
         'n_enc': [256, 64, 32, 16, 8, 4],
         'f_dec': [30, 30, 30, 30, 30, 30],
         'n_dec': [4, 8, 16, 32, 64, None],
         'f_conv': [30, 30, 1],
         'ups': gc.GF,
         'downs': gc.GF,
         'fmt': ['X-', 'X--']},
        {'type': 'Enc_Dec',
         'f_enc': [1, 5, 5],
         'n_enc': [256, 16, 4],
         'f_dec': [5, 5, 5],
         'n_dec': [4, 16, None],
         'f_conv': [30, 30, 1],
         'ups': gc.WEI,
         'downs': gc.WEI,
         'fmt': ['P-', 'P--']},
        ]

