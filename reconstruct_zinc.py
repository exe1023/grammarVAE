#!/usr/bin/env python2

from __future__ import print_function

import sys
#sys.path.insert(0, '..')
import molecule_vae
import numpy as np

# 0. Constants
nb_smiles = 100
chunk_size = 100
encode_times = 10
decode_times = 25

# 1. load the test smiles
smiles_file = '../../dropbox/data/dearomatic/250k_rndm_zinc_drugs_clean_dearomatic.smi'
smiles = [line.strip() for index, line in zip(xrange(nb_smiles), open(smiles_file).xreadlines())]

def reconstruct(model):
    decode_result = []

    for chunk_start in range(0, len(smiles), chunk_size):
        chunk = smiles[chunk_start: chunk_size]
        chunk_result = [[] for _ in range(len(chunk))]
        for _encode in range(encode_times):
            z1 = model.encode(chunk)
            this_encode = []
            for _decode in range(decode_times):
                _result = model.decode(z1)
                for index, s in enumerate(_result):
                    chunk_result[index].append(s)

        decode_result.extend(chunk_result)
    assert len(decode_result) == len(smiles)

    return decode_result

def save_decode_result(decode_result, filename):
    with open(filename, 'w') as fout:
        for s, cand in zip(smiles, decode_result):
            print(','.join([s] + cand), file=fout)

def cal_accuracy(decode_result):
    accuracy = [sum([1 for c in cand if c == s]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
    return (sum(accuracy) * 1.0 / len(accuracy))

# 2. test char model
if False:
    char_weights = "../../dropbox/grammar_vae/reproduce/zinc_vae_str_L56_E100_val.hdf5"
    char_model = molecule_vae.ZincCharacterModel(char_weights)
    decode_result = reconstruct(char_model)
    save_decode_result(decode_result, "../../dropbox/grammar_vae/reproduce/zinc_vae_str_L56_E100_reconstruct.csv")
    accuracy = cal_accuracy(decode_result)
    print('char:', accuracy)

# 3. test grammar
if True:
    grammar_weights = "../../dropbox/grammar_vae/reproduce/zinc_vae_grammar_L56_E100_val.hdf5"
    grammar_model = molecule_vae.ZincGrammarModel(grammar_weights)
    decode_result = reconstruct(grammar_model)
    save_decode_result(decode_result, "../../dropbox/grammar_vae/reproduce/zinc_vae_grammar_L56_E100_reconstruct.csv")
    accuracy = cal_accuracy(decode_result)
    print('grammar:', accuracy)
