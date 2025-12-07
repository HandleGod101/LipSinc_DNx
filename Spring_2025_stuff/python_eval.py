# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import sys
from collections import defaultdict
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import wavfile 

from pesq import pesq
from pystoi import stoi


def evaluate_dns(testset_path, enhanced_path, target):
    reverb = 'no'
    result = defaultdict(int)

    for i in tqdm(range(200)):
        try:
            rate, clean = wavfile.read(os.path.join(testset_path, "clean", "clean_fileid_{}.wav".format(i)))
            if target == 'noisy':
                rate, target_wav = wavfile.read(os.path.join(testset_path, "noisy", "noisy_fileid_{}.wav".format(i)))
            else:
                rate, target_wav = wavfile.read(os.path.join(enhanced_path, "enhanced_fileid_{}.wav".format(i)))

            if rate != 16000:
                continue

            # match length
            min_len = min(len(clean), len(target_wav))
            clean = clean[:min_len]
            target_wav = target_wav[:min_len]

            if np.all(clean == 0) or np.all(target_wav == 0):
                continue

            result['pesq_wb'] += pesq(16000, clean, target_wav, 'wb') * min_len
            result['pesq_nb'] += pesq(16000, clean, target_wav, 'nb') * min_len
            result['stoi'] += stoi(clean, target_wav, rate) * min_len
            result['count'] += min_len

        except Exception as e:
            continue
        
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='dns', help='dataset')
    parser.add_argument('-e', '--enhanced_path', type=str, help='enhanced audio path')
    parser.add_argument('-t', '--testset_path', type=str, help='testset path')
    args = parser.parse_args()

    # enhanced_path = args.enhanced_path
    # testset_path = args.testset_path

    enhanced_path = '/data/users/tqiu5/CleanUNet/exp/DNS-large-high/speech_urban/150k/' #change path for predict
    testset_path = '/data/users/tqiu5/CleanUNet/dns/datasets/test_set/synthetic/no_reverb/urban_set/' #change path for label 
    target = 'enhanced'

    if args.dataset == 'dns':
        result = evaluate_dns(testset_path, enhanced_path, target)
        
    # logging
    for key in result:
        if key != 'count':
            print('{} = {:.3f}'.format(key, result[key]/result['count']), end=", ")
