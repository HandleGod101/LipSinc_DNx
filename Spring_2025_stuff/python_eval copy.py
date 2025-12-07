import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from scipy.io import wavfile
from pesq import pesq
from pystoi import stoi

def evaluate_urbansound8k(testset_path, enhanced_path, metadata_path, target='enhanced'):
    # Read metadata
    metadata = pd.read_csv(metadata_path)  # Metadata CSV file
    class_names = metadata[['classID', 'class']].drop_duplicates().set_index('classID')['class'].to_dict()

    result = {cid: defaultdict(int) for cid in class_names.keys()}  # One result dict per class

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        try:
            class_id = row['classID']
            file_name = row['slice_file_name']
            fold = row['fold']

            # Build file paths
            clean_path = os.path.join(testset_path, "fileid_{}.wav".format(idx))
            if target == 'noisy':
                target_path = os.path.join(testset_path, "noisy", "noisy_fileid_{}.wav".format(idx))  # Noisy version assumed same structure
            else:
                target_path = os.path.join(enhanced_path, "enhanced_fileid_{}.wav".format(idx))

            # Load audio
            rate_clean, clean = wavfile.read(clean_path)
            rate_target, target_wav = wavfile.read(target_path)

            # if rate_clean != 16000 or rate_target != 16000:
            #     continue  # Skip non-16kHz files

            # Match lengths
            min_len = min(len(clean), len(target_wav))
            clean = clean[:min_len]
            target_wav = target_wav[:min_len]

            if np.all(clean == 0) or np.all(target_wav == 0):
                continue

            # Compute metrics
            result[class_id]['pesq_wb'] += pesq(16000, clean, target_wav, 'wb') * min_len
            result[class_id]['pesq_nb'] += pesq(16000, clean, target_wav, 'nb') * min_len
            result[class_id]['stoi'] += stoi(clean, target_wav, rate_clean) * min_len
            result[class_id]['count'] += min_len

        except Exception as e:
            print(f"Failed on {file_name}: {e}")
            continue

    return result, class_names


if __name__ == '__main__':
    enhanced_path = '//data/users/tqiu5/CleanUNet/exp/DNS-large-high/speech_urban/150k'   # change this
    testset_path = '/data/users/tqiu5/CleanUNet/dns/training_set/urbansound8k/clean'        # change this
    metadata_path = '/data/users/tqiu5/CleanUNet/urbansound8k/metadata/UrbanSound8K.csv'  # UrbanSound8K metadata CSV
    target = 'enhanced'  # or 'noisy'

    result, class_names = evaluate_urbansound8k(testset_path, enhanced_path, metadata_path, target)

    # Print results
    for cid in sorted(class_names.keys()):
        res = result[cid]
        if res['count'] > 0:
            print(f"Class {cid} ({class_names[cid]}): PESQ_WB = {res['pesq_wb']/res['count']:.3f}, "
                  f"PESQ_NB = {res['pesq_nb']/res['count']:.3f}, "
                  f"STOI = {res['stoi']/res['count']:.3f}")
        else:
            print(f"Class {cid} ({class_names[cid]}): No valid samples.")
