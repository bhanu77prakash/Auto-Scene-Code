import numpy as np
import json
import pandas as pd
import argparse
import os

from sympy import re

LM_DATA = "./"
FT_DATA = "../top_150_50_new"
INP_THRESH = 5

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=False, default=f"{FT_DATA}/ft_test.json", type=str)
parser.add_argument("--output_folder", required=False, default="scenes", type=str)
parser.add_argument("--n_samples", required=False, default=30, type=int)
parser.add_argument("--missing", required=False, default=-1, type=int)
args = parser.parse_args()

if os.path.exists(args.output_folder) == False:
    os.makedirs(args.output_folder, exist_ok=True)

test_data = json.load(open(args.file))
test_data = [x for x in test_data if len(x[0])-1 > INP_THRESH + max(args.missing-1, 4)]



if args.n_samples != -1:
    idxs = list(range(len(test_data)))
    selected_samples = [test_data[id] for id in np.random.choice(idxs, min(args.n_samples, len(test_data)), replace=False)]

    final_samples = []
    paths = []
    print(f"unique paths before saving {len(set([x[-1] for x in selected_samples]))}")
    for id, sample in enumerate(selected_samples):
        sample[0] = [x for x in sample[0] if len(x.strip().split(",")) == 3]
        if args.missing != -1: 
            sample[0] = np.random.choice(sample[0], len(sample[0]) - (args.missing - 1), replace=False).tolist()
        else:
            num_mask = np.random.randint(1, 5)
            sample[0] = np.random.choice(sample[0], len(sample[0]) - (num_mask), replace=False).tolist()

        with open(os.path.join(f"{args.output_folder}", "input", f"sample{id}.json"), "w") as fp:
            json.dump([sample], fp)
        final_samples.append(sample)
        
        try:
            paths.append(sample[-1]) 
        except:
            import pdb; pdb.set_trace()
    with open(os.path.join(f"{args.output_folder}", f"all_samples.json"), "w") as fp:
        json.dump(final_samples, fp)
    print(f"Added {len(final_samples)} files and {len(set(paths))} unique images")

else:
    idxs = list(range(len(test_data)))
    selected_samples = [test_data[id] for id in np.random.choice(idxs, args.n_samples, replace=False)]

    final_samples = []
    for id, sample in enumerate(test_data):
        sample[0] = [x for x in sample[0] if len(x.strip().split(",")) == 3]
        if args.missing != -1: 
            sample[0] = np.random.choice(sample[0], len(sample[0]) - (args.missing - 1), replace=False).tolist()
        else:
            num_mask = np.random.randint(1, 5)
            sample[0] = np.random.choice(sample[0], len(sample[0]) - (num_mask), replace=False).tolist()

        with open(os.path.join(f"{args.output_folder}", "input", f"sample{id}.json"), "w") as fp:
            json.dump([sample], fp)
        final_samples.append(sample)
    with open(os.path.join(f"{args.output_folder}", f"all_samples.json"), "w") as fp:
        json.dump(final_samples, fp)