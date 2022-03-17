#!/usr/bin/env python

import argparse
import glob
import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import seaborn as sns
from grant.mdsine2 import utils as mdu
from matplotlib import pyplot as plt
from mtist import infer_mtist as im

import mdsine2 as md2
from mdsine2.names import STRNAMES

# if __name__ == "__main__":

######################################################################

parser = argparse.ArgumentParser(description="thing")
parser.add_argument("-e", "--ecosystem", help="ecosystem")
parser.add_argument("-m", "--mtistpath", help="MTIST directory")
parser.add_argument("-t", "--truthpath", help="GT directory")
parser.add_argument("-n", "--nsamples", help="n_samples")

args = parser.parse_args()

ECOSYSTEM = args.ecosystem
MTIST_PATH = args.mtistpath
GT_PATH = args.truthpath

# if args.ecosystem is not None:
#     try:
#         ECOSYSTEM = args.ecosystem
#     except Exception as e:
#         print(e)
#         print("something went wrong")
# else:
#     sys.exit()

# if args.mtistpath is not None:
#     try:
#         MTIST_PATH = args.mtistpath
#     except Exception as e:
#         print(e)
#         print("something went wrong")
# else:
#     sys.exit()

# if args.GT_PATH is not None:
#     try:
#         GT_PATH = args.truthpath
#     except Exception as e:
#         print(e)
#         print("something went wrong")
# else:
#     sys.exit()

if args.nsamples is not None:
    try:
        N_SAMPLES = int(args.nsamples)
    except Exception as e:
        print(e)
        print("something went wrong")
else:
    print("default n_samples is 50, this is wrong, fix this")
    N_SAMPLES = 50

# MTIST_PATH = "/Users/granthussey/Lab/Schluter/MTIST_post_revision/MTIST2.0/final_mtist2/paper_story/10species/mtist2_10sp_assembled/"
# GT_PATH = "/Users/granthussey/Lab/Schluter/MTIST_post_revision/MTIST2.0/final_mtist2/paper_story/10species/mtist2_fixed_10sp_gts/interaction_coefficients"

# ECOSYSTEM = "10_sp_gt_196"

ENCLOSED_FOLDER = "run_over_mtist"

SAVE_PATH = os.path.join(ENCLOSED_FOLDER, f"mdsine2_inference_{ECOSYSTEM}")
OUTPUT_PATH = os.path.join(SAVE_PATH, "output")

RESULTS_PATH = os.path.join(SAVE_PATH, "results")

# N_SAMPLES = 50

####
# make req paths
for path in [ENCLOSED_FOLDER, SAVE_PATH, OUTPUT_PATH, RESULTS_PATH]:
    try:
        os.mkdir(path)
    except:
        pass
####

######################################################################

meta = (
    pd.read_csv(os.path.join(MTIST_PATH, "mtist_metadata.csv"))
    .drop(columns="Unnamed: 0")
    .set_index("did")
)

paths = glob.glob(os.path.join(MTIST_PATH, "dataset_*.csv"))

path_dict = dict(
    zip([int(path.split("/")[-1].split("_")[-1].split(".")[0]) for path in paths], paths)
)

# begin per-ecosystem inference #
dids = meta.query("n_species == 10 and seq_depth == 'high' and ground_truth == @ECOSYSTEM").index

# load all data for that ecosystem from mdsine2
df = pd.DataFrame([])
for did in dids:
    path = path_dict[did]
    # tmp = pd.read_csv(path).drop(columns='Unnamed: 0')
    tmp = mdu.format_dataset(path)
    df = pd.concat([df, tmp])

reads, taxonomy, qpcr, metadata = mdu.extract_mdsine2_dataframes(df)

taxonomy.to_csv(os.path.join(SAVE_PATH, "taxonomy.tsv"), sep="\t", index=None)
reads.to_csv(os.path.join(SAVE_PATH, "reads.tsv"), sep="\t", index=None)
qpcr.to_csv(os.path.join(SAVE_PATH, "qpcr.tsv"), sep="\t", index=None)
metadata.to_csv(os.path.join(SAVE_PATH, "metadata.tsv"), sep="\t", index=None)

mcmc_negbin, mcmc_uc0, reg = mdu.infer_mdsine2(
    SAVE_PATH, OUTPUT_PATH, viz=False, n_samples=N_SAMPLES
)

truth = pd.read_csv(os.path.join(GT_PATH, ECOSYSTEM.replace("gt", "aij")) + ".csv", header=None)

# Get base inferred
inferred = np.nanmean(
    mcmc_uc0.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section="posterior"), axis=0
)

# Get the self-interactions
self_int = -1 * np.nanmean(
    mcmc_uc0.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section="posterior"),
    axis=0,
)

# insert the self-interactions
for i in range(10):
    inferred[i, i] = self_int[i]

# calculate ES score
es = np.array([im.calculate_es_score(truth.values, inferred)])

# save everything
np.savetxt(os.path.join(RESULTS_PATH, f"{ECOSYSTEM}_truth.csv"), truth.values)
np.savetxt(os.path.join(RESULTS_PATH, f"{ECOSYSTEM}_inferred.csv"), inferred)
np.savetxt(os.path.join(RESULTS_PATH, f"{ECOSYSTEM}_es.csv"), es)
