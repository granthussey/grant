import os

import numpy as np
import pandas as pd

from mtist import assemble_mtist as am
from mtist import master_dataset_generation as mdg
from mtist import mtist_utils as mu


class GlobalParams:
    # These are just the default ones
    regression_names = ["default_score", "ridge_score", "lasso_score", "elasticnet_score"]
    prefixes = ["default", "ridge_CV", "lasso_CV", "elasticnet_CV"]
    formatted_regression_names = [
        "LinearRegression",
        "RidgeRegression",
        "LassoRegression",
        "ElasticNetRegression",
    ]
    abbreviations = [each[0] for each in regression_names]
    non_lr_methods = ["r", "l", "e"]


class Globals:

    ## These below should be set from the iPython interface when using this package###

    # mdg.MASTER_DATASET_DEFAULTS.random_seeds = mdg.MASTER_DATASET_DEFAULTS.expanded_random_seeds

    # mu.GLOBALS.GT_NAMES = [
    #     "3_sp_gt_2e1m",
    #     "3_sp_gt_2e1m_periodical",
    #     "3_sp_gt_3c",
    #     "3_sp_gt_2c1m",
    #     "3_sp_gt_2c1e",
    #     "3_sp_gt_3e",
    #     "3_sp_gt_2e1c",
    #     "3_sp_gt_2e1c_flip",
    # ]

    # mu.GLOBALS.GT_DIR = "gt_oldnew"
    # mu.GLOBALS.MTIST_DATASET_DIR = "expanded_mtist_datasets_oldnew_gt"
    # am.ASSEMBLE_MTIST_DEFAULTS.SAMPLING_FREQ_PARAMS = [3, 5, 8, 10, 15]
    # am.ASSEMBLE_MTIST_DEFAULTS.N_TIMESERIES_PARAMS = [3, 5, 10, 25, 50, 75, 100]

    n_datasets = mu.calculate_n_datasets()

    # Read in metadata
    meta = (
        pd.read_csv(os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, f"mtist_metadata.csv"))
        .drop(columns="Unnamed: 0")
        .set_index("did")
    )

    # Read in ground truths
    gt_names = mu.GLOBALS.GT_NAMES

    paths_to_gts = [
        os.path.join(
            mu.GLOBALS.GT_DIR,
            "interaction_coefficients",
            ecosystem_name.replace("gt", "aij") + ".csv",
        )
        for ecosystem_name in gt_names
    ]

    gts = {
        gt_name: pd.read_csv(path_to_gt, header=None)
        for gt_name, path_to_gt in zip(gt_names, paths_to_gts)
    }

    # n_dids_per_eco = 630
