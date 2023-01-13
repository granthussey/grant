import os

import numpy as np
import pandas as pd
from grant.mtist.shared import Globals, GlobalParams
from mtist import mtist_utils as mu

# meta = Globals.meta
# abbreviations = GlobalParams.abbreviations
# regression_names = GlobalParams.regression_names
# formatted_regression_names = GlobalParams.formatted_regression_names
# gts = Globals.gts
# prefixes = GlobalParams.prefixes
# n_datasets = Globals.n_datasets


def calc_ptiles(ptile_th, gts=None, name=None):
    """Calcs percentiles based on threadhold for each gt
    If no gt supplied, will generate one based on current mtist package parameters"""

    if gts is None:
        gts = Globals.gts

    if name is None:

        return {
            key: np.percentile(
                np.abs(gts[key].replace(0, np.nan).melt().dropna()["value"]), ptile_th
            )
            for key in gts.keys()
        }

    else:

        return np.percentile(
            np.abs(gts[name].replace(0, np.nan).melt().dropna()["value"]), ptile_th
        )


def load_es_scores(meta=None, drop_floored=False):
    """Loads ES scores from file, does a few datapoints per inference type:
    1. stdev of scores of inference methods
    2. max_scores of scores of inference methods
    etc"""

    if meta is None:
        meta = Globals.meta

    scores = pd.DataFrame([])
    for name, prefix in zip(GlobalParams.regression_names, GlobalParams.prefixes):

        # _tmp = pd.read_csv(
        #     os.path.join(
        #         mu.GLOBALS.MTIST_DATASET_DIR,
        #         f"{prefix}_inference_result",
        #         f"{prefix}_es_scores.csv",
        #     )
        # ).rename(columns={"Unnamed: 0": f"asdf{name}"})

        _tmp = pd.read_csv(
            os.path.join(
                mu.GLOBALS.MTIST_DATASET_DIR,
                f"{prefix}_inference_result",
                f"{prefix}_es_scores.csv",
            )
        ).drop(columns="Unnamed: 0")

        #     _tmp = _tmp.rename(columns=dict(zip(_tmp.columns, [f"{prefix}_{col}" for col in _tmp.columns])))

        # Executive decision to remove the floored scores as we don't like those

        if drop_floored:
            _tmp = _tmp.drop(columns="floored")
        _tmp = _tmp.rename(columns={"raw": name})

        scores = _tmp.join(scores)

    scores = scores.join(meta)

    transformations = {
        "winners": lambda v: GlobalParams.prefixes[
            ([v[col] for col in GlobalParams.regression_names]).index(
                max([v[col] for col in GlobalParams.regression_names])
            )
        ],
        "max_scores": lambda v: max([v[col] for col in GlobalParams.regression_names]),
        "stdev": lambda v: np.std([v[col] for col in GlobalParams.regression_names]),
        "median_score": lambda v: np.median([v[col] for col in GlobalParams.regression_names]),
        "mean_score": lambda v: np.mean([v[col] for col in GlobalParams.regression_names]),
    }

    for key in transformations:
        scores = scores.join(scores.apply(transformations[key], axis=1).to_frame(name=key))

    return scores


def load_inferences():
    inf = {}

    for prefix in GlobalParams.prefixes:
        inf[prefix[0]] = [""] * Globals.n_datasets  # use first letter as the key
        path = os.path.join(f"{mu.GLOBALS.MTIST_DATASET_DIR}", f"{prefix}_inference_result")
        for did in range(Globals.n_datasets):
            cur_inf_result_path = os.path.join(f"{path}", f"{prefix}_inferred_aij_{did}.csv")
            cur_inf_result = pd.read_csv(cur_inf_result_path, header=None)
            inf[prefix[0]][did] = cur_inf_result

    return inf
