import glob
import os

import numpy as np
import pandas as pd
import seaborn as sns
from grant import grant
from grant.mtist.analyze_inference.utils import calc_ptiles
from grant.mtist.shared import Globals, GlobalParams
from matplotlib import pyplot as plt
from mtist import infer_mtist as im
from mtist import mtist_utils as mu


def create_counts_dict(inf):
    """creates the count dictionary where correct inference is counted for
    each interaction coefficient"""

    th = 0

    # counts - how many times did you get it correct, for a certain inference method and ecosystem?
    # totals - ignore these, we can calculate on-the-fly later. Basically, thresholding.

    counts = {}
    # totals = {}
    for key in inf.keys():
        # Store counts here
        results = []
        # totals_results = []
        # Loop over all ecosystems for each inference method
        for cur_eco in Globals.meta["ground_truth"].unique():
            dids = Globals.meta.query("ground_truth == @cur_eco").index

            tmp = np.zeros((Globals.gts[cur_eco].shape), dtype=int)
            # tmp_totals = np.zeros((Globals.gts[cur_eco].shape), dtype=int)
            # Loop over dids of that specific ecosystem
            for did in dids:

                cur_inf_result = inf[key][did]
                nonzero_mask = Globals.gts[cur_eco] != 0

                # th_nz_cur_inf_result = cur_inf_result[nonzero_mask][np.abs(cur_inf_result[nonzero_mask]) > th]
                nz_cur_inf_result = cur_inf_result[nonzero_mask]
                th_nz_eco = Globals.gts[cur_eco][np.abs(Globals.gts[cur_eco][nonzero_mask]) > th]

                cur_correct = np.sign(nz_cur_inf_result) == np.sign(th_nz_eco)
                # cur_totals = len(th_nz_eco.melt().dropna()['value'])
                # cur_totals = ~th_nz_eco.isna()

                tmp = tmp + cur_correct.to_numpy(dtype=int)
                # tmp_totals = tmp_totals + cur_totals.to_numpy(dtype=int)

            results.append(tmp)
            # totals_results.append(tmp_totals)

        counts[key] = results
    #     totals[key] = total_results

    # grant.save_pickle(counts, "counts")
    return counts


def create_aijs_df(inf, th=0, save=False, save_dir=None, gts=None, meta=None):
    """Creates aijs_df (see table below)

    Args:
        inf (df): dataframe of all inference scores
        th (int, optional): Threshold for. Defaults to 0.
        save (bool, optional): Toggle to save aijs_df.
                These are saved into directory "./chunked_aijs_df" with filenames of form "{regression_name}_aijs_df_did_{did}.pickle". Defaults to False.

    Returns:
        df: the aijs_df (see below)

    +---------+------------+--------+------------+------------+------------+------+-----------------+
    |         | inf_result | coeff  | inf_or_not | coeff_type | inf_method | did  | cid             |
    +---------+------------+--------+------------+------------+------------+------+-----------------+
    | 0       | -0.242062  | (0, 0) | True       | -1.0       | e          | 4402 | e_4402_(0, 0)_n |
    +---------+------------+--------+------------+------------+------------+------+-----------------+
    | 1       | -0.330029  | (0, 1) | True       | -1.0       | e          | 4402 | e_4402_(0, 1)_n |
    +---------+------------+--------+------------+------------+------------+------+-----------------+
    | ...     | ...        | ...    | ...        | ...        | ...        | ...  | ...             |
    +---------+------------+--------+------------+------------+------------+------+-----------------+
    | n_coefs | ...        | ...    | ...        | ...        | ...        | ...  | ...             |
    +---------+------------+--------+------------+------------+------------+------+-----------------+

    For every row (coefficient in the Aij matrix of that did):
        * inf_result: the actual number inferred for that coefficient
        * coeff: coordinates in the Aij matrix
        * inf_or_not: True if sign properly inferred, else False
        * coeff_type: -1 if negative, 1 if positive, else 0 if 0
        * inf_method: abbreviation for inference method (d for default/LinearRegression,
                l for Lasso, e for ElasticNet, r for Ridge)
        * did: dataset id
        * cid: coefficient id of form {inference_method}_{did}_{coeff}_{coeff_type_abbreviation},
                where coeff_type_abbreviation is n for neg, p for positive, and z for zero


    """
    #################################################################################################################

    if save_dir is None:
        save_dir = "chunked_aijs_df"

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    labels = {
        n_species: [(i, j) for i in range(n_species) for j in range(n_species)]
        for n_species in [3, 10, 100]
    }

    #################################################################################################################
    # n_total_aijs = np.sum((Globals.meta["n_species"] * Globals.meta["n_species"]).values) * len(inf.keys())
    # n_total_dfs_that_will_be_made = n_datasets * len(inf.keys())
    # n_dfs_per_inf_method = n_datasets

    for method_key in inf.keys():
        print(method_key)
        n_datasets = len(inf[method_key])
        list_of_inferences = inf[method_key]

        # Create this inference
        aijs = np.array([""] * mu.calculate_n_datasets(), dtype=object)
        counter = 0

        for did in range(n_datasets):

            cur_inf = list_of_inferences[did]

            n_species = len(cur_inf)
            cur_gt_name = Globals.meta.loc[did, "ground_truth"]
            cur_gt = Globals.gts[cur_gt_name]

            all_coefs = cur_inf.values.reshape(-1)

            # now, for each coeff, check if correct (True), incorrect or not inferred or zero (False) ... (np.nan == np.nan evals as false)

            nonzero_mask = cur_gt != 0
            nz_th_mask = np.abs(cur_gt[nonzero_mask]) > th

            # print(cur_gt)
            # print(cur_inf)

            is_it_correct = (
                np.sign(cur_gt[nz_th_mask]) == np.sign(cur_inf[nonzero_mask])
            ).values.reshape(-1)

            # record type of interaction
            type_of_interaction = np.sign(cur_gt).values.reshape(-1)

            cur_aij_df = pd.DataFrame(
                [all_coefs, labels[n_species], is_it_correct, type_of_interaction],
                index=["inf_result", "coeff", "inf_or_not", "coeff_type"],
            ).T.assign(inf_method=method_key, did=did)

            aijs[counter] = cur_aij_df

            counter = counter + 1

        aijs_df = pd.concat(aijs)

        #################################################

        aijs_df.loc[:, "letter_type"] = aijs_df.loc[:, "coeff_type"].replace(
            {-1: "n", 1: "p", 0: "z"}
        )
        aijs_df = aijs_df.assign(
            cid=lambda v: v["inf_method"]
            + "_"
            + v["did"].astype(str)
            + "_"
            + v["coeff"].astype(str)
            + "_"
            + v["letter_type"]
        )
        aijs_df = aijs_df.drop(columns=["letter_type"])

        # SAVE THE FULL aijs_df
        if save:
            grant.save_pickle(
                aijs_df,
                f"aijs_df_infmethod_{GlobalParams.formatted_regression_names[GlobalParams.abbreviations.index(method_key)]}",
            )

            ###############
            # Now make did-by-did aijs_df

            cur_aij_df_name = f"aijs_df_infmethod_{GlobalParams.formatted_regression_names[GlobalParams.abbreviations.index(method_key)]}"

            path_to = lambda v: os.path.join(save_dir, v)

            # chunk by did
            for did in range(n_datasets):
                fn = f"{GlobalParams.formatted_regression_names[GlobalParams.abbreviations.index(method_key)]}_aijs_df_did_{did}"
                grant.save_pickle(aijs_df.loc[aijs_df["did"] == did], path_to(fn))

            ###

    return aijs_df


def recalc_100sp_es_with_th(th):

    hms = []

    sd = "high"  # used in .query command
    gt100 = Globals.gts["100_sp_gt"]

    load_inf = lambda did, inf_name: pd.read_csv(
        os.path.join(
            mu.GLOBALS.MTIST_DATASET_DIR,
            f"{inf_name}_inference_result",
            f"{inf_name}_inferred_aij_{did}.csv",
        ),
        header=None,
    )

    normalize = lambda dataframe, th: dataframe[
        dataframe.applymap(lambda v: np.abs(v) > th)
    ].fillna(0)

    idx = Globals.meta.query("n_species == 100 and seq_depth == @sd").index

    es = {}
    for inf_method in GlobalParams.prefixes:
        es[inf_method] = {}
        for did in idx:
            es[inf_method][did] = {}
            es[inf_method][did]["unth"] = im.calculate_es_score(gt100, load_inf(did, inf_method))
            #                 es[inf_method][did]['th'] = im.calculate_es_score(normalize(gt100, th), normalize(load_inf(did, inf_method), th))
            es[inf_method][did]["th"] = im.calculate_es_score(
                normalize(gt100, th), load_inf(did, inf_method)
            )

    ####################################################################################
    tmp = pd.DataFrame([])
    for inf_method in es.keys():
        cur_es = es[inf_method]
        cur_df = pd.DataFrame(cur_es).T.assign(method=inf_method, th_used=th)
        tmp = pd.concat([tmp, cur_df])

        hms.append(tmp)

    hm = pd.concat(hms).join(Globals.meta)

    return hm


def recalc_es_with_th(th):

    hms = []

    # sd = "high"  # used in .query command

    load_inf = lambda did, inf_name: pd.read_csv(
        os.path.join(
            mu.GLOBALS.MTIST_DATASET_DIR,
            f"{inf_name}_inference_result",
            f"{inf_name}_inferred_aij_{did}.csv",
        ),
        header=None,
    )

    threshold = lambda dataframe, th: dataframe[
        dataframe.applymap(lambda v: np.abs(v) > th)
    ].fillna(0)

    # idx = Globals.meta.query("seq_depth == @sd").index

    idx = Globals.meta.index

    es = {}
    for inf_method in GlobalParams.prefixes:
        es[inf_method] = {}
        for did in idx:

            cur_gt = Globals.gts[Globals.meta.loc[did, "ground_truth"]]

            es[inf_method][did] = {}
            es[inf_method][did]["unth"] = im.calculate_es_score(cur_gt, load_inf(did, inf_method))
            #                 es[inf_method][did]['th'] = im.calculate_es_score(threshold(gt100, th), threshold(load_inf(did, inf_method), th))
            es[inf_method][did]["th"] = im.calculate_es_score(
                threshold(cur_gt, th), load_inf(did, inf_method)
            )

    ####################################################################################
    tmp = pd.DataFrame([])
    for inf_method in es.keys():
        cur_es = es[inf_method]
        cur_df = pd.DataFrame(cur_es).T.assign(method=inf_method, th_used=th)
        tmp = pd.concat([tmp, cur_df])

        hms.append(tmp)

    hm = pd.concat(hms).join(Globals.meta)

    return hm


def recalc_es_with_th_at_median():

    hms = []

    # sd = "high"  # used in .query command

    load_inf = lambda did, inf_name: pd.read_csv(
        os.path.join(
            mu.GLOBALS.MTIST_DATASET_DIR,
            f"{inf_name}_inference_result",
            f"{inf_name}_inferred_aij_{did}.csv",
        ),
        header=None,
    )

    threshold = lambda dataframe, th: dataframe[
        dataframe.applymap(lambda v: np.abs(v) > th)
    ].fillna(0)

    # idx = Globals.meta.query("seq_depth == @sd").index

    idx = Globals.meta.index

    es = {}
    for inf_method in GlobalParams.prefixes:
        es[inf_method] = {}
        for did in idx:

            cur_gt = Globals.gts[Globals.meta.loc[did, "ground_truth"]]
            ptile = calc_ptiles(50)  # calc median
            th = ptile[Globals.meta.loc[did, "ground_truth"]]

            es[inf_method][did] = {}
            es[inf_method][did]["unth"] = im.calculate_es_score(cur_gt, load_inf(did, inf_method))
            #                 es[inf_method][did]['th'] = im.calculate_es_score(threshold(gt100, th), threshold(load_inf(did, inf_method), th))
            es[inf_method][did]["th"] = im.calculate_es_score(
                threshold(cur_gt, th), load_inf(did, inf_method)
            )

    ####################################################################################
    tmp = pd.DataFrame([])
    for inf_method in es.keys():
        cur_es = es[inf_method]
        cur_df = pd.DataFrame(cur_es).T.assign(method=inf_method, th_used=th)
        tmp = pd.concat([tmp, cur_df])

        hms.append(tmp)

    hm = pd.concat(hms).join(Globals.meta)

    return hm


def run_nonzero_analysis(expanded_dataset_dir, run_name):

    master_dset_loc = expanded_dataset_dir
    dir_prefix = run_name

    # Calculate the number of mtdids
    n_mdids = len(glob.glob(os.path.join(f"{master_dset_loc}/master_dataset_*.csv")))

    # Load the master dataset Globals.metadata
    mmeta = pd.read_csv(os.path.join(f"{master_dset_loc}", "master_metadata.csv")).set_index(
        "master_did"
    )

    mmeta["n_species"] = mmeta["name"].str.split("_").apply(lambda v: v[0]).astype(int)

    # Load and process the Globals.metadata results
    results = []
    for mdid in range(n_mdids):

        n_species = mmeta.loc[mdid, "n_species"]

        df = pd.read_csv(os.path.join(f"{master_dset_loc}", f"master_dataset_{mdid}.csv")).drop(
            columns="Unnamed: 0"
        )

        species = [f"species_{i_sp}" for i_sp in range(n_species)]

        result = (df[species] > 0).sum()
        result.index = f"mdid_{mdid}_" + result.index

        results.append(result)
    df = pd.concat(results).reset_index()
    df["mdid"] = df["index"].str.split("_").apply(lambda v: int(v[1]))
    df["sp_name"] = df["index"].str.split("_").apply(lambda v: "species_" + v[-1])
    df = df.set_index("mdid")
    df = df.join(mmeta)
    df = df.rename(columns={0: "n_nonzero"})
    df["n_zero"] = np.abs(100 - df["n_nonzero"])
    df.index.name = "mdid"
    df

    # REport the number in percentages as well
    tmp = (
        (
            df.reset_index().pivot(
                values="n_zero",
                columns=["name", "sp_name"],
                index=["seed", "noise"],
            )
        ).sum(axis=0)
        / 30000
        * 100
    )

    # #### CREATE HIST SHARED FIG ####
    # fig_hist_shared, axes = grant.easy_subplots(
    #     ncols=3,
    #     nrows=2,
    #     base_figsize=(6, 4),
    #     gridspec_kw=dict(hspace=0.35),
    #     sharex=True,
    #     sharey=True,
    # )

    # for i, name in enumerate(df["name"].unique()):

    #     if name == "100_sp_gt":
    #         continue

    #     ax = axes[i]

    #     sns.histplot(
    #         data=df.query("name==@name and n_zero > 0"),
    #         x="n_nonzero",
    #         hue="sp_name",
    #         ax=ax,
    #         stat="density",
    #     )
    #     sns.despine()
    #     ax.set_title(name)
    #     # ax.set_ylabel('Density')
    #     ax.set_xlabel("Number of times zero")

    # #### CREATE HIST INDEPENDENT FIG ####
    # fig_hist_independent, axes = grant.easy_subplots(
    #     ncols=3, nrows=2, base_figsize=(6, 4), gridspec_kw=dict(hspace=0.35)
    # )

    # for i, name in enumerate(df["name"].unique()):

    #     if name == "100_sp_gt":
    #         continue

    #     ax = axes[i]

    #     sns.histplot(
    #         data=df.query("name==@name and n_zero > 0"),
    #         x="n_nonzero",
    #         hue="sp_name",
    #         ax=ax,
    #         stat="density",
    #     )
    #     sns.despine()
    #     ax.set_title(name)
    #     # ax.set_ylabel('Density')
    #     ax.set_xlabel("Number of times zero")

    # #### CREATE LARGE HEATMAP INDEPENDENT FIG ####
    # fig_save_2, ax = plt.subplots(figsize=(40, 30))

    # # Since this is a VERY large pivot'd table (if not agging), I will agg across mdids
    # sns.heatmap(
    #     df.reset_index()
    #     .pivot(
    #         values="n_zero",
    #         columns=["name", "sp_name"],
    #         index=["seed", "noise"],
    #     )
    #     .T,
    #     cmap="plasma",
    #     ax=ax,
    #     center=50,
    # )

    # #### CREATE SMALL HEATMAP INDEPENDENT FIG ####
    # fig_save1, ax = plt.subplots(figsize=(20, 50))

    # sns.heatmap(
    #     (
    #         df.reset_index()
    #         .set_index(["name", "sp_name"])
    #         .loc[tmp[tmp > 0].index]
    #         .reset_index()
    #         .pivot(
    #             values="n_zero",
    #             columns=["name", "sp_name"],
    #             index=["noise", "seed"],
    #         )
    #     ),
    #     cmap="plasma",
    #     center=50,
    #     ax=ax,
    #     linewidth=0.21,
    #     vmin=0,
    #     vmax=100,
    #     square=True,
    #     cbar=False,
    # )

    if not os.path.isdir("nonzero_readouts"):
        os.mkdir("nonzero_readouts")

    #### SAVE STUFF

    tmp[tmp > 0].to_csv(os.path.join("nonzero_readouts", f"{dir_prefix}_perc_zero.csv"))
    tmp.to_csv(os.path.join("nonzero_readouts", f"{dir_prefix}_perc_zero_full.csv"))
    # grant.savefig(fig_save1, os.path.join("nonzero_readouts", f"{dir_prefix}_heatmap"))
    # grant.savefig(fig_save_2, os.path.join("nonzero_readouts", f"{dir_prefix}_big_heatmap"))
    # grant.savefig(
    #     fig_hist_shared, os.path.join("nonzero_readouts", f"{dir_prefix}_fig_hist_shared")
    # )
    # grant.savefig(
    #     fig_hist_independent, os.path.join("nonzero_readouts", f"{dir_prefix}_fig_hist_independent")
    # )

    # plt.close("all")
