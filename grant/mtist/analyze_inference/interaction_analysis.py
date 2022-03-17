from grant import grant
import numpy as np
import os
import pandas as pd
import ast
import seaborn as sns
from matplotlib import pyplot as plt
from grant.mtist.shared import Globals, GlobalParams
from grant.mtist.analyze_inference.utils import calc_ptiles, load_inferences


# ptile_th_dict = {
#     **{"100_sp_gt": calc_ptiles(94)["100_sp_gt"]},
#     **{each: calc_ptiles(50)[each] for each in ["10_sp_gt_1", "10_sp_gt_2", "10_sp_gt_3"]},
# }

ptile_th_dict = {
    gt_name: 0 for gt_name in Globals.gt_names
}  # all zero since 3-species, capture all

# Keys: Ecosystem names
# Values: Boolean Aij matrix with True if interaction is strong, else False
ptile_th_gt_dict = {
    gt_name: np.abs(Globals.gts[gt_name]) > ptile_th_dict[gt_name] for gt_name in Globals.gt_names
}


def check_interaction_type(tup):
    """For tuple "tup" of "coeff_type_abbreviation" values (i.e., "p", "z", or "n"), properly assign
    the sign-based interaction label (e.g., "+/+", "+/-", "+/0", etc.)"""

    coef1 = tup[0][-1]
    coef2 = tup[1][-1]

    if coef1 == "p" and coef2 == "p":
        interaction = "+/+"
    elif (coef1 == "n" and coef2 == "p") or (coef1 == "p" and coef2 == "n"):
        interaction = "+/-"
    elif coef1 == "n" and coef2 == "n":
        interaction = "-/-"
    elif (coef1 == "n" and coef2 == "z") or (coef1 == "z" and coef2 == "n"):
        interaction = "-/0"
    elif (coef1 == "z" and coef2 == "p") or (coef1 == "p" and coef2 == "z"):
        interaction = "+/0"
    elif coef1 == "z" and coef2 == "z":
        interaction = "0/0"
    else:
        interaction = "error"

    return interaction


def check_interaction_inference(tup):
    """For use in checking if both coefficients of an interaction pair have been properly inferred"""
    coef1_inf = tup[0]
    coef2_inf = tup[1]

    return coef1_inf and coef2_inf


def get_interaction_pair_idxs(n_species):
    """Since valid interaction pairs are in the form of ((i,j), (j, i)), find the Aij matrix
    indices of those pairs for a genetic n_species ecosystem"""

    # For a generic n_species interaction matrix,
    # get a list of pairs of coefficients (i.e., ones that are actually connected)
    top_triange_idx = np.triu_indices(n_species)

    arr1 = np.array(top_triange_idx)  # coefficients i, j (top triangle of array)
    arr2 = np.array((arr1[1], arr1[0]))  # and their corresponding j, i

    pairs = [(tuple(arr1[:, i]), tuple(arr2[:, i])) for i in range(arr1.shape[1])]
    return pairs


def determine_if_interaction_is_strong(iid):
    """For a given "iid" (interaction id), cross-check the ground truth to
    calculate if that interaction pair contains two "strong" coefficients."""

    # in an iid, there is the did and cid information
    # will use that to evaluate if strong or not

    cid1, cid2 = iid.split("__")

    # inf_abb, did, coef1_idx, coef1_type = cid1.split('_')
    # inf_abb, did, coef2_idx, coef2_type = cid2.split('_')

    inf_abb, did, coef1_idx, _ = cid1.split("_")
    _, _, coef2_idx, _ = cid2.split("_")

    coef1_idx = ast.literal_eval(coef1_idx)
    coef2_idx = ast.literal_eval(coef2_idx)

    cur_gt_name = Globals.meta.loc[0]["ground_truth"]
    cur_gt = Globals.gts[cur_gt_name]

    strong_interaction_bool = (
        ptile_th_gt_dict[cur_gt_name].iloc[coef1_idx]
        and ptile_th_gt_dict[cur_gt_name].iloc[coef2_idx]
    )

    return strong_interaction_bool


###################################
### CREATE INTERACTION DATAFRAME ##
###################################


def create_interaction_dataframe(chunked_df_dir=None, interaction_df_dir=None, save=False):
    """Returns nothing, but reads "aijs_df" from file and generates/saves "interaction_df":

    * One interaction_df exists per inference method
    * Each row is an interaction within mtist (3,296,790 interactions in total)

    +-----------+----------------------------------+------------+------+------------+----------+
    |           | iid                              | inf_or_not | did  | inf_method | int_type |
    +-----------+----------------------------------+------------+------+------------+----------+
    |     0     | d_0_(0, 0)_n__d_0_(0, 0)_n       | True       | 0    | d          | -/-      |
    +-----------+----------------------------------+------------+------+------------+----------+
    |     1     | d_0_(0, 1)_p__d_0_(1, 0)_n       | False      | 0    | d          | +/-      |
    +-----------+----------------------------------+------------+------+------------+----------+
    | ...       | ...                              | ...        | ...  | ...        | ...      |
    +-----------+----------------------------------+------------+------+------------+----------+
    | 3,296,790 | d_4409_(2, 2)_n__d_4409_(2, 2)_n | True       | 4409 | d          | -/-      |
    +-----------+----------------------------------+------------+------+------------+----------+

    * iid: interaction id, of form:
            {inference_method_abbreviation}_{did}_{coef1_index}_{coef1_coef_type}__{inference_method_abbreviation}_{did}_{coef2_index}_{coef2_coef_type}
    * inf_or_not: True if both coefs in interaction were properly inferred
    * did: dataset id
    * inf_method: which inference method it came from
    * int_type: what kind of interaction it was

    """

    if chunked_df_dir is None:
        chunked_df_dir = "chunked_aijs_df"
    # No dir creation here because it needs to be already initialized before

    if interaction_df_dir is None:
        interaction_df_dir = "interaction_df"
    else:
        if not os.path.isdir(interaction_df_dir):
            os.mkdir(interaction_df_dir)

    ############################################################################################

    unique_coef_pairs = [
        ("n", "n"),
        ("p", "n"),
        ("n", "p"),
        ("z", "z"),
        ("z", "p"),
        ("n", "z"),
        ("z", "n"),
        ("p", "z"),
        ("p", "p"),
    ]

    corrs_interactions = [
        "-/-",
        "+/-",
        "+/-",
        "0/0",
        "0/+",
        "0/-",
        "0/-",
        "0/+",
        "+/+",
    ]

    rename_dict = dict(zip(unique_coef_pairs, corrs_interactions))
    path_to_chunked = lambda v: os.path.join(chunked_df_dir, v)
    # path_to_saved_interactions = lambda v: os.path.join("interaction_dfs", v)

    # dictionary with all inferences loaded into memory
    inf = load_inferences()

    # for each n-species ecosystem, get a list of all interaction pairs by aij indices
    pairs = dict(zip([3, 10, 100], [get_interaction_pair_idxs(i) for i in [3, 10, 100]]))

    # for each regression method, load all dids and then calculate the number correct
    for name_short, name in zip(
        GlobalParams.abbreviations, GlobalParams.formatted_regression_names
    ):
        # name = "LinearRegression"
        # did = 892
        # name_short = "d"
        # th = 0

        dfs = []
        for did in Globals.meta.index:

            n_species = Globals.meta.loc[did, "n_species"]
            cur_gt_name = Globals.meta.loc[did, "ground_truth"]
            cur_gt = Globals.gts[cur_gt_name]

            cur_aijs_df = grant.load_pickle(path_to_chunked(f"{name}_aijs_df_did_{did}"))

            n_pairs = len(pairs[n_species])
            iid_arr = []
            inf_or_not_arr = []

            for i, pair in enumerate(pairs[n_species]):

                idx1 = cur_aijs_df.loc[cur_aijs_df["coeff"] == pair[0]].index[0]
                idx2 = cur_aijs_df.loc[cur_aijs_df["coeff"] == pair[1]].index[0]
                iid = cur_aijs_df.loc[idx1, "cid"] + "__" + cur_aijs_df.loc[idx2, "cid"]
                inf_or_not = (
                    cur_aijs_df.loc[idx1, "inf_or_not"] and cur_aijs_df.loc[idx2, "inf_or_not"]
                )

                iid_arr.append(iid)
                inf_or_not_arr.append(inf_or_not)

            inf_method_arr = np.full(n_pairs, name_short, dtype=str)
            did_arr = np.full(n_pairs, did, dtype=int)

            cur_df = pd.DataFrame(
                [iid_arr, inf_or_not_arr, did_arr, inf_method_arr],
                index=["iid", "inf_or_not", "did", "inf_method"],
            ).T

            dfs.append(cur_df)

        df = pd.concat(dfs)

        df["int_type"] = (
            df["iid"]
            .apply(lambda iid: tuple(each_coeff[-1] for each_coeff in iid.split("__")))
            .rename(rename_dict)
        )

        if save:
            grant.save_pickle(df, os.path.join(interaction_df_dir, f"{name}_int_df"))

        del dfs


####################################
# DETERMINE IF INTERACTION IS STRONG
# CREATE A NEW LIGHTWEIGHT DATAFRAME
####################################


def create_counted_interaction_dataframe(chunked_df_dir=None, interaction_df_dir=None, save=False):
    """Returns nothing, but creates "interaction_df_counted":

    * Needs `create_interaction_dataframe` to run first


    +--------+----------+------+----------------------+---------------+-------+-------+
    |        | int_type |  did |           inf_method | strong_or_not | count | total |
    +--------+----------+------+----------------------+---------------+-------+-------+
    |    0   | +/+      | 0    | LinearRegression     | False         | 9     | 40    |
    +--------+----------+------+----------------------+---------------+-------+-------+
    |    1   | +/+      | 0    | LinearRegression     | True          | 1     | 5     |
    +--------+----------+------+----------------------+---------------+-------+-------+
    | ...    | ...      | ...  | ...                  | ...           | ...   | ...   |
    +--------+----------+------+----------------------+---------------+-------+-------+
    | 211680 | 0/0      | 4409 | ElasticNetRegression | True          | 0     | 0     |
    +--------+----------+------+----------------------+---------------+-------+-------+

    * Rows are every interaction within ALL inference methods
    * strong_or_not: Whether BOTH coeffs are strong
    * count: number of coefficients that are correctly inferred
    * total: total umber of coeffs that *could* be correctly inferred
    """

    if chunked_df_dir is None:
        chunked_df_dir = "chunked_aijs_df"
        # No dir creation here because it needs to be already initialized before

    if interaction_df_dir is None:
        interaction_df_dir = "interaction_df"
        # No dir creation here because it needs to be already initialized before

    ####################################################################################

    unique_coef_pairs = [
        "('n', 'n')",
        "('p', 'n')",
        "('n', 'p')",
        "('z', 'z')",
        "('z', 'p')",
        "('n', 'z')",
        "('z', 'n')",
        "('p', 'z')",
        "('p', 'p')",
    ]

    corrs_interactions = [
        "-/-",
        "+/-",
        "+/-",
        "0/0",
        "0/+",
        "0/-",
        "0/-",
        "0/+",
        "+/+",
    ]

    rename_dict = dict(zip(unique_coef_pairs, corrs_interactions))

    ####################################################################################

    tmp = []
    for name, short_name in zip(
        GlobalParams.formatted_regression_names, GlobalParams.abbreviations
    ):
        df = grant.load_pickle(os.path.join(interaction_df_dir, f"{name}_int_df"))
        df = df.astype(
            {"iid": str, "inf_or_not": int, "did": int, "inf_method": str, "int_type": str}
        )

        strong_or_not = tuple(map(determine_if_interaction_is_strong, df["iid"]))
        df["strong_or_not"] = strong_or_not
        df["strong_or_not"] = df["strong_or_not"].astype(bool)

        df = df.pivot_table(
            index=["int_type", "did", "inf_method", "strong_or_not"],
            values="inf_or_not",
            aggfunc=[sum, "count"],
        )
        df.columns = df.columns.droplevel(1)
        df = df.rename(columns={"count": "total", "sum": "count"})
        df = df.reset_index()
        tmp.append(df)

    df = pd.concat(tmp)
    df["inf_method"] = df["inf_method"].replace(
        dict(zip(GlobalParams.abbreviations, GlobalParams.formatted_regression_names))
    )

    df["int_type"] = df["int_type"].replace(rename_dict)

    if save:
        grant.save_pickle(df, os.path.join(interaction_df_dir, "int_df_counted"))


###################################
###           PLOTTING           ##
###################################


def create_agg_dfs(interaction_df_dir=None):

    if interaction_df_dir is None:
        interaction_df_dir = "interaction_df"
        # No dir creation here because it needs to be already initialized before

    df = grant.load_pickle(os.path.join(interaction_df_dir, "int_df_counted"))

    df_all_pairs = df.pivot_table(
        index=["int_type", "did", "inf_method"],
        values=["count", "total"],
        aggfunc=sum,
    ).reset_index()

    # Pivot table on count - will collapse all of the dids in a single ground truth/inference method using aggfunc sum.
    agg_counts = (
        df_all_pairs.set_index("did")
        .join(Globals.meta)
        .pivot_table(
            index=["inf_method", "ground_truth"], columns="int_type", values="count", aggfunc=sum
        )
        # .drop(columns=["0/0", "-/0", "+/0"])
    )

    # Pivot table on totals - will collapse all of the dids in a single ground truth/inference method using aggfunc sum.
    agg_totals = (
        df_all_pairs.set_index("did")
        .join(Globals.meta)
        .pivot_table(
            index=["inf_method", "ground_truth"], columns="int_type", values="total", aggfunc=sum
        )
        # .drop(columns=["0/0", "-/0", "+/0"])
    )
    # Divide these two to get the "percentage" correct per ecosystem, inference methods

    return agg_counts, agg_totals


def plot_inf_by_interation(interaction_df_dir=None):

    if interaction_df_dir is None:
        interaction_df_dir = "interaction_df"
        # No dir creation here because it needs to be already initialized before

    agg_counts, agg_totals = create_agg_dfs(interaction_df_dir=interaction_df_dir)
    # agg_frac = agg_counts / agg_totals

    ##########################################
    to_drop = ["0/0"]

    for each in to_drop:
        if each in agg_counts.columns:
            agg_counts = agg_counts.drop(columns=each)

    for each in to_drop:
        if each in agg_totals.columns:
            agg_totals = agg_totals.drop(columns=each)
    ###################################################

    n_ecosystems = len(Globals.meta["ground_truth"].unique())
    n_interations = len(agg_counts.columns)

    # fig, ax = plt.subplots(figsize=(4, 8))
    fig, axes, cbar_ax = grant.easy_multi_heatmap(
        ncols=4,
        # sharey=True,
        base_figsize=((2.3 / 3) * n_interations, (0.3) * n_ecosystems),
    )  # 4 being n_inf

    for i, name in enumerate(GlobalParams.formatted_regression_names):

        ax = axes[i]
        sns.heatmap(
            (agg_counts / agg_totals).loc[name],
            cmap="plasma",
            center=0.5,
            annot=True,
            # square=True,
            ax=ax,
            # cbar=False,
            cbar_ax=cbar_ax,
            linewidth=1,
        )
        if i > 0:
            ax.set_ylabel(None)
            ax.set_yticks([])

        ax.set_title(name)

        # [ax.axhline(j, color="tab:gray", linewidth=5) for j in [7 * i for i in range(4)]];

    axes[-1].set_axis_off()


def plot_inf_differences(interaction_df_dir=None, default_inf=None):

    if interaction_df_dir is None:
        interaction_df_dir = "interaction_df"
        # No dir creation here because it needs to be already initialized before

    if default_inf is None:
        default_inf = "LinearRegression"

    ##########################################

    agg_counts, agg_totals = create_agg_dfs(interaction_df_dir=interaction_df_dir)

    ##########################################

    to_drop = ["0/0"]

    for each in to_drop:
        if each in agg_counts.columns:
            agg_counts = agg_counts.drop(columns=each)

    for each in to_drop:
        if each in agg_totals.columns:
            agg_totals = agg_totals.drop(columns=each)

    ###################################################

    n_ecosystems = len(Globals.meta["ground_truth"].unique())
    n_interations = len(agg_counts.columns)

    agg_frac = agg_counts / agg_totals
    default = agg_frac.loc[default_inf]

    fig, axes, cbar_ax = grant.easy_multi_heatmap(
        ncols=3,
        base_figsize=((2.3 / 3) * n_interations, (0.33) * n_ecosystems),
    )

    # for all non-LinearRegression methods
    for i, name in enumerate(GlobalParams.formatted_regression_names[1:]):

        cur_table = agg_frac.loc[name]
        plot_table = cur_table - default

        sns.heatmap(
            plot_table,
            cmap="coolwarm",
            center=0,
            vmax=0.1,
            vmin=-0.2,
            annot=True,
            # square=True,
            ax=axes[i],
            linewidth=1,
            cbar_ax=cbar_ax,
        )

        if i > 0:
            axes[i].get_yaxis().set_visible(False)
            # axes[i].set_xticklabels([])

        axes[i].set_title(name)

    fig.suptitle("RegularizedRegression - LinearRegression")
    plt.tight_layout()

    # grant.savefig(fig, os.path.join('final_figs', 'all_pairs_interaction_heatmap'))
