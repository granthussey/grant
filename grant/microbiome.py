import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from grant import grant


def check_matched_ASVs(
    matched_ASVs, rel_counts, axes=None, name=None, remove_zeros=False, log_scale=False, barplot_kw={}, hist_kw={}
):

    """Check distribution of ASVs in rel_counts and tax table

    Args:
        matched_ASVs ([type]): [description]
        rel_counts ([type]): [description]
        axes (iterable): Iterable of matplotlib ax objects. Must be length of 2.
        name ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    try:
        if len(axes) != 2:
            _, axes = grant.easy_subplots(ncols=2)
    except TypeError:
        _, axes = grant.easy_subplots(ncols=2)

    if name is None:
        name = ""

    # First plot #

    if remove_zeros:
        ASVs_agg_by_sample = (
            rel_counts[matched_ASVs].sum(axis=1)[rel_counts[matched_ASVs].sum(axis=1) != 0].sort_values(ascending=False)
        )
    else:
        ASVs_agg_by_sample = (
            rel_counts[matched_ASVs].sum(axis=1).sort_values(ascending=False)
        )

    if log_scale:
        sns.histplot(np.log10(ASVs_agg_by_sample), ax=axes[0], **hist_kw)
    else:
        sns.histplot(ASVs_agg_by_sample, ax=axes[0], **hist_kw)

    if name == "":
        axes[0].set_title("Prevalance\nn = {}".format(len(ASVs_agg_by_sample)))
    else:
        axes[0].set_title("{} prevalance\nn_Samples = {}".format(name, len(ASVs_agg_by_sample)))

    axes[0].set_xlabel("Summed rel_abundances per sample")
    
    if log_scale:
        axes[0].set_ylabel("Log Count")
    else:
        axes[0].set_ylabel("Count")

    # Second plot #

    ASVs_by_freq = (
        rel_counts[matched_ASVs]
        .applymap(lambda v: 1 if v > 0 else 0)
        .sum()
        .sort_values(ascending=False)
    )

    if log_scale:
        sns.barplot(
            x=ASVs_by_freq.index,
            y=np.log10(ASVs_by_freq.values),
            color="tab:blue",
            ax=axes[1],
            **barplot_kw
        )


    else:
        sns.barplot(
            x=ASVs_by_freq.index,
            y=ASVs_by_freq.values,
            color="tab:blue",
            ax=axes[1],
            **barplot_kw
        )

    if name == "":
        axes[1].set_title("Non-zero ASVs\nn = {}".format(len(ASVs_by_freq)))
    else:
        axes[1].set_title("{} Non-zero ASVs\nn_ASVs = {}".format(name, len(ASVs_by_freq)))

    axes[1].set_xlabel("ASV")

    if log_scale:
        axes[1].set_ylabel("Log Count")
    else:
        axes[1].set_ylabel("Count")

    grant.despine(axes=axes)

    _ = plt.xticks(rotation="vertical")

    return axes

