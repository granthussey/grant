from collections.abc import Iterable

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import pickle
import numpy as np
import time
from functools import wraps


def fill_tax_table(tax):
    """
    Helper function that fills nan's in a taxonomy table. Such gaps are filled 'from the left' with the next higher non-nan taxonomy level and the lowest level (e.g. OTU# or ASV#) appended.
    """

    # Assumes that the taxonomy is ordered! NEEd to fix this for the future UMAP code.
    taxlevels = list(tax.columns[1::])
    root_level = tax.columns[0]

    # Add the unknown for the root level.
    tax[root_level] = tax[root_level].fillna("unknown_%s" % root_level)

    # For all levels, add in the unknown if it is none.
    for i, level in enumerate(taxlevels):
        _missing_l = tax[level].isna()
        tax.loc[_missing_l, level] = [
            "unknown_%s_of_" % level + str(x) for x in tax.loc[_missing_l][taxlevels[i - 1]]
        ]

    for i, (ix, c) in enumerate(tax.iteritems()):
        tax.loc[:, ix] = tax[ix].astype(str) + "____" + str(tax.index[i])

    tax = tax.applymap(lambda v: v if "unknown" in v else v.split("____")[0])

    return tax


def new_fill_tax_table(tax):
    return fill_tax_table(tax)


# From hctmicrobiome


def calculate_relative_counts(counts, label="OTU"):
    """From a counts table, calculate OTU relative abundances and include those as new column"""
    try:
        relative_counts = counts.groupby("SampleID").apply(
            lambda g: pd.DataFrame(
                {
                    "RelativeCount": g["Count"] / g["Count"].sum(),
                    label: g[label],
                    "SampleID": g.name,
                }
            )
        )
    except KeyError:
        raise KeyError("Default label is OTU, set to ASV if needed using label parameter")
    else:
        c = counts.set_index(["SampleID", label]).join(
            relative_counts.set_index(["SampleID", label])
        )
        c = c.reset_index()
        return c


def custom_legend(
    n_entries,
    names,
    color_array=None,
    ax=None,
    line_weight=4,
    marker="o",
    linestyle="None",
    **kwargs
):
    """Creates a custom legend on current graphic"""

    try:
        names = list(names)
    except Exception as e:
        raise Exception(
            "\n\n".join(
                [
                    str(e),
                    "'names' must be a list, or be able to be cast as a list",
                ]
            )
        )

    legend = []

    if color_array is not None:
        try:
            color_array = list(color_array)
        except Exception as e:
            raise Exception(
                "\n\n".join(
                    [
                        str(e),
                        "'color_array' must be a list, or be able to be cast as a list",
                    ]
                )
            )

        for i in range(n_entries):
            legend.append(
                Line2D(
                    [0],
                    [0],
                    color=color_array[i],
                    lw=line_weight,
                    marker=marker,
                    linestyle=linestyle,
                )
            )

    elif color_array is None:
        for i in range(n_entries):
            legend.append(Line2D([0], [0], lw=line_weight), marker=marker, linestyle=linestyle)

    if ax is None:
        plt.legend(legend, names, **kwargs)
    else:
        ax.legend(legend, names, **kwargs)


def easy_subplots(ncols=1, nrows=1, base_figsize=None, **kwargs):

    if base_figsize is None:
        base_figsize = (8, 5)

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(base_figsize[0] * ncols, base_figsize[1] * nrows),
        **kwargs
    )

    # Lazy way of doing this
    try:
        axes = axes.reshape(-1)
    except:
        pass

    return fig, axes


def despine(fig=None, axes=None):
    if fig is not None:
        sns.despine(trim=True, offset=0.5, fig=fig)

    elif axes is not None:

        if not isinstance(axes, Iterable):  # to generalize to a single ax
            axes = [axes]

        for ax in axes:
            sns.despine(trim=True, offset=0.5, ax=ax)

    else:
        fig = plt.gcf()
        sns.despine(trim=True, offset=0.5, fig=fig)


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def savefig(fig, filename, ft=None):
    if ft is None:
        ft = "jpg"
    fig.savefig("{}.{}".format(filename, ft), dpi=300, bbox_inches="tight")


def sns_add_n_points(df, axes, remove_legend=False):

    if not isinstance(axes, Iterable):  # to generalize to a single ax
        axes = [axes]

    for ax in axes:

        if remove_legend:
            ax.get_legend().remove()

        new_labels = []

        cur_xlabel = ax.get_xlabel()  # this initializes the per-ax xlabel (the COL in df)

        for text_obj in ax.get_xticklabels():

            cur_xtick = text_obj.get_text()  # get method to get the actual text from Text obj

            cur_n = len(
                df[df[cur_xlabel] == cur_xtick]
            )  # use data from ax, cur_xtick to calc len() for that query'd column in df!

            new_xtick = cur_xtick + "\nn = {}\n".format(
                cur_n
            )  # generate new label with string methods

            new_labels.append(new_xtick)

        ax.set_xticklabels(new_labels)


def save_pickle(thing, fn):
    with open("{}.pickle".format(fn), "wb") as handle:
        pickle.dump(thing, handle, protocol=3)


def load_pickle(fn):
    with open("{}.pickle".format(fn), "rb") as handle:
        thing = pickle.load(handle)
    return thing


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def set_right_cbar(fig, axes):
    """Take a fig, axes pair (from grant.grant.easy_subplots() may I suggest)
    and make the RIGHTMOST column of figures into a cbar axis.

    For best results:
    * Set gridspec_kw with width_ratios where the last axis is very thin (1:10, perhaps)

    Args:
        fig (matplotlib figure)
        axes (ndarray): numpy array of axes
        ncols and nrows are ints
    """

    gs = axes[0, 0].get_gridspec()

    # For the ENTIRE last column
    for ax in axes[0:, -1]:
        # Remove all of those axes
        ax.remove()

    # Take up the space you removed with a single large axis
    cbar_ax = fig.add_subplot(gs[0:, -1])

    return fig, axes, cbar_ax


def easy_multi_heatmap(ncols=1, nrows=1, base_figsize=None, width_ratios=None, **kwargs):
    """returns (fig, axes, cbar_ax)"""

    if width_ratios is None:
        width_ratios = [1] * (ncols + 1)
        width_ratios[-1] = 0.1

        update_dict = {"gridspec_kw": {"width_ratios": width_ratios}}

        kwargs.update(update_dict)

    if base_figsize is None:
        base_figsize = (6, 6)

    ncols = ncols + 1  # to account for new cbar_ax

    fig, axes = easy_subplots(ncols, nrows, base_figsize=base_figsize, **kwargs)

    axes = axes.reshape((nrows, ncols))

    fig, axes, cbar_ax = set_right_cbar(fig, axes)

    if 1 in axes.shape:
        axes = axes.reshape(-1)

    return fig, axes, cbar_ax


def timefn(fn):
    """wrapper to time the enclosed function"""

    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn: {} took {} seconds".format(fn.__name__, t2 - t1))
        return result

    return measure_time
