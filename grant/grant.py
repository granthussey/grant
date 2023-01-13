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


def old_fill_tax_table(tax):
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


def old_new_fill_tax_table(tax):
    return fill_tax_table(tax)


def fill_tax_table(tax):
    """Fills missing values in the taxonomy table. Will recognize only 'np.nan' data types as empty values.

    Args:
        tax (pd.DataFrame): Dataframe with index of ASV/OTU and columns of left -> right increasing specificity in taxonomy (e.g., Kingdom -> Species)

    Output:
        new_tax (pd.DataFrame): Properly-filled taxonomy dataframe
    """
    if len(tax.index) != len(tax.index.unique()):
        print(
            "Repeated OTUs/ASVs in the taxonomy index. Check to make sure there is only _one_ entry per OTU in taxonomy table."
        )

    # MUST be in increasing specificity order (Kingdom -> Species)
    # OTU/ASV must be the INDEX.
    tax_labels = tax.columns
    table_name = tax.index.name  # Important - don't remove this and its corresponding stpe below.

    # Gather all OTUs to iterate over
    otus = tax.index.unique()

    new_tax = []  # Collector for new taxonomy pd.Series
    for otu in otus:

        series = tax.loc[otu]

        # If there are no NaNs in the OTU, don't do anything.
        if (~series.isna()).all():
            new_tax.append(series)

        # However, if NaNs do exist, fill the taxonomy "from-the-left"
        else:
            first_nan = np.argwhere(series.isna().values == True)[0][0]

            # In case "Kingdom" is NaN (or other highest level taxa)
            if first_nan == 0:
                last_not_nan = first_nan
            else:
                last_not_nan = first_nan - 1

            ##### Below commented-out code I'm saving here, ignore #####
            # for i in range(first_nan, len(series)):
            #     series.iloc[i] = f'unk_{series.index[i]}_of_{series.index[i-1]}_{series.iloc[i-1]}'
            #####                                                  #####

            # Perform "fill-from-the-left"
            # For each and every NaN, fill it with the last non-NaN taxonomy, and append the ASV/OTU name at the end as well.
            for i in range(first_nan, len(series)):

                # In case "Kingdom" is NaN (or other highest level taxa)
                if i == 0:
                    series.iloc[i] = f"unk_{series.index[i]}"
                else:
                    series.iloc[
                        i
                    ] = f"unk_{series.index[i]}_of_{series.index[last_not_nan]}_{series.iloc[last_not_nan]}"

            # Add in the ASV/OTU name to the end of every unknown

            for i in range(first_nan, len(series)):
                series.iloc[i] = f"{series.iloc[i]}__{otu}"

            new_tax.append(series)

    new_tax = pd.concat(new_tax, axis=1).T

    # This name gets erased in the above transformation, so return it.
    new_tax.index.name = table_name

    return new_tax


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
    **kwargs,
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
        **kwargs,
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


def _save(fxn, outdir, filename, **kwargs):
    # TODO: I do not see the need for this failure catching routine. It makes the code less easy to follow. Consider removing
    """Routine to create directories that do not exist

    Args:
        fxn (function handle): f(Path-like object or str)
        outdir ([type]): [description]
        filename ([type]): [description]

    Raises:
        TypeError: [description]
    """

    if not callable(fxn):
        raise TypeError("'fxn' passed is not callable")

    if outdir is not None:
        try:
            outdir = Path(outdir).resolve(strict=True)
        except (FileNotFoundError, TypeError) as e:
            logger_taxumap.warning(
                '\nNo valid outdir was declared.\nSaving data into "./results" folder.\n'
            )

    elif outdir is None:
        outdir = Path("./results").resolve()
        try:
            os.mkdir(outdir)
            logger_taxumap.info("Making ./results folder...")
        except FileExistsError:
            logger_taxumap.info("./results folder already exists")
        except Exception as e:
            throw_unknown_save_error(e)
            sys.exit(2)

    try:
        fxn(os.path.join(outdir, filename), **kwargs)
    except Exception as e:
        throw_unknown_save_error(e)
    else:
        logger_taxumap.info("Save successful")


def calculate_inverse_simpson(X):
    ivs = X.apply(lambda r: 1 / np.sum(r**2), axis=1)
    return ivs


def draw_biplot_arrows(pca, pca_embedding, features, ax=None, th=False):

    if ax is None:
        fig, ax = plt.subplots()

    xvector = pca.components_[0]
    yvector = pca.components_[1]

    xs = pca_embedding[:, 0]
    ys = pca_embedding[:, 1]

    if th is True:
        arrow_lengths = np.sqrt(np.square(pca.components_[0, :]), np.square(pca.components_[1, :]))

        idx_to_draw = np.arange(len(arrow_lengths))[
            arrow_lengths > np.percentile(arrow_lengths, 99)
        ]

    elif th is False:
        idx_to_draw = np.arange(len(xvector))

    elif (isinstance(th, int) or isinstance(th, float)) and (th < 100):
        arrow_lengths = np.sqrt(np.square(pca.components_[0, :]), np.square(pca.components_[1, :]))

        idx_to_draw = np.arange(len(arrow_lengths))[
            arrow_lengths > np.percentile(arrow_lengths, th)
        ]

    else:
        idx_to_draw = np.arange(len(xvector))

    for i in idx_to_draw:

        plt.arrow(
            0,
            0,
            xvector[i] * max(xs),
            yvector[i] * max(ys),
            color="r",
            width=0.0005,
            head_width=0.005,
        )
        plt.text(
            xvector[i] * max(xs) * 1.2,
            yvector[i] * max(ys) * 1.2,
            list(features)[i],
            color="r",
        )

    result = {}
    for i in idx_to_draw:
        result[features[i]] = {
            "vector": np.array((xvector[i], yvector[i])),
            "arrow_length": np.sqrt(xvector[i] ** 2 + yvector[i] ** 2),
        }

    return result
