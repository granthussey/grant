import seaborn as sns
from grant import grant
from grant.mtist.ground_truths.network_graphs import graph_network
from grant.mtist.ground_truths.simulation import run_and_graph


def plot_hm_and_graph(n_sp, gts, gt_names=None, debug=False, axes=None):

    if gt_names is None:
        gt_names = [f"{n_sp}_sp_gt_{i}" for i in range(1, 4)]

    n_gts_to_plot = len(gt_names)

    if axes is None:
        fig, axes = grant.easy_subplots(ncols=2, nrows=n_gts_to_plot, base_figsize=(3, 3))
        axes = axes.reshape(n_gts_to_plot, 2)

    for i, name in enumerate(gt_names):
        aij = gts[name].values

        ax_hm = axes[i, 0]
        ax_network = axes[i, 1]

        sns.heatmap(
            aij,
            ax=ax_hm,
            square=True,
            center=0,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            annot=True,
            linewidth=3,
        )

        graph_network(aij, ax=ax_network)

        ax_network.set_xlim([1.2 * x for x in ax_network.get_xlim()])
        ax_network.set_ylim([1.2 * y for y in ax_network.get_ylim()])

        ax_hm.set_title(name)

    if debug:
        return axes


def plot_hm_and_graph_and_run(n_sp, gts, gt_names=None):

    if gt_names is None:
        gt_names = [f"{n_sp}_sp_gt_{i}" for i in range(1, 4)]

    n_gts_to_plot = len(gt_names)

    fig, big_axes = grant.easy_subplots(nrows=n_gts_to_plot, ncols=3, base_figsize=(3, 3))
    big_axes = big_axes.reshape(n_gts_to_plot, 3)

    plot_hm_and_graph(n_sp, gts, gt_names=gt_names, axes=big_axes)

    for i, key in enumerate(gt_names):
        aij = gts[key]

        run_and_graph(aij.values, ax=big_axes[i, 2])

        big_axes[i, 2].set_xticks([])
        big_axes[i, 2].set_xticklabels([])

        big_axes[i, 2].set_yticks([])
        big_axes[i, 2].set_yticklabels([])

        sns.despine(ax=big_axes[i, 2])
