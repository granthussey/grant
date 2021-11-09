import os

import numpy as np
import pandas as pd

from grant import grant
import seaborn as sns
import matplotlib.pyplot as plt

from mtist import master_dataset_generation as mdg
from mtist import assemble_mtist as am
from mtist import mtist_utils as mu
from mtist import infer_mtist as im
from mtist import graphing_utils as gu
from mtist.graphing_utils import score_heatmap


class GLOBALS:
    """Some globals that I *might* want to change later"""

    regression_names = ["default_score", "ridge_score", "lasso_score", "elasticnet_score"]

    formatted_regression_names = [
        "LinearRegression",
        "RidgeRegression",
        "LassoRegression",
        "ElasticNetRegression",
    ]

    prefixes = ["default", "ridge_CV", "lasso_CV", "elasticnet_CV"]


class MultiAssembly:
    """A class to help merge multiple assemblies together, if needed
    This is mostly used with the previous version of MTIST where I hadn't expanded it yet. So
    I had to mix together multiple assemblies (hence, MultiAssembly) into one big heatmap."""

    local_dir = "/Users/granthussey/Lab/Schluter/MTIST_post_revision/MTIST2.0"
    inference_names = ["default", "ridge_CV", "lasso_CV", "elasticnet_CV"]
    prefixes = [f"{name}_" for name in inference_names]
    diff_assemblies = [f"datasets_{dataset_name}" for dataset_name in [3, 4, 5]] + ["mtist1.0"]
    n_datasets = 1134

    def load_score_data(self):
        """Returns dict {assembly_name: score_dict}. score_dict is dict {inf_method: score_dataframe}."""

        scores = {}
        for assembly in self.diff_assemblies:
            scores[assembly] = {}
            for prefix, name in zip(self.prefixes, self.inference_names):
                scores[assembly][name] = pd.read_csv(
                    os.path.join(
                        self.local_dir, f"{assembly}/{prefix}inference_result/{prefix}es_scores.csv"
                    )
                ).drop(columns="Unnamed: 0")

        return scores

    def generate_report_df(self, use_floored_scores=False):
        """Report df is generated.
        Columns: winners, max_scores, stdev, median_score, mean_score, default_score,
                ridge_score, lasso_score, elasticnet_score, assembly.

        For use in comparing ES scores between the assemblies."""

        scores = self.load_score_data()

        if use_floored_scores:
            floored_score_indexer = 1
        else:
            floored_score_indexer = 0

        ################################################################################################################

        assemblies = self.diff_assemblies

        collect = {}
        for assembly in assemblies:
            collect[assembly] = dict(
                winners=[],
                max_scores=[],
                stdev=[],
                median_score=[],
                mean_score=[],
                default_score=[],
                ridge_score=[],
                lasso_score=[],
                elasticnet_score=[],
            )

            # For each did
            for did in range(self.n_datasets):

                # Get a list of scores, only for unFLOORED SCORES
                cur_scores = [
                    scores[assembly][key].iloc[did, floored_score_indexer]
                    for key in scores[assembly].keys()
                ]

                collect[assembly]["max_scores"].append(max(cur_scores))
                collect[assembly]["winners"].append(
                    self.inference_names[cur_scores.index(max(cur_scores))]
                )
                collect[assembly]["stdev"].append(np.std(cur_scores))
                collect[assembly]["median_score"].append(np.median(cur_scores))
                collect[assembly]["mean_score"].append(np.mean(cur_scores))
                collect[assembly]["default_score"].append(cur_scores[0])
                collect[assembly]["ridge_score"].append(cur_scores[1])
                collect[assembly]["lasso_score"].append(cur_scores[2])
                collect[assembly]["elasticnet_score"].append(cur_scores[3])

        df = pd.DataFrame([])

        for assembly in assemblies:

            _df = pd.DataFrame(
                [
                    collect[assembly]["winners"],
                    collect[assembly]["max_scores"],
                    collect[assembly]["stdev"],
                    collect[assembly]["median_score"],
                    collect[assembly]["mean_score"],
                    collect[assembly]["default_score"],
                    collect[assembly]["ridge_score"],
                    collect[assembly]["lasso_score"],
                    collect[assembly]["elasticnet_score"],
                ],
                index=[
                    "winners",
                    "max_scores",
                    "stdev",
                    "median_score",
                    "mean_score",
                    "default_score",
                    "ridge_score",
                    "lasso_score",
                    "elasticnet_score",
                ],
                columns=pd.Index(range(self.n_datasets), name="did"),
            ).T

            _df = _df.astype(
                dict(zip(df.columns, [str, float, float, float, float, float, float, float, float]))
            )
            _df = _df.assign(assembly=assembly)

            df = pd.concat([df, _df])

        df = df.astype(
            dict(
                zip(df.columns, [str, float, float, float, float, float, float, float, float, str])
            )
        )

        return df

    def generate_combined_report_df(self, use_floored_scores=False):

        df = self.generate_report_df(use_floored_scores)

        full_3 = df.query("assembly == 'datasets_3'").join(
            pd.read_csv(os.path.join("datasets_3", "mtist_metadata.csv")).drop(columns="Unnamed: 0")
        )
        full_4 = df.query("assembly == 'datasets_4'").join(
            pd.read_csv(os.path.join("datasets_4", "mtist_metadata.csv")).drop(columns="Unnamed: 0")
        )
        full_5 = df.query("assembly == 'datasets_5'").join(
            pd.read_csv(os.path.join("datasets_5", "mtist_metadata.csv")).drop(columns="Unnamed: 0")
        )
        full_1 = df.query("assembly == 'mtist1.0'").join(
            pd.read_csv(os.path.join("mtist1.0", "mtist_metadata.csv")).drop(columns="Unnamed: 0")
        )

        full = pd.concat(
            [
                full_3.drop(columns="did"),
                full_4.drop(columns="did"),
                full_5.drop(columns="did"),
                full_1.drop(columns="did"),
            ]
        ).reset_index()

        # REmoving the duplicated results!

        rm_idx1 = full[(full["n_timeseries"] == 5) & (full["assembly"] == "datasets_3")].index
        rm_idx2 = full[(full["n_timepoints"] == 10) & (full["assembly"] == "mtist1.0")].index
        rm_idx3 = full[(full["n_timepoints"] == 10) & (full["assembly"] == "datasets_5")].index
        rm_idx4 = full[(full["n_timeseries"] == 5) & (full["assembly"] == "datasets_5")].index

        full = full.drop(rm_idx1.union(rm_idx2).union(rm_idx3).union(rm_idx4))

        return full

    def plot_combined_differences(full, save=False):

        diff_hm_names = ["ridge_score", "lasso_score", "elasticnet_score"]
        format_names = ["RidgeRegression", "LassoRegression", "ElasticNetRegression"]

        for name, fm_name in zip(diff_hm_names, format_names):

            scores_default = (
                full.sort_values(by=["ground_truth", "n_timeseries"])
                .query("seq_depth == 'high'")
                .pivot(
                    index=["noise", "sampling_scheme", "n_timepoints"],
                    columns=["ground_truth", "n_timeseries"],
                    values="default_score",
                )
            )

            scores_other = (
                full.sort_values(by=["ground_truth", "n_timeseries"])
                .query("seq_depth == 'high'")
                .pivot(
                    index=["noise", "sampling_scheme", "n_timepoints"],
                    columns=["ground_truth", "n_timeseries"],
                    values=name,
                )
            )

            to_plot = (scores_other - scores_default).applymap(lambda v: round(v, 2))

            fig, ax = plt.subplots(figsize=(22, 12))

            sns.heatmap(
                to_plot,
                center=0,
                cmap="coolwarm",
                ax=ax,
                annot=True,
                linewidths=1,
                vmax=0.4,
                vmin=-0.6,
            )

            draw_v_lines = [5 * i for i in range(7)]
            draw_h_lines = [5 * i for i in range(9)]

            [ax.axvline(i, c="tab:purple", linewidth=3) for i in draw_v_lines]
            [ax.axhline(i, c="tab:purple", linewidth=3) for i in draw_h_lines]

            fig.suptitle(f"Difference of Scores from: {fm_name} - LinearRegression")
            plt.tight_layout()

            if save:
                grant.savefig(fig, f"combined_difference_heatmap_{fm_name}")

    def plot_combined_es_scores(full, save=False):

        for name, fm_name in zip(GLOBALS.regression_names, GLOBALS.formatted_regression_names):

            score = (
                full.sort_values(by=["ground_truth", "n_timeseries"])
                .query("seq_depth == 'high'")
                .pivot(
                    index=["noise", "sampling_scheme", "n_timepoints"],
                    columns=["ground_truth", "n_timeseries"],
                    values=name,
                )
            ).applymap(lambda v: round(v, 2))

            fig, ax = plt.subplots(figsize=(22, 12))

            sns.heatmap(
                score,
                center=0.5,
                cmap="coolwarm",
                ax=ax,
                annot=True,
                linewidths=1,
            )

            draw_v_lines = [5 * i for i in range(7)]
            draw_h_lines = [5 * i for i in range(9)]

            [ax.axvline(i, c="tab:purple", linewidth=3) for i in draw_v_lines]
            [ax.axhline(i, c="tab:purple", linewidth=3) for i in draw_h_lines]

            fig.suptitle(f"Scores from: {fm_name}")
            plt.tight_layout()

            if save:
                grant.savefig(fig, f"combined_score_heatmap_{fm_name}")

    def plot_combined_winners(full, save=False):
        fig, ax = plt.subplots(figsize=(12, 12))

        sns.heatmap(
            (
                full.sort_values(by=["ground_truth", "n_timeseries"])
                .query("seq_depth == 'high'")
                .pivot(
                    index=["noise", "sampling_scheme", "n_timepoints"],
                    columns=["ground_truth", "n_timeseries"],
                    values="winners",
                )
                .replace(dict(zip(GLOBALS.regression_names, [i for i in range(4)])))
            ),
            cmap="tab10",
            ax=ax,
            annot=(
                full.sort_values(by=["ground_truth", "n_timeseries"])
                .query("seq_depth == 'high'")
                .pivot(
                    index=["noise", "sampling_scheme", "n_timepoints"],
                    columns=["ground_truth", "n_timeseries"],
                    values="winners",
                )
                .replace(
                    dict(
                        zip(
                            GLOBALS.regression_names,
                            [v[0].capitalize() for v in GLOBALS.regression_names],
                        )
                    )
                )
            ).values,
            fmt="",
            cbar=False,
        )

        draw_v_lines = [5 * i for i in range(7)]
        draw_h_lines = [5 * i for i in range(9)]

        [ax.axvline(i, c="tab:purple", linewidth=3) for i in draw_v_lines]
        [ax.axhline(i, c="tab:purple", linewidth=3) for i in draw_h_lines]

        # ax.text(10, 10, "adf", fontdict=dict(ha="center"))
        fig.suptitle(f"Winners")
        plt.tight_layout()

        if save:
            grant.savefig(fig, f"winners_full")

    def plot_combined_stdev(full, save=False):
        fig, ax = plt.subplots(figsize=(15, 12))

        sns.heatmap(
            (
                full.sort_values(by=["ground_truth", "n_timeseries"])
                .query("seq_depth == 'high'")
                .pivot(
                    index=["noise", "sampling_scheme", "n_timepoints"],
                    columns=["ground_truth", "n_timeseries"],
                    values="stdev",
                )
                .replace(
                    dict(
                        zip(
                            GLOBALS.regression_names,
                            [v[0].capitalize() for v in GLOBALS.regression_names],
                        )
                    )
                )
            ),
            cmap="coolwarm",
            ax=ax,
            annot=(
                full.sort_values(by=["ground_truth", "n_timeseries"])
                .query("seq_depth == 'high'")
                .pivot(
                    index=["noise", "sampling_scheme", "n_timepoints"],
                    columns=["ground_truth", "n_timeseries"],
                    values="winners",
                )
                .replace(
                    dict(
                        zip(
                            GLOBALS.regression_names,
                            [v[0].capitalize() for v in GLOBALS.regression_names],
                        )
                    )
                )
            ).values,
            fmt="",
            #     cbar=False,
        )

        draw_v_lines = [5 * i for i in range(7)]
        draw_h_lines = [5 * i for i in range(9)]

        [ax.axvline(i, c="tab:purple", linewidth=3) for i in draw_v_lines]
        [ax.axhline(i, c="tab:purple", linewidth=3) for i in draw_h_lines]

        # ax.text(10, 10, "adf", fontdict=dict(ha="center"))

        fig.suptitle(f"StDev full")
        plt.tight_layout()

        if save:
            grant.savefig(fig, f"stdev_full")

    def plot_combined_max_scores(full, save=False):

        fig, ax = plt.subplots(figsize=(18, 12))
        sns.heatmap(
            full.sort_values(by=["ground_truth", "n_timeseries"])
            .query("seq_depth == 'high'")
            .pivot(
                index=["noise", "sampling_scheme", "n_timepoints"],
                columns=["ground_truth", "n_timeseries"],
                values="max_scores",
            )
            .applymap(lambda v: round(v, 2)),
            center=0.5,
            cmap="coolwarm",
            ax=ax,
            annot=True,
            linewidths=1,
        )

        draw_v_lines = [5 * i for i in range(7)]
        draw_h_lines = [5 * i for i in range(9)]

        [ax.axvline(i, c="tab:purple", linewidth=3) for i in draw_v_lines]
        [ax.axhline(i, c="tab:purple", linewidth=3) for i in draw_h_lines]

        fig.suptitle(f"Max Scores")
        plt.tight_layout()

        if save:
            grant.savefig(fig, f"max_scores_full")


def plot_differences(scores, save=False):

    diff_hm_names = ["ridge_score", "lasso_score", "elasticnet_score"]
    format_names = ["RidgeRegression", "LassoRegression", "ElasticNetRegression"]

    for name, fm_name in zip(diff_hm_names, format_names):

        scores_default = (
            scores.sort_values(by=["ground_truth", "n_timeseries"])
            .query("seq_depth == 'high'")
            .pivot(
                index=["noise", "sampling_scheme", "n_timepoints"],
                columns=["ground_truth", "n_timeseries"],
                values="default_score",
            )
        )

        scores_other = (
            scores.sort_values(by=["ground_truth", "n_timeseries"])
            .query("seq_depth == 'high'")
            .pivot(
                index=["noise", "sampling_scheme", "n_timepoints"],
                columns=["ground_truth", "n_timeseries"],
                values=name,
            )
        )

        to_plot = (scores_other - scores_default).applymap(lambda v: round(v, 2))

        fig, ax = plt.subplots(figsize=(24, 12))

        sns.heatmap(
            to_plot, center=0, cmap="coolwarm", ax=ax, annot=True, linewidths=1, vmax=0.4, vmin=-0.6
        )

        draw_v_lines = [7 * i for i in range(7)]
        draw_h_lines = [5 * i for i in range(9)]

        [ax.axvline(i, c="tab:purple", linewidth=3) for i in draw_v_lines]
        [ax.axhline(i, c="tab:purple", linewidth=3) for i in draw_h_lines]

        fig.suptitle(f"Difference of Scores from: {fm_name} - LinearRegression")
        plt.tight_layout()

        if save:
            grant.savefig(fig, f"combined_difference_heatmap_{fm_name}")


def plot_es_scores(scores, save=False):

    for name, fm_name in zip(GLOBALS.regression_names, GLOBALS.formatted_regression_names):

        score = (
            scores.sort_values(by=["ground_truth", "n_timeseries"])
            .query("seq_depth == 'high'")
            .pivot(
                index=["noise", "sampling_scheme", "n_timepoints"],
                columns=["ground_truth", "n_timeseries"],
                values=name,
            )
        ).applymap(lambda v: round(v, 2))

        fig, ax = plt.subplots(figsize=(22, 12))

        sns.heatmap(
            score,
            center=0.5,
            cmap="coolwarm",
            ax=ax,
            annot=True,
            linewidths=1,
        )

        draw_v_lines = [7 * i for i in range(7)]
        draw_h_lines = [5 * i for i in range(9)]

        [ax.axvline(i, c="tab:purple", linewidth=3) for i in draw_v_lines]
        [ax.axhline(i, c="tab:purple", linewidth=3) for i in draw_h_lines]

        fig.suptitle(f"Scores from: {fm_name}")
        plt.tight_layout()

        if save:
            grant.savefig(fig, f"combined_score_heatmap_{fm_name}")


def plot_winners(scores, save=False):
    fig, ax = plt.subplots(figsize=(12, 12))

    sns.heatmap(
        (
            scores.sort_values(by=["ground_truth", "n_timeseries"])
            .query("seq_depth == 'high'")
            .pivot(
                index=["noise", "sampling_scheme", "n_timepoints"],
                columns=["ground_truth", "n_timeseries"],
                values="winners",
            )
            .replace(dict(zip(GLOBALS.prefixes, [i for i in range(4)])))
        ),
        cmap="tab10",
        ax=ax,
        annot=(
            scores.sort_values(by=["ground_truth", "n_timeseries"])
            .query("seq_depth == 'high'")
            .pivot(
                index=["noise", "sampling_scheme", "n_timepoints"],
                columns=["ground_truth", "n_timeseries"],
                values="winners",
            )
            .replace(
                dict(
                    zip(
                        GLOBALS.prefixes,
                        [v[0].capitalize() for v in GLOBALS.regression_names],
                    )
                )
            )
        ).values,
        fmt="",
        cbar=False,
    )

    draw_v_lines = [7 * i for i in range(7)]
    draw_h_lines = [5 * i for i in range(9)]

    [ax.axvline(i, c="tab:purple", linewidth=3) for i in draw_v_lines]
    [ax.axhline(i, c="tab:purple", linewidth=3) for i in draw_h_lines]

    # ax.text(10, 10, "adf", fontdict=dict(ha="center"))
    fig.suptitle(f"Winners")
    plt.tight_layout()

    if save:
        grant.savefig(fig, f"winners_full")


def plot_stdev(scores, save=False):
    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(
        (
            scores.sort_values(by=["ground_truth", "n_timeseries"])
            .query("seq_depth == 'high'")
            .pivot(
                index=["noise", "sampling_scheme", "n_timepoints"],
                columns=["ground_truth", "n_timeseries"],
                values="stdev",
            )
            .replace(
                dict(
                    zip(
                        GLOBALS.regression_names,
                        [v[0].capitalize() for v in GLOBALS.regression_names],
                    )
                )
            )
        ),
        cmap="coolwarm",
        ax=ax,
        annot=(
            scores.sort_values(by=["ground_truth", "n_timeseries"])
            .query("seq_depth == 'high'")
            .pivot(
                index=["noise", "sampling_scheme", "n_timepoints"],
                columns=["ground_truth", "n_timeseries"],
                values="winners",
            )
            .replace(
                dict(
                    zip(
                        GLOBALS.prefixes,
                        [v[0].capitalize() for v in GLOBALS.regression_names],
                    )
                )
            )
        ).values,
        fmt="",
        #     cbar=False,
    )

    draw_v_lines = [7 * i for i in range(7)]
    draw_h_lines = [5 * i for i in range(9)]

    [ax.axvline(i, c="tab:purple", linewidth=3) for i in draw_v_lines]
    [ax.axhline(i, c="tab:purple", linewidth=3) for i in draw_h_lines]

    # ax.text(10, 10, "adf", fontdict=dict(ha="center"))

    fig.suptitle(f"StDev scores")
    plt.tight_layout()

    if save:
        grant.savefig(fig, f"stdev_full")


def plot_max_scores(scores, save=False):

    fig, ax = plt.subplots(figsize=(22, 12))
    sns.heatmap(
        scores.sort_values(by=["ground_truth", "n_timeseries"])
        .query("seq_depth == 'high'")
        .pivot(
            index=["noise", "sampling_scheme", "n_timepoints"],
            columns=["ground_truth", "n_timeseries"],
            values="max_scores",
        )
        .applymap(lambda v: round(v, 2)),
        center=0.5,
        cmap="coolwarm",
        ax=ax,
        annot=True,
        linewidths=1,
    )

    draw_v_lines = [7 * i for i in range(7)]
    draw_h_lines = [5 * i for i in range(9)]

    [ax.axvline(i, c="tab:purple", linewidth=3) for i in draw_v_lines]
    [ax.axhline(i, c="tab:purple", linewidth=3) for i in draw_h_lines]

    fig.suptitle(f"Max Scores")
    plt.tight_layout()

    if save:
        grant.savefig(fig, f"max_scores_full")
