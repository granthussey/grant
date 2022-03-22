import glob
import os
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import mdsine2 as md2
from mdsine2.names import STRNAMES


def format_dataset(path):
    """Format a certain MTIST dataset into form that mdsine2 can run inference on

    Args:
        path (Path-like): Path to MTIST dataset (csv file)

    Returns:
        tuple of pd.DataFrames in form (df, metadata, qpcr, reads, taxonomy)
    """

    df = pd.read_csv(path).drop(columns="Unnamed: 0")

    #### Calculate some stuff ####

    species_cols = df.columns[df.columns.str.contains("species_")]
    n_species = df["n_species"].unique()[0]
    OTU_labels = [
        "OTU_" + str(n_species) + "sp_" + "sp" + col.split("_")[-1] for col in species_cols
    ]

    #### Rename and prep for slicing ####

    df = df.assign(
        sampleID=lambda v: v["did"].astype(str)
        + "_"
        + v["timeseries_id"].astype(str)
        + "_"
        + v.index.astype(str)
    )
    df["measurement1"] = df[species_cols].sum(axis=1)
    df = df.rename(columns={"timeseries_id": "subject"})
    # df = df.rename(columns=dict(zip(species_cols, ['OTU_' + str(n_species) + 'sp_' + 'sp' + col.split('_')[-1] for col in species_cols]))).rename(columns={'timeseries_id':'subject'})

    # go from absolute -> rel_counts -> counts
    tmp = df.set_index("sampleID")
    rel_abundances = (tmp[species_cols].T / tmp["measurement1"]).T
    counts = (rel_abundances * 100000).applymap(np.floor).astype(int)
    to_concat = counts.rename(columns=dict(zip(species_cols, OTU_labels)))

    # put it all together
    df = df.set_index("sampleID").join(to_concat).reset_index()

    return df


def extract_mdsine2_dataframes(df):
    """From df, extract taxonomy, reads, qpcr, metadata dataframes"""

    metadata = df[["sampleID", "subject", "time"]]

    for i_subject in metadata["subject"].unique():
        metadata.loc[metadata["subject"] == i_subject, "time"] = (
            metadata.query("subject == @i_subject")["time"]
            - metadata.query("subject == @i_subject")["time"].min()
        )

    # metadata

    qpcr = df[["sampleID", "measurement1"]]
    qpcr["measurement2"] = qpcr["measurement1"] * 1.02
    qpcr["measurement3"] = qpcr["measurement1"] * 0.98
    # qpcr["measurement2"] = qpcr["measurement1"] * 1
    # qpcr["measurement3"] = qpcr["measurement1"] * 1
    qpcr = (qpcr.set_index("sampleID") * 1e10).reset_index()
    # qpcr

    # pertubations = pd.DataFrame(columns=["name", "start", "end", "subject"])
    # pertubations

    reads = df[list(df.columns[df.columns.str.contains("OTU_")].values) + ["sampleID"]].set_index(
        "sampleID"
    )  # grab only OTU and sampleID
    reads = reads.T.reset_index()
    reads = reads.rename(columns={"index": "name"})
    # reads.index.name = None
    # reads

    taxonomy = reads[["name"]]
    taxonomy["sequence"] = "ATGC"
    taxonomy[["kingdom"]] = "Bacteria"

    tmp = pd.concat([reads[["name"]]] * 6, axis=1)
    tmp.columns = ["phylum", "class", "order", "family", "genus", "species"]
    tmp = tmp.apply(lambda v: v.name[0] + "__" + v, axis=0)

    taxonomy = taxonomy.join(tmp)
    # taxonomy

    return (reads, taxonomy, qpcr, metadata)


# def format_dataset_and_save(path):
#     """Format a certain MTIST dataset into form that mdsine2 can run inference on

#     Args:
#         path (Path-like): Path to MTIST dataset (csv file)

#     Returns:
#         Nothing. It saves to disk the dataframes in
#     """


def infer_mdsine2(data_dir, output_dir, viz=False, n_samples=None, n_species=None, clustering=None):
    """data_dir is path to the data to use to infer, output_dir is the location to put output pickles"""

    if n_samples is None:
        n_samples = 200

    # if n_species is None:
    #     n_species = "10"
    # elif cluster = 'no-clusters'
    #     n_species=None
    # else:
    #     n_species = str(n_species)

    # if clustering is None:
    #     clustering = "fixed-clustering"

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    reg = md2.dataset.parse(
        name="test",
        taxonomy=data_dir / "taxonomy.tsv",
        reads=data_dir / "reads.tsv",
        qpcr=data_dir / "qpcr.tsv",
        perturbations=None,
        metadata=data_dir / "metadata.tsv",
    )

    params = md2.config.NegBinConfig(
        seed=0, burnin=100, n_samples=200, checkpoint=100, basepath=str(output_dir / "negbin")
    )

    print("Setting up negbin regression")
    mcmc_negbin = md2.negbin.build_graph(params=params, graph_name=reg.name, subjset=reg)

    print("Running negbin regression")
    mcmc_negbin = md2.negbin.run_graph(mcmc_negbin, crash_if_error=True)

    # Get a0 and a1 from negbin (get the mean of the posterior) and fixes them for inference
    from mdsine2.names import STRNAMES

    a0 = md2.summary(mcmc_negbin.graph[STRNAMES.NEGBIN_A0])["mean"]
    a1 = md2.summary(mcmc_negbin.graph[STRNAMES.NEGBIN_A1])["mean"]

    basepath = output_dir / "mdsine2" / "uc0"
    basepath.mkdir(exist_ok=True, parents=True)

    # Initialize parameters of the model (Seed = 0) burnin=50, total steps=100
    params = md2.config.MDSINE2ModelConfig(
        basepath=str(basepath),
        seed=0,
        burnin=50,
        n_samples=n_samples,
        negbin_a0=a0,
        negbin_a1=a1,
        checkpoint=50,
    )

    params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]["value_option"] = "no-clusters"
    # params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]["value_option"] = "fixed-clustering"
    # params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]["n_clusters"] = 10

    # initilize the graph
    print("Configure mdsine2 inference")
    mcmc_uc0 = md2.initialize_graph(params=params, graph_name=reg.name, subjset=reg)

    print("Start mdsine2 inference")
    mcmc_uc0 = md2.run_graph(mcmc_uc0, crash_if_error=True)

    if viz:

        pprint(md2.summary(mcmc_negbin.graph[STRNAMES.NEGBIN_A0]))
        print()
        pprint(md2.summary(mcmc_negbin.graph[STRNAMES.NEGBIN_A1]))

        fig = md2.negbin.visualize_learned_negative_binomial_model(mcmc_negbin)
        fig.tight_layout()
        plt.show()

        processvar = mcmc_uc0.graph[STRNAMES.PROCESSVAR]
        pv_rates_trace = processvar.get_trace_from_disk(section="entire")

        md2.visualization.render_trace(pv_rates_trace, n_burnin=50, **{"title": "process variance"})
        plt.show()

        clustering = mcmc_uc0.graph[STRNAMES.CLUSTERING_OBJ]
        md2.generate_cluster_assignments_posthoc(clustering, set_as_value=True)
        taxa = mcmc_uc0.graph.data.taxa

        # Visualize co-cluster posterior probability
        coclusters = md2.summary(mcmc_uc0.graph[STRNAMES.CLUSTERING_OBJ].coclusters)["mean"]
        md2.visualization.render_cocluster_probabilities(
            coclusters,
            taxa=reg.taxa,
            yticklabels="%(paperformat)s | %(index)s",
        )

        # Visualize trace for number of modules
        md2.visualization.render_trace(clustering.n_clusters)
        plt.show()

    return (mcmc_negbin, mcmc_uc0, reg)
