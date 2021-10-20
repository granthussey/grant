import numpy as np
import pandas as pd
import seaborn as sns
from hctmicrobiomemskcc.dataloading.dataloading import load_microbiome_tables
from matplotlib import pyplot as plt


def match_on_hct(tax, HCTDATAPATH=None):

    if HCTDATAPATH is None:
        HCTDATAPATH = "/Users/granthussey/Lab/Schluter/Archive/data/"

    hctmicrobiometables = load_microbiome_tables(local_dir=HCTDATAPATH)
    hct_taxonomy = hctmicrobiometables[0].join(hctmicrobiometables[1])

    # Match colors on Genus
    tax = tax.join(
        hct_taxonomy.set_index(["Genus"])[["Class", "HexColor"]]
        .groupby("Genus")
        .agg("first"),
        on=["Genus"],
        rsuffix=["hct"],
        how="left",
    )

    # Match colors on Family
    tax.loc[tax.HexColor.isna(), "HexColor"] = (
        tax.loc[tax.HexColor.isna()]
        .join(
            hct_taxonomy.set_index(["Family"])[["Class", "HexColor"]]
            .groupby("Family")
            .agg("first"),
            on=["Family"],
            rsuffix=["hct"],
            how="left",
        )["HexColor['hct']"]
        .values
    )

    # Match colors on Order
    tax.loc[tax.HexColor.isna(), "HexColor"] = (
        tax.loc[tax.HexColor.isna()]
        .join(
            hct_taxonomy.set_index(["Order"])[["Class", "HexColor"]]
            .groupby("Order")
            .agg("first"),
            on=["Order"],
            rsuffix=["hct"],
            how="left",
        )["HexColor['hct']"]
        .values
    )

    # Match colors on Class
    tax.loc[tax.HexColor.isna(), "HexColor"] = (
        tax.loc[tax.HexColor.isna()]
        .join(
            hct_taxonomy.set_index(["Class"])[["HexColor"]]
            .groupby("Class")
            .agg("first"),
            on=["Class"],
            rsuffix=["hct"],
            how="left",
        )["HexColor['hct']"]
        .values
    )

    # Match colors on Phylum
    tax.loc[tax.HexColor.isna(), "HexColor"] = (
        tax.loc[tax.HexColor.isna()]
        .join(
            hct_taxonomy.set_index(["Phylum"])[["HexColor"]]
            .groupby("Phylum")
            .agg("first"),
            on=["Phylum"],
            rsuffix=["hct"],
            how="left",
        )["HexColor['hct']"]
        .values
    )

    # Assign ALL OTHERS as WHITE (#FFFFFF)
    tax.loc[tax["HexColor"].isna(), "HexColor"] = "#FFFFFF"
    tax = tax.drop(columns=[col for col in tax.columns if "hct" in col])

    return tax
