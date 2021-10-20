import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

FAMILY = {
    "Aerococcaceae": "#5875DE",
    "Alicyclobacillaceae": "#5875DE",
    "Bacillaceae": "#5875DE",
    "Carnobacteriaceae": "#5875DE",
    "Lactobacillaceae": "#5875DE",
    "Leuconostocaceae": "#5875DE",
    "Paenibacillaceae": "#5875DE",
    "Planococcaceae": "#5875DE",
    "Christensenellaceae": "#BEA89A",
    "Clostridiaceae 1": "#BEA89A",
    "Clostridiaceae": "#BEA89A",
    "Clostridiales vadinBB60 group": "#BEA89A",
    "Defluviitaleaceae": "#BEA89A",
    "Eubacteriaceae": "#BEA89A",
    "Family XI": "#BEA89A",
    "Family XIII": "#BEA89A",
    "Lachnospiraceae": "#BEA89A",
    "Peptococcaceae": "#BEA89A",
    "Peptostreptococcaceae": "#BEA89A",
    "Ruminococcaceae": "#BEA89A",
    "Staphylococcaceae": "#F4EE26",
    "Streptococcaceae": "#AFCF3C",
    "Enterococcaceae": "#0D7E2B",
    "Erysipelotrichaceae": "#FBA22E",
}

PHYLUM = {
    "Acidobacteria": "#A3A3A3",
    "Actinobacteria": "#A3A3A3",
    "Bacteroidetes": "#16DDD3",
    "Firmicutes": "#BEA89A",
    "Proteobacteria": "#EE2C2C",
    "Verrucomicrobia": "#CA0BE8",
}

DF_FAMILY = pd.DataFrame(FAMILY.values(), index=FAMILY.keys(), columns=["HexColor"])
DF_PHYLUM = pd.DataFrame(PHYLUM.values(), index=PHYLUM.keys(), columns=["HexColor"])

DF_FAMILY.index.name = "Family"
DF_PHYLUM.index.name = "Phylum"


def match_colors(tax):
    """takes a tax table and matches colors
    
    First, we match on family.
    Next, we match leftover on phylum.
    If all fails, assign #D0D0D0.

    This is CASE-SENSITIVE ,,,,

    """

    # NAME OF INDEX
    if tax.index.name is None:
        index_col = "index"
    else:
        index_col = tax.index.name

    # MATCH ON FAMILY
    matched_family = tax.reset_index().set_index("Family").join(DF_FAMILY)
    not_processed = matched_family[matched_family["HexColor"].isnull()]
    matched_family = matched_family.dropna()

    # MATCH ON PHYLUM
    matched_phylum = (
        not_processed.reset_index()
        .set_index("Phylum")
        .drop(columns="HexColor")
        .join(DF_PHYLUM)
    )
    not_processed = matched_phylum[matched_phylum["HexColor"].isnull()]
    matched_phylum = matched_phylum.dropna()

    # ASSIGN GRAY TO ALL OTHERS
    not_processed["HexColor"] = "#D0D0D0"

    # CONCAT TOGETHER
    matched_family = matched_family.reset_index().set_index(index_col)
    matched_phylum = matched_phylum.reset_index().set_index(index_col)
    not_processed = not_processed.reset_index().set_index(index_col)

    matched = pd.concat((matched_family, matched_phylum, not_processed))

    return matched


def display_colors():
    _, ax = plt.subplots(figsize=(0.2, 0.2))
    ax.set_title("FAMILY")

    for key in FAMILY.keys():
        fig, ax = plt.subplots(figsize=(0.2, 0.2))
        plt.axis("off")
        plt.title(key)

        sns.palplot([FAMILY[key]])

    _, ax = plt.subplots(figsize=(0.2, 0.2))
    ax.set_title("PHYLUM")

    for key in PHYLUM.keys():
        fig, ax = plt.subplots(figsize=(0.2, 0.2))
        plt.axis("off")
        plt.title(key)

        sns.palplot([PHYLUM[key]])

    _, ax = plt.subplots(figsize=(0.2, 0.2))
    ax.set_title("ELSE")
    sns.palplot(["#D0D0D0"])


def display_result(matched_tax):
    """run on matched_tax, or the output of match_colors(), to see the assigned colors"""

    matched_tax["PF"] = matched_tax["Phylum"] + "_" + matched_tax["Family"]
    unique_combos = np.sort(matched_tax["PF"].unique())

    for pf in unique_combos:
        fig, ax = plt.subplots(figsize=(0.2, 0.2))
        plt.axis("off")
        plt.title(pf)

        sns.palplot(matched_tax.loc[matched_tax["PF"] == pf, "HexColor"])
