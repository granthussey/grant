import os

import numpy as np
import pandas as pd


def calc_midpoint(u, v):
    x1 = u[0]
    x2 = v[0]

    y1 = u[1]
    y2 = v[1]

    midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)

    return midpoint


def save_stable_aijs(root, aijs):
    """3 species only"""

    def make_dir(dir_name):
        if os.path.isdir(dir_name):
            print(f'"{dir_name}" is already made')
        else:
            os.mkdir(dir_name)

    make_dir(root)
    make_dir(os.path.join(root, "interaction_coefficients"))
    make_dir(os.path.join(root, "growth_rates"))

    for key in aijs.keys():

        # if not "3_sp_" in key:
        #     gr_file = f"3_sp_gr_{key}.csv"
        #     aij_file = f"3_sp_aij_{key}.csv"

        # else:
        #     gr_file = f"3_sp_gr_{key}.csv"
        #     aij_file = f"3_sp_aij_{key}.csv"

        gr_file = f"3_sp_gr_{key}.csv"
        aij_file = f"3_sp_aij_{key}.csv"

        aij = aijs[key].values
        gr = solve_for_stable_gr(aij)

        np.savetxt(os.path.join(root, "interaction_coefficients", aij_file), aij, delimiter=",")
        np.savetxt(os.path.join(root, "growth_rates", gr_file), gr, delimiter=",")


def solve_for_stable_gr(Aij):
    ri = -np.sum(Aij, axis=1)
    return ri


def judge_interactions(aij):
    n_species = aij.shape[0]

    pairs = get_interaction_pair_idxs_unique(n_species)

    int_types = []
    for coef1_idx, coef2_idx in pairs:

        interaction = None

        coef1_pos = aij[coef1_idx] > 0
        coef2_pos = aij[coef2_idx] > 0

        coef1_neg = aij[coef1_idx] < 0
        coef2_neg = aij[coef2_idx] < 0

        coef1_zero = aij[coef1_idx] == 0
        coef2_zero = aij[coef2_idx] == 0

        if coef1_pos and coef2_pos:
            interaction = "+/+"
        elif (coef1_pos and coef2_neg) or (coef1_neg and coef2_pos):
            interaction = "+/-"
        elif coef1_neg and coef2_neg:
            interaction = "-/-"
        elif (coef1_pos and coef2_zero) or (coef1_zero and coef2_pos):
            interaction = "+/0"
        elif (coef1_zero and coef2_neg) or (coef1_neg and coef2_zero):
            interaction = "-/0"
        elif coef1_zero and coef2_zero:
            interaction = "0/0"

        int_types.append(interaction)

    return int_types


def color_interactions(int_types):

    change_dict = {
        "+/+": "tab:red",
        "+/-": "tab:purple",
        "-/-": "tab:blue",
        "0/0": "tab:grey",
        "-/0": "tab:cyan",
        "+/0": "tab:pink",
    }

    colors = pd.Series(int_types).replace(change_dict).to_list()

    return colors


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


def get_coef_color(coef_type):

    if coef_type == "+":
        return "tab:red"
    elif coef_type == "-":
        return "tab:blue"
    elif coef_type == "0":
        return "tab:gray"


def get_interaction_pair_idxs_unique(n_species):
    """Since valid interaction pairs are in the form of ((i,j), (j, i)), find the Aij matrix
    indices of those pairs for a genetic n_species ecosystem"""

    # For a generic n_species interaction matrix,
    # get a list of pairs of coefficients (i.e., ones that are actually connected)
    top_triange_idx = np.triu_indices(n_species)

    arr1 = np.array(top_triange_idx)  # coefficients i, j (top triangle of array)
    arr2 = np.array((arr1[1], arr1[0]))  # and their corresponding j, i

    pairs = [(tuple(arr1[:, i]), tuple(arr2[:, i])) for i in range(arr1.shape[1])]

    container = []
    for coef1, coef2 in pairs:
        if coef1 != coef2 or coef1 != coef2[::-1]:
            container.append((coef1, coef2))

    return container


def judge_coefficients(aij):
    n_species = aij.shape[0]
    pairs = get_interaction_pair_idxs_unique(n_species)

    int_types = []
    for coef1_idx, coef2_idx in pairs:

        check_pos = lambda v: aij[v] > 0
        check_neg = lambda v: aij[v] < 0
        check_zero = lambda v: aij[v] == 0

        int_tuple = []
        for idx in [coef1_idx, coef2_idx]:

            if check_pos(idx):
                int_tuple.append("+")
            elif check_neg(idx):
                int_tuple.append("-")
            elif check_zero(idx):
                int_tuple.append("0")

        int_tuple = tuple(int_tuple)

        int_types.append(int_tuple)

    return int_types


def save_new_gt(new_gts, new_gt_folder_name):
    from distutils import dir_util

    from_dir = "ground_truths"
    to_dir = new_gt_folder_name
    dir_util.copy_tree(from_dir, to_dir)

    for key in new_gts.keys():
        name = str(key)
        name = key.replace("gt", "aij")
        np.savetxt(
            os.path.join(new_gt_folder_name, "interaction_coefficients", f"{name}.csv"),
            new_gts[key].values,
            delimiter=",",
        )


def calc_p(Aij):
    """From an Aij matrix, calculates P, Pm, Pc, Pplus, Pneg, Pe ecological interactions

    Returns:
        ns, numpy array, raw numbers of (p, pm, pc, pe, pplus, pminus)
        ps, numpy array, proportions of (p, pm, pc, pe, pplus, pminus)
    """

    # From an Aij matrix, calculates P, Pm, Pc, Pplus, Pneg, Pe.
    # Returns: n
    Aij = Aij.copy()
    n_sp = len(Aij)

    # Don't count self_interaction terms
    Aij[np.diag_indices_from(Aij)] = np.nan

    n_zero = 0
    n_pm = 0
    n_pc = 0
    n_pe = 0
    n_pplus = 0
    n_pminus = 0

    # Count all interactions
    for i in range(n_sp):
        for j in range(n_sp):

            for m in range(n_sp):
                for n in range(n_sp):

                    # You're at a pair to analyze
                    if i == n and j == m:

                        if Aij[i, j] > 0 and Aij[m, n] > 0:
                            n_pm = n_pm + 1

                        elif Aij[i, j] < 0 and Aij[m, n] < 0:
                            n_pc = n_pc + 1

                        elif (Aij[i, j] > 0 and Aij[m, n] < 0) or (Aij[i, j] < 0 and Aij[m, n] > 0):
                            n_pe = n_pe + 1

                        elif (Aij[i, j] > 0 and Aij[m, n] == 0) or (
                            Aij[i, j] == 0 and Aij[m, n] > 0
                        ):
                            n_pplus = n_pplus + 1

                        elif (Aij[i, j] < 0 and Aij[m, n] == 0) or (
                            Aij[i, j] == 0 and Aij[m, n] < 0
                        ):
                            n_pminus = n_pminus + 1

                        elif Aij[i, j] == 0 and Aij[m, n] == 0:
                            n_zero = n_zero + 1

    # Don't include the self-interaction terms in this (minus n_sp)
    # Also don't include the n_zero's (per supplement)
    n_interactions_no_zeros = (np.product(Aij.shape) - n_sp - n_zero) / 2
    n_interactions = np.product(Aij.shape) - n_sp

    # Divide by 2 because things get counted twice in the loop above
    ns = np.array((n_zero, n_pm, n_pc, n_pe, n_pplus, n_pminus)) / 2

    # Divide by n_interactions to calc proportion
    ps = np.array(ns) / n_interactions_no_zeros
    ps[0] = 1 - (n_zero / n_interactions)

    return ps


def calc_n(Aij):
    """From an Aij matrix, calculates P, Pm, Pc, Pplus, Pneg, Pe ecological interactions

    Returns:
        ns, numpy array, raw numbers of (p, pm, pc, pe, pplus, pminus)
        ps, numpy array, proportions of (p, pm, pc, pe, pplus, pminus)
    """

    # From an Aij matrix, calculates P, Pm, Pc, Pplus, Pneg, Pe.
    # Returns: n
    Aij = Aij.copy()
    n_sp = len(Aij)

    # Don't count self_interaction terms
    Aij[np.diag_indices_from(Aij)] = np.nan

    n_zero = 0
    n_pm = 0
    n_pc = 0
    n_pe = 0
    n_pplus = 0
    n_pminus = 0

    # Count all interactions
    for i in range(n_sp):
        for j in range(n_sp):

            for m in range(n_sp):
                for n in range(n_sp):

                    # You're at a pair to analyze
                    if i == n and j == m:

                        if Aij[i, j] > 0 and Aij[m, n] > 0:
                            n_pm = n_pm + 1

                        elif Aij[i, j] < 0 and Aij[m, n] < 0:
                            n_pc = n_pc + 1

                        elif (Aij[i, j] > 0 and Aij[m, n] < 0) or (Aij[i, j] < 0 and Aij[m, n] > 0):
                            n_pe = n_pe + 1

                        elif (Aij[i, j] > 0 and Aij[m, n] == 0) or (
                            Aij[i, j] == 0 and Aij[m, n] > 0
                        ):
                            n_pplus = n_pplus + 1

                        elif (Aij[i, j] < 0 and Aij[m, n] == 0) or (
                            Aij[i, j] == 0 and Aij[m, n] < 0
                        ):
                            n_pminus = n_pminus + 1

                        elif Aij[i, j] == 0 and Aij[m, n] == 0:
                            n_zero = n_zero + 1

    # Don't include the self-interaction terms in this (minus n_sp)
    # Also don't include the n_zero's (per supplement)
    # n_interactions_no_zeros = (np.product(Aij.shape) - n_sp - n_zero) / 2
    # n_interactions = np.product(Aij.shape) - n_sp

    # Divide by 2 because things get counted twice in the loop above
    ns = np.array((n_zero, n_pm, n_pc, n_pe, n_pplus, n_pminus)) / 2

    # Divide by n_interactions to calc proportion
    # ps = np.array(ns) / n_interactions_no_zeros
    # ps[0] = 1 - (n_zero / n_interactions)

    return ns


def calc_u(Aij):
    u = np.max(np.real(np.linalg.eigvals(Aij)))
    return u