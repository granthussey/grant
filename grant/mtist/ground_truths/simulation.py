import numpy as np
from grant import grant
from grant.mtist.ground_truths.utils import calc_n, calc_p, solve_for_stable_gr
from matplotlib import pyplot as plt
from mtist import mtist_utils as mu


def run_aij(Aij, random_seed=False, tend=None, sample_freq=None, noise=None):
    """Run an Aij matrix, get (t, y) numerical solution from Ecosims

    Default args for sim: sample_freq=100, tend=30, noise=0.01

    If random_seed is None, will pull unpredictable entropy from OS.
    If random_seed is False, sets random_seed to 8927.
    Otherwise, set to a desired seed. Used to seed initial abundances in simulation.

    """

    if random_seed is False:
        random_seed = 8927

    if tend is None:
        tend = 30

    if sample_freq is None:
        sample_freq = 100

    if noise is None:
        noise = 0.01

    dt = 0.1

    n_species = len(Aij)
    self_int = np.diag(Aij)[0]  # assume a single "self_int" self_interaction term (not always true)

    gr = solve_for_stable_gr(Aij)

    t, y = mu.simulate(
        aij=Aij, gr=gr, seed=random_seed, noise=noise, tend=tend, dt=dt, sample_freq=sample_freq
    )

    return t, y


def graph_simulation(t, y, Aij, ax=None, inner_text=False, remove_text=False):
    """Graphs a numerical solution (t, y) from Ecosims

    inner_text controls where the calc_p textbox goes - if True, goes bottom right of figure.
    remove_text controls existance of calc_p textbox - if True, removes textbox altogether

    """

    n_species = y.shape[1]
    self_int = np.diag(Aij)[0]  # assume a single "self_int" self_interaction term (not always true)

    if ax is None:
        fig, ax = plt.subplots()
        despine = True
    else:
        despine = False

    for i in range(n_species):
        ax.plot(t, y[:, i])

    ax.set_xlabel("time [d]")
    ax.set_ylabel("species abundances")

    p, pm, pc, pe, pplus, pminus = calc_p(Aij)
    ns = calc_n(Aij)

    u = np.max(np.real(np.linalg.eigvals(Aij)))

    # fmt: off
    if remove_text == False:
        if inner_text:

            ax.text(0.4, 0.3, "$p$" + "={}".format(round(p, 2)), transform=ax.transAxes, fontsize=12)
            ax.text(0.4, 0.2, "$p_{m}$" + "={}".format(round(pm, 2)), transform=ax.transAxes, fontsize=12)
            ax.text(0.4, 0.1, "$p_{c}$" + "={}".format(round(pc, 2)), transform=ax.transAxes, fontsize=12)

            ax.text(0.6, 0.3, "$p_{e}$" + "={}".format(round(pe, 2)), transform=ax.transAxes, fontsize=12)
            ax.text(0.6, 0.2, "$p_{+}$" + "={}".format(round(pplus, 2)), transform=ax.transAxes, fontsize=12)
            ax.text(0.6, 0.1, "$p_{-}$" + "={}".format(round(pminus, 2)), transform=ax.transAxes, fontsize=12)

            ax.text(0.8, 0.3, "$a_{ii}$" + "={}".format(round(self_int, 2)), transform=ax.transAxes, fontsize=12)
            ax.text(0.8, 0.2, "$U$" + "={}".format(round(u, 2)), transform=ax.transAxes, fontsize=12)

        elif inner_text == False:

            ax.text(1.05, 0.9, "$p$" + "={}".format(round(p, 2)), transform=ax.transAxes, fontsize=12)
            ax.text(1.05, 0.8, "$p_{m}$" + "={}".format(round(pm, 2)), transform=ax.transAxes, fontsize=12)
            ax.text(1.05, 0.7, "$p_{c}$" + "={}".format(round(pc, 2)), transform=ax.transAxes, fontsize=12)
            ax.text(1.05, 0.6, "$p_{e}$" + "={}".format(round(pe, 2)), transform=ax.transAxes, fontsize=12)

            ax.text(1.05, 0.5, "$p_{+}$" + "={}".format(round(pplus, 2)), transform=ax.transAxes, fontsize=12)
            ax.text(1.05, 0.4, "$p_{-}$" + "={}".format(round(pminus, 2)), transform=ax.transAxes, fontsize=12)
            ax.text(1.05, 0.3, "$a_{ii}$" + "={}".format(round(self_int, 2)), transform=ax.transAxes, fontsize=12)
            ax.text(1.05, 0.2, "$U$" + "={}".format(round(u, 2)), transform=ax.transAxes, fontsize=12)

    if despine:
        grant.despine()
    # fmt: on


def run_and_graph(
    Aij,
    random_seed=None,
    tend=None,
    sample_freq=None,
    noise=None,
    ax=None,
    inner_text=False,
    remove_text=False,
):
    """

    If random_seed is None, will pull unpredictable entropy from OS.
    If random_seed is False, sets random_seed to 8927.
    Otherwise, set to a desired seed. Used to seed initial abundances in simulation.

    inner_text controls where the calc_p textbox goes - if True, goes bottom right of figure.
    remove_text controls existance of calc_p textbox - if True, removes textbox altogether

    """

    t, y = run_aij(Aij, random_seed=random_seed, tend=tend, sample_freq=sample_freq, noise=noise)
    graph_simulation(t, y, Aij, ax=ax, inner_text=inner_text, remove_text=remove_text)
