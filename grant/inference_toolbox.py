import numpy as np
import pandas as pd


def calc_dlogydt(x, times):
    """x: an n_timepoints long column vector, from one timeseries, of one species
    times: an n_timepoints long column vector of corresponding timepoints

    returns: (dlogydts, dlogys, dts, times, valid_idxs_list)
    """

    n_intervals = len(x) - 1

    dlogydts = []
    dts = []
    valid_idxs = set()

    for i in range(n_intervals):

        # Check if valid interval
        validity_check = x[i] > 0 and x[i + 1] > 0
        if validity_check:

            # Calculate dt, dlogydt
            dt = times[i + 1] - times[i]
            dts.append(dt)

            dlogydt = (np.log(x[i + 1]) - np.log(x[i])) / dt
            dlogydts.append(dlogydt)

            # Save valid indices
            valid_idxs.update([i, i + 1])

        else:
            pass

    # Return empty arrays if all else fails
    if len(dlogydts) <= 0:
        # Return empty arrays
        return (np.array([]), np.array([]), np.array([]), np.array([], dtype=bool))

    else:

        valid_idxs_list = list(valid_idxs)
        times = times[valid_idxs_list].copy()
        times = np.array(times)
        dlogydts = np.array(dlogydts)
        dts = np.array(dts)

        return (dlogydts, dts, times, valid_idxs_list)


def calc_gmeans(x, times):
    """
    x: an n_timepoints long column vector, from one timeseries, of one species
    times: an n_timepoints long column vector of corresponding timepoints

    returns: (gmeans, times, valid_idxs_list)
    """

    n_intervals = len(x) - 1
    gmeans = []
    valid_idxs = set()

    for i in range(n_intervals):

        # Check if valid interval
        validity_check = x[i] > 0 and x[i + 1] > 0
        if validity_check:

            gmean = np.sqrt(x[i] * x[i + 1])
            gmeans.append(gmean)

            # Save valid indices
            valid_idxs.update([i, i + 1])

        else:
            pass

    # If no valid idxs found
    if len(gmeans) <= 0:

        # Return empty arrays
        return (np.array([]), np.array([]), np.array([], dtype=bool))

    else:

        valid_idxs_list = list(valid_idxs)
        times = np.array(times[valid_idxs_list].copy())  # only copy those with valid idx pairs
        gmeans = np.array(gmeans)

        return (gmeans, times, valid_idxs_list)

    pass
