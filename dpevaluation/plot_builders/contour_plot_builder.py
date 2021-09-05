#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
import pandas as pd
import math


def build_levels(rmspe_values, round_up, n_values):
    # Get levels for the range of results in contour plot
    l = [item for sublist in rmspe_values for item in sublist]
    _max = roundup(np.amax(rmspe_values), round_up)
    _min = roundup(np.amin(rmspe_values), round_up)

    return np.arange(_min, _max, roundup(_max / n_values, round_up))

def roundup(x, round_up):
    # Round up to nearest round_up
    return int(math.ceil(x / round_up)) * round_up


def plot_contour(df: pd.DataFrame, x_dim, y_dim, metric, name, path=None):
    plt.figure()
    matplotlib.rcParams.update({
        'axes.titlesize': 20,
        'axes.titlepad': 10,
        'axes.labelsize': 18,
        'axes.labelpad': 1.8,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16})

    x_dim_name, x_dim_dim = x_dim['name'], x_dim['dim']
    y_dim_name, y_dim_dim = y_dim['name'], y_dim['dim']
    metric_name, metric_dim = metric['name'], metric['dim']

    fig, ax = plt.subplots()
    x_dim_data = sorted(df[x_dim_dim].unique())
    y_dim_data = sorted(df[y_dim_dim].unique())

    metric_data = []
    for y_val in y_dim_data:
        current_row = []
        for x_val in x_dim_data:
            value = df[(df[x_dim_dim] == x_val) & (
                df[y_dim_dim] == y_val)][metric_dim]

            if value.shape[0] == 0:
                value = None
            else:
                value = float(value)

            current_row.append(value)

        metric_data.append(current_row)

    # rmspe = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22] # ML utility
    # rmspe = [190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240] # Opacus time health
    # rmspe = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2.5, 5, 7.5, 10]
    
    cs = plt.contourf(
        x_dim_data,
        y_dim_data,
        metric_data,
        16, #Memory ML
        # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        extend='both',
        cmap="terrain",  # RdGy
        # locator=matplotlib.ticker.MaxNLocator(integer=True)
        # norm=matplotlib.colors.LogNorm(),
        # locator=matplotlib.ticker.LogLocator(base=1, subs=range(10), numticks=12)
        # locator=matplotlib.ticker.FixedLocator(rmspe, nbins=len(rmspe))
    )

    norm = matplotlib.colors.Normalize(
        vmin=cs.cvalues.min(),
        vmax=cs.cvalues.max())

    # plt.yticks([1000, 2000, 3000, 4000, 5000, 5010]) # Parkinson SQ tools

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
    sm.set_array([])
    fig.colorbar(cs, label=metric_name)
    ax.set_title(name)
    plt.xlabel(x_dim_name)
    plt.ylabel(y_dim_name)

    # x10^3 at the top of y-axis
    # formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1, 1))
    # ax.yaxis.set_major_formatter(formatter)
    # ax.get_yaxis().get_offset_text().set_visible(False)
    # ax.annotate('x'+r'$10^{3}$', xy=(-.13, .96), xycoords='axes fraction', fontsize=12, color='black')

    plt.savefig(
        '/dp-tools-evaluation/results/plots/' + name + '.png',
        dpi=200,
        facecolor='w',
        edgecolor='w',
        orientation='portrait',
        bbox_inches='tight')

    plt.close()
