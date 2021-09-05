#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

def plot_violin(df: pd.DataFrame, x_dim, y_dim, name, path):
    plt.figure()
    matplotlib.rcParams.update({
        'axes.titlesize': 20,
        'axes.titlepad': 10,
        'axes.labelsize': 18,
        'axes.labelpad': 1.8,
        'xtick.labelsize': 10,
        'ytick.labelsize': 14})

    _, ax = plt.subplots()

    x_dim_name, x_dim_dim = x_dim['name'], x_dim['dim']
    y_dim_name, y_dim_dim = y_dim['name'], y_dim['dim']


    data_as_dict = dict(df.groupby(x_dim_dim)[y_dim_dim].apply(list))
    data = data_as_dict.values()


    plt.violinplot(
        data,
        showmeans=True,
    )

    labels = list(data_as_dict.keys())
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    ax.set_title(name)


    plt.xlabel(x_dim_name)
    plt.ylabel(y_dim_name)

    plt.savefig(
        path + name + '.png',
        dpi=200,
        facecolor='w',
        edgecolor='w',
        orientation='portrait',
        bbox_inches='tight')

    plt.close()
