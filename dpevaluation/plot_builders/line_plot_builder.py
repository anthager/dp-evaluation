#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from dpevaluation.utils.log import log


def plot_line(df: pd.DataFrame, x_dim, y_dim, line_dim, name, x_ticks, y_ticks):
    plt.figure()
    matplotlib.rcParams.update({
        'axes.titlesize': 20,
        'axes.titlepad': 10,
        'axes.labelsize': 18,
        'axes.labelpad': 1.8,
        'xtick.labelsize': 10,
        'ytick.labelsize': 14})

    _, ax = plt.subplots()

    x_dim_name, x_dim_dim = x_dim['name'], x_dim['dim'] # epsilon
    y_dim_name, y_dim_dim = y_dim['name'], y_dim['dim'] # result
    _, line_dim_dim = line_dim['name'], line_dim['dim']

    # colors = ["orange", "green", "purple", "xkcd:azure"]

    colors = {"gretel": "orange", "smartnoise_dpctgan": "green", "smartnoise_mwem": "purple", "smartnoise_patectgan": "xkcd:azure"}
    synth_names = {'gretel': 'Gretel', 'smartnoise_dpctgan': 'Smartnoise DPCTGAN', 'smartnoise_patectgan': 'Smartnoise PATECTGAN', 'smartnoise_mwem': 'Smartnoise MWEM'}

    last_values = []

    for i, (line_value, line_data) in enumerate(df.groupby([line_dim_dim])):
    # for i, (line_value, line_data) in enumerate(df.groupby(["query"])): # SQ utility
        # if line_value.lower() != "diffprivlib":
        aggregated_line_data = line_data \
            .groupby([x_dim_dim]) \
            .aggregate({y_dim_dim: 'mean'}) \
            .reset_index() \
            .sort_values([x_dim_dim]) \

        plt.plot(
            aggregated_line_data[x_dim_dim],
            aggregated_line_data[y_dim_dim],
            label=synth_names[line_value],
            color=colors[line_value],
        )
        # Append last x-value
        last_values.append(aggregated_line_data[y_dim_dim].values)
        # last_values.append(
        #     round(aggregated_line_data[y_dim_dim].values[-1], 1))

    # plt.yscale("log")
    # MAX = 10000000000000000000

    flat_list = [item for sublist in last_values for item in sublist]
    # ymax = max(flat_list)

    ylim = [0, 40]

    plt.title(name)

    plt.ylim(ymax=max(y_ticks), ymin=0)
    plt.xlim(xmax=max(x_ticks), xmin=min(x_ticks))

    # remove the last tick since it looks bad on the x-axis
    plt.xticks(x_ticks[:-1])
    plt.yticks(y_ticks[:-1]) 


    # ML, time
    # plt.yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]) # H
    # plt.yticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 225]) # P


    # ML, memory
    # plt.yticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]) # H
    # plt.yticks([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]) # H
    # plt.yticks([1,2,3,4,5,6,7,8,9,10]) # P


    # SQ, Postgres, memory
    # plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])# 18, 20, 22, 24, 26, 28, 30])
    # SQ, memory    
    # plt.yticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250])# 18, 20, 22, 24, 26, 28, 30])

    # plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])# 18, 20, 22, 24, 26, 28, 30]) # SQ utility plots
    # plt.yticks([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]) # SQ utility plots
    # plt.yticks([0, 100, 200, 300, 400, 500, 600]) # SQ tools time
    # plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18]) # ML tools line plots
    # plt.yticks([100, 100000, 100000000, 100000000000, 100000000000000, 100000000000000000]) # Diffpriv line plots

    # plt.title(name)

    ax.set_title(name)

    plt.xlabel(x_dim_name)
    plt.ylabel(y_dim_name)

    plt.legend()
    plt.grid(True)

    # Append last values to right axes and color same as their respective line
    # second_axes = plt.twinx()
    # second_axes.set_ylim([0, 40])
    # second_axes.yaxis.set_tick_params(labelsize=9)
    # second_axes.set_yticks(last_values)
    # for color, tick in zip(colors, second_axes.yaxis.get_major_ticks()):
    #     tick.label2.set_color(color)
    _name = name.replace('ε', 'e')
    plt.savefig(
        '/dp-tools-evaluation/results/plots/' + _name + '.png',
        dpi=200,
        facecolor='w',
        edgecolor='w',
        orientation='portrait',
        bbox_inches='tight')

    plt.close()

