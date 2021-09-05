#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

from dpevaluation.machine_learning import extractor
from dpevaluation.utils.utils import RESULTS_PATH
from dpevaluation.utils.log import log
from dpevaluation.utils.timestamp import timestamp
import matplotlib.pyplot as plt
import matplotlib
from math import ceil
import numpy as np
import sys
import os


def main():
    """
    python3.7 plot_builder.py <FILE_NAME> "OPACUS"|"TF-PRIV
    """
    try:
        file_name = sys.argv[1:][0]
        lib = sys.argv[1:][1]  # OPACUS or TF-PRIV
    except Exception as e:
        log("error", "No file_name and lib args provided")
        exit(1)

    log('info', "Building plots...")
    # Extract data from csv
    csv = extractor.load_csv(file_name)
    data = extractor.extract_data(csv)
    # Set size etc.
    plt.rcParams["figure.figsize"] = [12, 12]
    matplotlib.rcParams.update({
        "axes.titlesize": 22,
        "axes.titlepad": 12.0,
        "legend.fontsize": 20.0,
        "axes.labelsize": 20,
        "axes.labelpad": 2.0,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18})
    plot_prediction(lib, file_name, data)
    # plot_error(lib, file_name, data)

    log('info', "Plots done")


def plot_prediction(lib, file_name, data):
    for i, val in data.iterrows():
        # Get x/y-axis max
        ax_len = ceil(
            max(max(val["prediction"]) + 1, max(val["y_test"]) + 1))
        a = plt.axes(aspect='equal')

        if np.isnan(val["epsilon"]):
            mode = "non-private"
            label = "epochs=%s" % val["epoch"]
        else:
            mode = "private"
            label = "e=%s, ε=%.4f, nm=%.4f mgn=%.1f" % (
                val["epoch"],
                val["epsilon"],
                val["noise_multiplier"],
                val["l2_norm_clip"] if lib == "TF-PRIV" else val["max_grad_norm"])

        a.set_title("%s (%s)" % (mode, label))

        plt.rcParams["figure.figsize"] = [12, 12]
        plt.scatter(
            val["y_test"],
            val["prediction"],
            marker='.',
            s=40,
            alpha=0.75)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')

        lims = [0, ax_len]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)

        save_plot(file_name, mode)


def plot_error(lib, file_name, data):
    # Non-private error
    # np_mapes = data[data["epsilon"].isnull()]["mape"].values[0]
    np_mapes = []
    # data = data.groupby("epsilon")
    data = data.groupby("epoch")
    for val in data:
        # epsilon = val[0]
        # [[mape, ...], [...], [...]]
        mapes = val[1]["mape"].values.tolist()
        # Flatten
        fmapes = [i for subl in mapes for i in subl]
        # Find min/max error
        len_x_axis = len(mapes[0])
        start_y_axis, end_y_axis = \
            min(fmapes + np_mapes), max(fmapes + np_mapes)
        # Corresponding noise_multiplier for each epsilon
        # noise_multiplier = val[1]["noise_multiplier"].values[0]
        # Plot non-private error
        # plt.plot(np_mapes, label="non-private", color="blue")
        # Save last y-value
        # last_values = []
        # last_values.append(np_mapes[-1])
        # Use same color of dashed lines for their last y-value
        colors = ["blue", "purple", "green", "red"]
        # mgn = max_grad_norm
        # items = val[1]["l2_norm_clip"].values if lib == "TF-PRIV" \
        #     else val[1]["max_grad_norm"].values
        # for i, grad_norm in enumerate(items):
        #     _label = "l2_norm_clip=%.1f" % grad_norm if lib == "TF-PRIV" \
        #         else "max_grad_norm=%.1f" % grad_norm
        #     plt.plot(
        #         mapes[i], label=_label, linestyle='dashed', color=colors[i + 1])
        items = val[1]["epsilon"].values
        for i, epsilon in enumerate(items):
            if np.isnan(epsilon):
                plt.plot(np_mapes,
                         label="non-private (%.2f)" % mapes[i][-1])
                # color=colors[i])
            else:
                plt.plot(
                    mapes[i],
                    label="ε=%.4f (%.2f)" % (epsilon, mapes[i][-1]),
                    linestyle='dashed')
                # color=colors[i])

            # last_val = mapes[i][-1]
            # last_values.append(last_val)

        ylim = [start_y_axis - 1, end_y_axis]
        plt.ylim(ylim)
        plt.xlim([0, len_x_axis - 1])

        plt.xlabel('Epoch')
        plt.ylabel('MAPE')
        # plt.title("Epochs=%s (ε=%.4f, noise_multiplier=%.4f)" %
        #           (len_epochs, epsilon, noise_multiplier))
        plt.grid(True)
        plt.legend()
        # Append last values to right axes and color same as their respective line
        # second_axes = plt.twinx()
        # second_axes.set_ylim(ylim)
        # second_axes.yaxis.set_tick_params(labelsize=12)
        # second_axes.set_yticks(last_values)
        # for color, tick in zip(colors, second_axes.yaxis.get_major_ticks()):
        #     tick.label2.set_color(color)

        save_plot(file_name, "error")


def save_plot(file_name, mode):
    os.makedirs(RESULTS_PATH["ML"], exist_ok=True)
    path = "%s%s-%s-%s%s" % \
        (RESULTS_PATH["ML"], file_name, mode, timestamp("%S%f"), '.png')
    plt.savefig(path)
    plt.clf()


if __name__ == '__main__':
    main()
