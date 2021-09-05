#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

from dpevaluation.utils import utils
from dpevaluation.utils.log import log
from ast import literal_eval
import pandas as pd
import numpy as np
import os


def run():
    csv = load_csv("opacus")
    data = extract_data(csv)


def extract_data(csv):
    log('info', "Extracting test result data...")

    data = csv.sort_values(by=["epsilon"])
    data["mse"] = data["mse"].apply(literal_eval)
    data["mape"] = data["mape"].apply(literal_eval)
    data["y_test"] = data["y_test"].apply(literal_eval)
    data["prediction"] = data["prediction"].apply(literal_eval)

    return data


def load_csv(file_name):
    log('debug', "Loading test result data...")

    latest_test, path = utils.get_latest_test(file_name, "ML")
    data = pd.read_csv(path)

    log('debug', "%s loaded" % latest_test)

    return data


if __name__ == '__main__':
    run()
