#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
from dpevaluation.aggregators.utils import remove_percentile_outliers_from_group
from dpevaluation.utils.log import log
import pandas as pd
import numpy as np
import re
import os


def main():
    csv = load_csv('')
    data = relative_result(csv)
    print(data.to_string())


# given a dataset with both non private (epsilon -1) and private (any other epsilon) will calculate the rmspe between them for all rows,
# works for histograms as well. propaged_values_from_priv are columns that should be kept from the private row in the resulting row
def relative_result(df, groupby_dimensions=["query", "dataset_size"], metrics=["result", "time"], propaged_values_from_priv=['epsilon']):
    log('info', "Extracting test result data...")

    # remove outliers from each group
    df = remove_percentile_outliers_from_group(df, 
        dimensions_for_grouping=[*groupby_dimensions, *propaged_values_from_priv], 
        lower=0.05, 
        upper=0.05)

    # select non-private results
    non_priv = df[df.epsilon == -
                   1].drop_duplicates(subset=groupby_dimensions)

    # select private results
    priv = df[(df.epsilon != -1)]

    after_merge_suffix_to_prefix = {
        '_x': 'non_priv_',
        '_y': 'priv_',
    }
    # will rename the col with prefix instead of suffix if there are a match, if not, skip
    def col_rename(col: str):
        for suffix, prefix in after_merge_suffix_to_prefix.items():
            # will match anything before suffix (positive lookahead)
            m = re.search(f'.*(?={suffix}$)', col)
            if m is not None:
                if m.group() in propaged_values_from_priv and suffix == '_y':
                    return m.group()
                return prefix + m.group()
        return col

    # inner join on query, dataset_size
    # groupby query, epsilon, dataset_size
    # calc rmspe for each epsilon value for each query and dataset_size
    data = pd \
        .merge(non_priv, priv, on=groupby_dimensions) \
        .rename(columns=col_rename) \
        .groupby(groupby_dimensions + propaged_values_from_priv) \
        .apply(lambda row: reduce_metrics(row, metrics)) \
        .reset_index()

    log('info', "Extraction done")

    return data


def load_csv(file_name):
    log('debug', "Loading test result data...")

    latest_test, path = utils.get_latest_test(file_name, "SQ")
    data = pd.read_csv(path)

    log('debug', "%s loaded" % latest_test)

    return data


def reduce_metrics(arr: pd.DataFrame, metrics):
    res = {}
    for metric in metrics:
        _priv = arr['priv_' + metric]
        _non_priv = arr['non_priv_' + metric]

        try:
            # convert to float, will throw if it fails
            priv = _priv.astype(np.float)
            non_priv = _non_priv.astype(np.float)
        except:
            # if the convertion to float fails, assume the result is an array of ints i.e. histogram
            p = np.array(_priv.to_numpy()[
                0].split(','), np.int)
            n_p = np.array(_non_priv.to_numpy()[
                0].split(','), np.int)

            priv_bin = np.array(
                arr['priv_' + metric + '_bin'].to_numpy()[0].split(','), np.float)
            non_priv_bin = np.array(arr['non_priv_' + metric + '_bin'].to_numpy()[
                                    0].split(','), np.float)

            priv_df = pd.DataFrame(dict(zip(priv_bin, p)), index=['priv'])
            non_priv_df = pd.DataFrame(dict(zip(non_priv_bin, n_p)), index=['non_priv'])

            df = priv_df.append(non_priv_df)
            priv = df.loc['priv']
            non_priv = df.loc['non_priv']



        # we will only get here when if the result isn't a histogram
        _rmspe = rmspe(non_priv, priv)
        res[metric] = _rmspe

    return pd.Series(res)


def rmspe(a, b):
    if a is None or b is None:
        return None

    p_err = (a - b) / a
    squared_p_err = np.square(p_err)
    # axis=0 makes it explicit that the error is computed row-wise
    mean_squared_p_err = np.mean(squared_p_err, axis=0)
    root_mean_squared_p_err = np.sqrt(mean_squared_p_err)
    return root_mean_squared_p_err * 100


def get_epsilons(data):
    return data.drop_duplicates(subset=["epsilon"])["epsilon"].to_numpy()


def get_dataset_sizes(data):
    return data.drop_duplicates(subset=["dataset_size"])["dataset_size"].to_numpy()


def get_rmspe(data, rmspe):
    results = data.groupby(["query", "dataset_size"])[rmspe]
    # {query: [[], [], ...], [[], [], ...], ...}
    rmspe = dict()
    # For each query add all dataset_size -> for each dataset_size append rmspe for all epsilons
    for i in results:
        query = i[0][0]
        values = i[1].values
        # Add default value [] if key (query) does not exist
        rmspe.setdefault(query, []).append(values)

    return rmspe


def get_avg_rmspe(data, epsilons, rmspe):
    # {("query", "dataset_size"): [accuracy, ...]}
    results = data.groupby(["query", "dataset_size"])[rmspe]
    means = dict()
    identity_arr = [0] * len(epsilons)
    # For each query for each dataset_size -> calc average rmspe for all epsilons
    for i in results:
        size = i[0][1]
        values = i[1].values
        mean = means.get(size, identity_arr)
        means[size] = np.array([mean, values]).mean(axis=0)

    return {"all": list(means.values())}


if __name__ == '__main__':
    main()
