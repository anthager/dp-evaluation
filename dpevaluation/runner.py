
from pandas.core.indexes.base import Index
from dpevaluation.utils.meta import Meta
from dpevaluation.utils.log import log, log_elapsed_time
from dpevaluation.statistical_queries import sq_utils

import itertools
import os
import time
import pandas as pd


def run_tests(generate, meta, configuration, additional_data={'test_start_time' : time.time()}, merge_result=False):
    configuration_values = configuration.values()

    carth_product_of_configuration = list(itertools.product(*configuration_values))

    num_of_tests = len(carth_product_of_configuration)

    log("info", "Running tests...")
    log("debug", "Number of tests: %s" % num_of_tests)
    for key, values in configuration.items():
        log("debug", "{}: {}".format(key, values))

    results = []

    for i, values in enumerate(carth_product_of_configuration):
        config = dict(zip(configuration.keys(), values))
        sq_utils.print_progress_extended(i, num_of_tests, config)
        res = generate(meta, {**config, **additional_data})

        if merge_result:
            header, rows = res
            m_rows = sq_utils.merge_rows(rows)

            if m_rows is None:
                continue
            # add configuration and additional_data to results
            for key, value in {**config, **additional_data}.items():
                # obviously skip functions
                if callable(value):
                    continue
                header.append(key)
                m_rows.append(value)

            res = dict(zip(header, m_rows))
            results.append(res)


    end_time = time.time()
    log_elapsed_time(additional_data['test_start_time'], end_time)

    if merge_result:
        results = pd.DataFrame(results)

    return results, additional_data['test_start_time'] - end_time
