# -*- coding: utf-8 -*-

from dpevaluation.queries import QUIERIES
from dpevaluation.utils.log import log, log_elapsed_time
from dpevaluation.statistical_queries import sq_utils, bootstrap_db
from dpevaluation.utils.meta import Meta
import time
import os


def run_test(run_query,
             tool,
             ds='medical-survey',
             rewrite=False,
             ):

    mem_test = 'MEM_TEST' in os.environ
    meta = Meta(tool, ds)

    if mem_test:
        return __run_mem_test__(run_query, meta)
    else:
        bootstrap_db.run(ds, meta.dataset_sizes, rewrite)
        results = __run_test__(run_query, meta)
        # Save results
        meta.save_results(results)
        return results

# in bytes
def tot_allocated(snapshot):
    return sum(stat.size for stat in snapshot.statistics('lineno'))

# it's not set in stone that we want to ONLY use metadata for running tests. We might want to do it in another way
# for synthetic data while still use some of this logic
def __run_test__(run_query, meta: Meta):
    start_time = time.time()

    num_of_runs = meta.num_of_runs
    epsilons = meta.epsilons
    dataset_sizes = meta.dataset_sizes
    columns = meta.non_private_column_names

    num_of_tests = \
        len(epsilons) * \
        len(dataset_sizes) * \
        len(QUIERIES) * \
        num_of_runs

    log("info", "Running tests...")
    log("debug", "Number of tests: %s" % num_of_tests)
    log("debug", "Dataset sizes: %s" % str(dataset_sizes)[1:-1])
    log("debug", "Epsilons:\n%s" % str(epsilons)[1:-1])

    results = []
    count = 0

    for name, q in meta.queries.items():

        for col in columns:

            is_histogram_query = name == 'histogram'
            if is_histogram_query and col not in [c['name'] for c in meta.histogram_columns]:
                continue

            query = q.format(column_name=col)

            for size in dataset_sizes:
                log('debug', "q: %s" % query)

                for epsilon in epsilons:
                    for _ in range(num_of_runs):

                        start = time.time()

                        header, rows = run_query(
                            epsilon,
                            size,
                            query,
                        )

                        elapsed_time = time.time() - start

                        merged_rows = sq_utils.merge_rows(rows)

                        # no rows returned, can happen when epsilon is low and dataset_size small and all data is censored
                        if merged_rows is None:
                            continue

                        values = {
                            "epsilon": epsilon,
                            "dataset_size": size,
                            "time": elapsed_time
                        }

                        for i, col in enumerate(header):
                            values[col] = merged_rows[i]

                        results.append(values)
                        count += 1
                        sq_utils.print_progress(epsilon, size, count)

    end_time = time.time()
    log_elapsed_time(start_time, end_time)

    return results


def __run_mem_test__(run_query, meta: Meta):
    num_of_runs = meta.num_of_runs
    epsilons = meta.epsilons
    dataset_sizes = meta._dataset_sizes
    columns = meta.non_private_column_names

    num_of_tests = \
        len(epsilons) * \
        len(dataset_sizes) * \
        len(meta.queries) * \
        len(columns) * \
        num_of_runs
    count = 0

    for name, q in meta.queries.items():

        for col in columns:
            query = q.format(column_name=col)

            is_histogram_query = name == 'histogram'
            if is_histogram_query and col not in [c['name'] for c in meta.histogram_columns]:
                continue

            for size in dataset_sizes:

                for epsilon in epsilons:
                    for _ in range(num_of_runs):
                        
                        run_query(
                            epsilon,
                            size,
                            query,
                        )

                        count += 1
                        sq_utils.print_progress_extended(count, num_of_tests)

