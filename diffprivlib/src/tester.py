#!/usr/bin/python3.7

import time

from dpevaluation.machine_learning.model_factory import ModelFactory
from dpevaluation.utils.meta import Meta
from dpevaluation.utils.log import log, log_test_params
from dpevaluation.machine_learning.ml_utils import get_datasets, load_datasets, log_info, split_and_normalize_data

import pandas as pd
import os

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def main(dataset, synthetic_test, memory_test):
    meta = Meta("diffprivlib", dataset)
    # Get benchmark datasets (and synthetic dataset names if synthetic test).
    datasets, benchmark_datasets = get_datasets(synthetic_test, meta)
    log_test_params(synthetic_test, memory_test, meta, len(datasets))

    results = []

    for ds in datasets:
        '''
        If we running synthetic test, ds will be the name of the dataset and the actual
        dataset otherwise. The name is needed when we later saves the results of the synth run
        '''
        synthetic_name = ds
        '''
        If synthetic test: we load the synthetic dataset och get their config 
        files, else: return ds from benchmark datasets for private and non-private 
        test.
        '''
        benchmark_dataset = None
        synthetic_config = None
        try:
            ds, benchmark_dataset, synthetic_config = \
                load_datasets(synthetic_test, meta, ds, benchmark_datasets)
        except:
            continue

        x_train, y_train, x_test, y_test, \
            x_train_min, x_train_max, \
            y_train_min, y_train_max = prepare_data(
                ds, benchmark_dataset, meta.target)

        for epsilon in meta.epsilons:
            for _ in range(meta.num_of_runs):
                # Build model
                model = ModelFactory("diffprivlib").build_model(
                    model=meta.model,
                    optimizer=None,
                    config={
                        "epsilon": epsilon,
                        "bounds_X": (x_train_min, x_train_max),
                        "bounds_y": (y_train_min, y_train_max)})
                '''
                Set mode and epsilon (based on mode) after building model with epsilon
                for private and non-private training so we can save epsilon of 
                synthetic dataset if we are running synthetic test.
                '''
                mode = "synthetic" if synthetic_test else "non_private" if epsilon == -1 else "private"
                epsilon = synthetic_config["epsilon"] if mode == "synthetic" else epsilon
                log_info(mode, ds, epsilon, None, None)
                start_time = time.time()
                # Train model
                model.fit(x_train, y_train)
                elapsed_time = time.time() - start_time
                # Evaluate model
                predictions = model.predict(x_test)
                mse = mean_squared_error(y_test, predictions)
                mape = mean_absolute_percentage_error(y_test, predictions)
                log("debug", 'mse: %.4f, mape: %.4f' % (mse, mape))

                if not memory_test:
                    values = {
                        "epsilon": epsilon,
                        "dataset_size": ds.shape[0],
                        "mse": mse,
                        # to reuse other stuff call the metric we want to measure result
                        "result": mse,
                        "time": elapsed_time,
                        "mape": mape,
                        "y_test": y_test,
                        "prediction": predictions.tolist()
                    }
                    if synthetic_test:
                        meta.save_synthetic_ML_result(pd.DataFrame([values]), synthetic_name)
                    else:
                        results.append(values.copy())
                # Run only once for non-private
                if epsilon == -1:
                    break

    if not memory_test and not synthetic_test:
        meta.save_results(results)


def prepare_data(ds, benchmark_data, target):
    x_train, y_train, x_test, y_test, _ = split_and_normalize_data(
        "diffprivlib", ds, benchmark_data, target, None)

    x_train_min = x_train.min().values
    x_train_max = x_train.max().values

    x_train = x_train.values
    y_train = y_train.values.tolist()
    x_test = x_test.values
    y_test = y_test.values.tolist()

    y_train_min = min(y_train)
    y_train_max = max(y_train)

    return x_train, y_train, x_test, y_test, x_train_min, x_train_max, y_train_min, y_train_max


if __name__ == "__main__":
    dataset = os.environ['DATASET'] if 'DATASET' in os.environ else "medical-survey"
    synthetic_test = True if 'SYNTH_TEST' in os.environ else False
    memory_test = True if 'MEM_TEST' in os.environ else False
    main(dataset, synthetic_test, memory_test)
