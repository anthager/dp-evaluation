#!/usr/bin/python3.7

from dpevaluation.utils.meta import Meta
from dpevaluation.utils.log import log_elapsed_time, log_test_params
from dpevaluation.machine_learning.ml_utils import get_datasets, load_datasets, split_and_normalize_data, log_info, log_losses
from dpevaluation.machine_learning.dp_utils import DP_Utils
from dpevaluation.machine_learning.model_factory import ModelFactory

import pandas as pd

import time
import os


def main(dataset, trace_epsilon_calc, synthetic_test, memory_test):
    meta = Meta("opacus", dataset)
    # Get benchmark datasets (and synthetic dataset names if synthetic test).
    datasets, benchmark_datasets = get_datasets(synthetic_test, meta)
    log_test_params(synthetic_test, memory_test, meta, len(datasets))
    dp_utils = DP_Utils("opacus", meta.epsilons, meta.batch_size, meta.epochs)
    results = []

    for ds in datasets:
        '''
        If we running synthetic test, ds will be the name of the dataset and the actual
        dataset otherwise. The name is needed when we later saves the results of the synth run
        '''
        synthetic_name = ds
        '''
        If synthetic test: we load the synthetic dataset och get their config 
        files, else return ds from benchmark datasets for private and non-private 
        test.
        '''
        benchmark_dataset = None
        synthetic_config = None
        try:
            ds, benchmark_dataset, synthetic_config = \
                load_datasets(synthetic_test, meta, ds, benchmark_datasets)
        except:
            continue

        train_loader, test_loader, input_shape = split_and_normalize_data(
            "opacus", ds, benchmark_dataset, meta.target, meta.batch_size)

        sample_size = len(train_loader.dataset)
        delta = 1 / (sample_size * 2)

        if synthetic_test:
            noise_multipliers = [-1]
            epsilons = [synthetic_config["epsilon"]]
            opt_orders = None
        else:
            noise_multipliers, epsilons, opt_orders = dp_utils.calc_noise_multipliers(
                sample_size, delta, trace_epsilon_calc)

        for max_grad_norm in meta.norm_clips:
            for i, noise_multiplier in enumerate(noise_multipliers):
                for _ in range(meta.num_of_runs):
                    mode = "synthetic" if synthetic_test else "non_private" if noise_multiplier == -1 else "private"
                    log_info(
                        mode, ds, epsilons[i], noise_multiplier, max_grad_norm)
                    # Build model
                    model = ModelFactory("opacus").build_model(
                        model=meta.model,
                        optimizer=meta.optimizer,
                        config={
                            "input_shape": input_shape,
                            "learning_rates": meta.learning_rate,
                            "momentum": 0,
                            "nesterov": False,
                            "sample_size": sample_size,
                            "batch_size": meta.batch_size,
                            "max_grad_norm": max_grad_norm,
                            "noise_multiplier": noise_multiplier,
                            "alphas": opt_orders,
                            "secure_rng": True,
                            "target_delta": delta,
                            "epochs": meta.epochs})
                    start_time = time.time()
                    # Train model
                    losses = model.fit(mode, train_loader, delta)
                    end_time = time.time()
                    log_losses("Training", losses)
                    # Evaluate model
                    metrics, y_test, predictions = model.evaluate(test_loader)
                    print(metrics)
                    log_losses("Evaluation", metrics)
                    log_elapsed_time(start_time, end_time)
                    # Save results
                    if not memory_test:
                        values = {
                            "epsilon": epsilons[i],
                            "dataset_size": ds.shape[0],
                            "mse": metrics["mse"],
                            # to reuse other stuff call the metric we want to measure result
                            "result": metrics["mse"],
                            "time": end_time - start_time,
                            "epochs": meta.epochs,
                            "batch_size": meta.batch_size,
                            "sample_size": sample_size,
                            "delta": delta,
                            "noise_multiplier": noise_multiplier,
                            "max_grad_norm": max_grad_norm,
                            "mape": metrics["mape"],
                            "y_test": y_test.tolist(),
                            "prediction": predictions.tolist()
                        }
                        if synthetic_test:
                            meta.save_synthetic_ML_result(pd.DataFrame([values]), synthetic_name)
                        else:
                            results.append(values.copy())


                    # Run only once for non-private
                    if mode == "non_private":
                        break

    if not memory_test and not synthetic_test:
        meta.save_results(results)


if __name__ == "__main__":
    dataset = os.environ['DATASET'] if 'DATASET' in os.environ else "medical-survey"
    trace_epsilon_calc = True if 'TRACE' in os.environ else False
    synthetic_test = True if 'SYNTH_TEST' in os.environ else False
    memory_test = True if 'MEM_TEST' in os.environ else False
    main(dataset, trace_epsilon_calc, synthetic_test, memory_test)

