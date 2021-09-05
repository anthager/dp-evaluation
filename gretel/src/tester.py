# Train a synthetic data model with differential privacy guarantees
from dpevaluation.utils.log import log
import os
import numpy as np
import time
import pandas as pd

import pandas as pd

from dpevaluation.runner import run_tests
from dpevaluation.utils.meta import Meta


try:
    from gretel_helpers.synthetics import SyntheticDataBundle
except FileNotFoundError:
    from gretel_helpers.synthetics import SyntheticDataBundle



def gretel_generator(meta: Meta, test_params):
    dataset_size = test_params['dataset_size']
    data_subset = meta.parsed_dataset.head(dataset_size)

    start_time = time.time()
    checkpoint_dir = '/checkpoint'

    config_template = {
        "max_lines": test_params['max_lines']                     if 'max_lines' in test_params else 1e5,
        "predict_batch_size": test_params['predict_batch_size']   if 'predict_batch_size' in test_params else 1,
        "rnn_units": test_params['rnn_units']                     if 'rnn_units' in test_params else 32, # 128 originally
        "batch_size": test_params['batch_size']                   if 'batch_size' in test_params else 4, # 128 originally
        "learning_rate": test_params['learning_rate']             if 'learning_rate' in test_params else .001, #0.005 originally
        "dp_noise_multiplier": test_params['dp_noise_multiplier'] if 'dp_noise_multiplier' in test_params else .3, # 1 originally
        "dp_l2_norm_clip": test_params['dp_l2_norm_clip']         if 'dp_l2_norm_clip' in test_params else .1, # 0.1 originally
        "dropout_rate": 0.5,
        "reset_states": False,
        "overwrite": True,
        "checkpoint_dir" : checkpoint_dir,
        "dp": True,
        "dp_microbatches": 1,
        "epochs": 100,
    }


    model = SyntheticDataBundle(
        training_df=data_subset,
        auto_validate=True, # build record validators that learn per-column, these are used to ensure generated records have the same composition as the original
        synthetic_config=config_template, # the config for Synthetics
    )

    model.build()
    model.train()


    synthetic_dataset = pd.DataFrame()

    iteration = 0
    rows_to_generate = dataset_size
    gretel_valid = 100
    invalid_per_gretel_valid = 50
    gretel_invalid = gretel_valid * invalid_per_gretel_valid
    invalid_per_valid = 250


    while(synthetic_dataset.shape[0] < rows_to_generate):
        
        model.generate(num_lines=gretel_valid, max_invalid=gretel_invalid)
        _synthetic_dataset = model.get_synthetic_df()
        validated_rows = meta.validate_rows(_synthetic_dataset)
        synthetic_dataset = synthetic_dataset.append(validated_rows, ignore_index=True)

        log('debug', f'Validated rows: {validated_rows.shape[0]}')
        log('debug', f'Iteration {iteration}: {synthetic_dataset.shape[0]}/{rows_to_generate}')
        # if the validated rows are less than the threshold, abort
        if (validated_rows.shape[0] < gretel_invalid / invalid_per_valid):
            log('warn', 'aborting due to bad model')
            return


    tp = test_params.copy()
    tp['generate_time'] = time.time() - start_time

    # make sure all histogram columns still are ints (or at least floats w/o decimals)
    hist_cols = [col['name'] for col in meta.histogram_columns]
    round_dict = dict([(c, 0) for c in hist_cols])
    synthetic_dataset = synthetic_dataset.round(round_dict)

    # get privacy guarantees
    priv = pd.read_csv(f"{checkpoint_dir}/batch_0/model_history.csv")
    epsilon = priv[priv['best'] == 1]['epsilon'].values[0]
    delta = priv[priv['best'] == 1]['delta'].values[0]
    tp['epsilon'] = int(epsilon)
    tp['delta'] = int(delta)

    meta.save_synthetic_data(synthetic_dataset, tp)
    return synthetic_dataset


def main(ds):
    tool = 'gretel'
    meta = Meta(tool, ds)
    meta.only_use_allowlisted_columns()

    additional_data = {'test_start_time': time.time(), 'tool': tool}

    configuration = {
        **{
            'dataset_size': meta.dataset_sizes,
            'num_of_runs': list(range(meta.num_of_runs)),
        },
        **meta.tool_specific_meta()
    }

    run_tests(gretel_generator, meta, configuration, additional_data)
    



if __name__ == "__main__":
    ds = os.environ['DATASET']
    if not ds:
        raise('make sure the environment variable DATASET is set')
    main(ds)
