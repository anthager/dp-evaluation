# -*- coding: utf-8 -*-

from dpevaluation.utils.log import log
from dpevaluation.runner import run_tests
from dpevaluation.utils.meta import Meta

from opendp.smartnoise.synthesizers.mwem import MWEMSynthesizer
from opendp.smartnoise.synthesizers.pytorch.nn.dpctgan import DPCTGAN
from opendp.smartnoise.synthesizers.pytorch.nn.patectgan import PATECTGAN
from opendp.smartnoise.synthesizers.pytorch.pytorch_synthesizer import PytorchDPSynthesizer

import os
import time
import pandas as pd

invalid_per_valid = 100

def generate_rows(meta, synth_model, dataset_size):
    # we want the new dataset to be the same size as the original dataset, otherwise count, sum and histograms queries work poorly
    synthetic_dataset = pd.DataFrame()
    invalid = 0
    while(synthetic_dataset.shape[0] < dataset_size):
        new_valid_rows = meta.validate_rows(synth_model.sample(dataset_size))

        synthetic_dataset = synthetic_dataset.append(new_valid_rows).head(dataset_size)
        valid = synthetic_dataset.shape[0]
        invalid += dataset_size - new_valid_rows.shape[0]

        # protect against /0 by adding with 1
        invalid_per_valid = invalid / (valid + 1)
        log('info', f'{valid}/{dataset_size} rows. invalid ratio: {round(invalid_per_valid, 2)}')
        if invalid > 1000 and invalid_per_valid > 10000:
            log('warn', f'too many invalid rows which means shitty model, aborting.....')
            raise Exception('shitty model')

    # make sure all histogram columns still are ints (or at least floats w/o decimals)
    hist_cols = [col['name'] for col in meta.histogram_columns]
    round_dict = dict([(c, 0) for c in hist_cols])
    synthetic_dataset = synthetic_dataset.round(round_dict)

    return synthetic_dataset



def mwem_generator(meta: Meta, test_params):
    start_time = time.time()

    epsilon = test_params['epsilon']
    dataset_size = test_params['dataset_size']
    query_count = test_params['query_count']
    data_subset = meta.parsed_dataset.head(dataset_size)
    split_factor = len(data_subset.columns)

    synth_model = MWEMSynthesizer(
        epsilon=epsilon, q_count=query_count, split_factor=split_factor)
    synth_model.fit(data_subset, categorical_columns=meta.categorical_columns,
                    ordinal_columns=meta.ordinal_columns)

    try:
        synthetic_dataset = generate_rows(meta, synth_model, dataset_size)

        tp = test_params.copy()
        tp['generate_time'] = time.time() - start_time

        meta.save_synthetic_data(synthetic_dataset, tp)
        return synthetic_dataset
    except:
        pass


def dpctgan_generator(meta: Meta, test_params):
    start_time = time.time()

    epsilon = test_params['epsilon']
    dataset_size = test_params['dataset_size']
    epochs = test_params['epochs']
    batch_size = test_params['batch_size']
    data_subset = meta.parsed_dataset.head(dataset_size)

    gan = DPCTGAN(
        epsilon=epsilon,
        epochs=epochs,
        batch_size=batch_size,
    )
    synth_model = PytorchDPSynthesizer(
        preprocessor=None, gan=gan, epsilon=epsilon)
    synth_model.fit(data_subset, categorical_columns=meta.categorical_columns,
                    ordinal_columns=meta.ordinal_columns)
    
    try:
        synthetic_dataset = generate_rows(meta, synth_model, dataset_size)

        tp = test_params.copy()
        tp['generate_time'] = time.time() - start_time

        meta.save_synthetic_data(synthetic_dataset, tp)
        return synthetic_dataset
    except:
        pass

def patectgan_generator(meta: Meta, test_params):
    start_time = time.time()

    epsilon = test_params['epsilon']
    dataset_size = test_params['dataset_size']
    epochs = test_params['epochs']
    batch_size = test_params['batch_size']
    # teacher_iters = test_params['teacher_iters']
    # student_iters = test_params['student_iters']
    # sample_per_teacher = test_params['sample_per_teacher']
    # noise_multiplier = test_params['noise_multiplier']
    data_subset = meta.parsed_dataset.head(dataset_size)

    gan = PATECTGAN(
        epsilon=epsilon,
        epochs=epochs,
        batch_size=batch_size,
        # teacher_iters=teacher_iters,
        # student_iters=student_iters,
        # sample_per_teacher=sample_per_teacher,
        # noise_multiplier=noise_multiplier,
    )
    synth_model = PytorchDPSynthesizer(
        preprocessor=None, gan=gan, epsilon=epsilon)
    synth_model.fit(data_subset, categorical_columns=meta.categorical_columns,
                    ordinal_columns=meta.ordinal_columns)
    
    try:
        synthetic_dataset = generate_rows(meta, synth_model, dataset_size)

        tp = test_params.copy()
        tp['generate_time'] = time.time() - start_time

        meta.save_synthetic_data(synthetic_dataset, tp)
        return synthetic_dataset
    except:
        pass


tool_to_generator = {
    'smartnoise_mwem': mwem_generator,
    'smartnoise_dpctgan': dpctgan_generator,
    'smartnoise_patectgan': patectgan_generator,
}


def main(tool, ds):
    meta = Meta(tool, ds)
    meta.only_use_allowlisted_columns()
    additional_data = {'test_start_time': time.time(), 'tool': tool}


    if tool not in tool_to_generator:
        log('error', tool + ' not in meta')
        quit(0)


    log('debug', f'Tool: {tool}')
    configuration = {
        **{
            'dataset_size': meta._dataset_sizes,
            'epsilon': [e for e in meta.epsilons if e != -1],
            'num_of_runs': list(range(meta.num_of_runs)),
        },
        **meta.tool_specific_meta()
    }
    generator = tool_to_generator[tool]

    run_tests(generator, meta, configuration, additional_data)



if __name__ == "__main__":
    tool = os.environ['TOOL']
    ds = os.environ['DATASET']
    if not ds:
        raise('make sure the environment variable DATASET is set')
    main(tool, ds)
