from dpevaluation.utils.log import log
from dpevaluation.utils.meta import Meta
from dpevaluation.statistical_queries.sq_utils import print_progress_extended
from sdv.metrics.tabular import CSTest, KSTest
import pandas as pd
import numpy as np


def score_of_synthtic_dataset(ds, private_data):
    meta = Meta(None, ds)
    synth_data = meta.load_synthetic_data(private_data)
    meta.only_use_allowlisted_columns()
    m = meta.get_evaluation_meta()


    ks_score = KSTest.compute(meta.parsed_dataset, synth_data, metadata=m)
    cs_score = CSTest.compute(meta.parsed_dataset, synth_data, metadata=m)

    def or_one(v):
        return v if not np.isnan(v) else 1.0

    score = or_one(ks_score) * or_one(cs_score)
    
    return score

def data_with_score(ds, private_data):
    meta = Meta(None, ds)
    score = score_of_synthtic_dataset(ds, private_data)
    private_config = meta.load_synthetic_configuration_with_fallback(private_data)
    private_config['score'] = score
    private_config['dataset'] = ds
    private_config['synthetic_dir'] = private_data

    log('info', f'{private_data}: {score}')
    return private_config


def dataset_scores(ds):
    meta = Meta(None, ds, load_data=False)
    result = pd.DataFrame()
    private_data_dirs = meta.synthetic_datasets()

    for i, pdd in enumerate(private_data_dirs):
        res = data_with_score(ds, pdd)
        result = result.append(res, ignore_index=True)
        print_progress_extended(i, len(private_data_dirs))

    meta.save_synthetic_SQ_scores(result)


def dataset_with_least_score(scores_df):
    scores_df['min_score'] = scores_df.groupby('dataset').score.transform(np.min)
    scores_df = scores_df[scores_df.min_score == scores_df.score]
    return scores_df
