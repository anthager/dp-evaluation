from dpevaluation.aggregators.utils import remove_percentile_outliers_from_group
from posix import listdir
from dpevaluation.utils.meta import Meta
from dpevaluation.statistical_queries.sq_utils import log
import pandas as pd
import numpy as np
import os

def all(cache = False):
    cache_path = Meta._memory_path + 'cache.csv'
    res = pd.DataFrame()

    if cache:
        log('info', 'loading all memory data from cache')
        try:
            return pd.read_csv(cache_path)
        except:
            log('warn', 'didnt find any cache defaulting to load all data')

    log('info', 'loading all memory data without cache')
    for ds in Meta.all_datasets():
        # Remove outliers
        try:
            _res = memory(ds)
            res = res.append(remove_percentile_outliers_from_group(
                _res,
                dimensions_for_grouping=["epsilon",
                                            "dataset_size",
                                            "query",
                                            "tool",
                                            "dataset"],
                metric='memory',
                lower=0.05, upper=0.05), ignore_index=True)

            log('info', f'saving all memory data to {cache_path}...')
        except:
            log('warn', f'no memory data for {ds}')
    res.to_csv(cache_path, index=False)

    return res

def memory(ds):
    meta = Meta(None, ds, load_data=False)
    domains = meta._domains
    results = pd.DataFrame()

    # tools = ['smartnoise_dpctgan', 'smartnoise_patectgan', 'smartnoise_mwem']
    tools = domains.items()

    for tool, _ in tools:

        m = Meta(tool, ds, load_data=False)

        res = m.load_monitoring_data()
        
        results = results.append(res, ignore_index=True)
    return results
