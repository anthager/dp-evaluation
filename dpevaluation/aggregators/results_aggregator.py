from numpy.core.numeric import Inf
from dpevaluation.statistical_queries.extractor import relative_result
from dpevaluation.utils.meta import Meta
from dpevaluation.statistical_queries.sq_utils import log
import pandas as pd


def interactive_data(ds):
    meta = Meta(None, ds)
    domains = meta._domains
    results = pd.DataFrame()

    for tool, domain in domains.items():
        if domain != 'statistical_queries' and domain != 'machine_learning':
            continue
            
        meta = Meta(tool, ds)
        raw_results = meta.load_latest_results()

        # the default will groupby both dataset_size and query and machine_learning data
        # does not contain any query column
        if domain == 'machine_learning':
            _results = relative_result(raw_results, groupby_dimensions=["dataset_size"])
        elif domain == 'statistical_queries':
            _results = relative_result(raw_results)

        _results['tool'] = tool

        # ignore the error if it is inf
        _results = _results[_results['result'] != Inf]

        results = results.append(_results, ignore_index=True)

    return results


def synthetic_SQ_data(ds):
    meta = Meta(None, ds)
    results = pd.DataFrame()

    for sd in meta.best_performing_synthetic_datasets():
        raw_results = meta.load_synthetic_SQ_result(sd)
        _results = relative_result(raw_results, metrics=['result'])

        config = meta.load_synthetic_configuration_with_fallback(sd)

        _results['tool'] = config['tool']
        _results['evaluation_tool'] = 'query'
        results = results.append(_results, ignore_index=True)

    return results


def synthetic_ML_data(ds):
    meta = Meta(None, ds)
    domains = meta._domains
    results = pd.DataFrame()


    for tool, _ in domains.items():
        if tool != 'tensorflow_privacy':
            continue

        meta = Meta(tool, ds, load_data=False)
        last_result = meta.load_latest_results()
        benchmark = last_result[last_result['epsilon'] == -1]


        for sd in meta.best_performing_synthetic_datasets():
            raw_results = None 
            try:
                raw_results = meta.load_synthetic_ML_result(sd)
            except:
                log('warn', f'ml result for {sd} for {tool} doesnt exist, run {tool}-evaluation with SYNTH_TEST=true')
                continue
            
            config = meta.load_synthetic_configuration_with_fallback(sd)
            benchmark_for_result = benchmark[benchmark['dataset_size'] == config['dataset_size']]

            raw_results = raw_results.append(benchmark_for_result, ignore_index=True)
            _results = relative_result(raw_results, groupby_dimensions=["dataset_size"], metrics=['result'])
            _results['evaluation_tool'] = tool
            _results['tool'] = config['tool']
            _results['sd'] = sd
            results = results.append(_results, ignore_index=True)

    return results



def aggregated_results(ds, cache = False):
    meta = Meta(None, ds)

    if cache:
        log('info', 'loading all aggregated results data from cache')
        try:
            return meta.load_aggregated_results()
        except:
            log('warn', 'didn\'t find any cache defaulting to load all data')

    data = pd.DataFrame()
    data = data.append(synthetic_ML_data(ds), ignore_index=True)
    data = data.append(synthetic_SQ_data(ds), ignore_index=True)
    data = data.append(interactive_data(ds), ignore_index=True)
    if data.shape[0] > 0:
        meta.save_aggregated_results(data)
    return data
