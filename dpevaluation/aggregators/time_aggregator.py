from dpevaluation.aggregators.utils import remove_percentile_outliers_from_group
from dpevaluation.utils.meta import Meta
from dpevaluation.statistical_queries.sq_utils import log
import pandas as pd


def interactive_data(ds):
    meta = Meta(None, ds)
    domains = meta._domains
    results = pd.DataFrame()

    for tool, domain in domains.items():
        if domain == 'statistical_queries':
            groupby = ['dataset_size', 'query']
        elif domain == 'machine_learning':
            groupby = ['dataset_size']
        else:
            continue


        meta = Meta(tool, ds)
        raw_results = meta.load_latest_results()
        raw_results = remove_percentile_outliers_from_group(raw_results, dimensions_for_grouping=[*groupby, 'epsilon'], metric='result', lower=0.05, upper=0.05)


        # Remove outliers
        raw_results = remove_percentile_outliers_from_group(
            raw_results, 
            dimensions_for_grouping=[*groupby, *['epsilon']], 
            lower=0.05, upper=0.05)

        priv = raw_results[raw_results['epsilon'] != -1]
        non_priv = raw_results[raw_results['epsilon'] == -1]


        non_priv = non_priv.groupby(['epsilon', *groupby]).aggregate({
            'time': 'mean'}).reset_index()


        _results = priv.merge(non_priv, on=groupby)
        _results['time'] = (_results['time_x'] - _results['time_y']) / _results['time_y'] * 100
        _results = _results.rename(columns={'epsilon_x': 'epsilon'})[[*groupby, 'epsilon', 'time']]
        _results['tool'] = tool



        results = results.append(_results, ignore_index=True)


    return results

def synthetic_data(ds):
    results = []

    for tool in ['gretel', 'smartnoise_dpctgan', 'smartnoise_patectgan', 'smartnoise_mwem', 'smartnoise_synthetic']:
        meta = Meta(tool, ds, load_data=False)

    # dataset_size,query,epsilon,time,tool
    for sdd in meta.best_performing_synthetic_datasets():
        config = meta.load_synthetic_configuration_with_fallback(sdd)
        data = {
            'dataset_size': config['dataset_size'],
            'epsilon': config['epsilon'],
            'time': config['generate_time'],
            'tool': config['tool'],
        }
        
        results.append(data)
        
    return pd.DataFrame(results)


def times(ds, cache=False):
    meta = Meta(None, ds, load_data=False)

    if cache:
        log('info', 'loading all aggregated results data from cache')
        try:
            return meta.load_aggregated_time()
        except:
            log('warn', 'didn\'t find any cache defaulting to load all data')

    data = pd.DataFrame()
    data = data.append(synthetic_data(ds), ignore_index=True)
    data = data.append(interactive_data(ds), ignore_index=True)

    if data.shape[0] > 0:
        meta.save_aggregated_time(data)
    return data
