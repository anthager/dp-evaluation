from dpevaluation.statistical_queries.sq_utils import print_progress_extended
from dpevaluation.runner import run_tests
from dpevaluation.utils.meta import Meta
from dpevaluation.queries import PANDAS_QUERIES

# loops over all the pandas queries and all the columns and creates a lambda for each combination of query and
# column with both in the closure. The lambda takes a dataframe and calls the query with the dataframe and the
# column in closure. In practice this adds the column name to the query closure
def query_closures(meta, private_columns):
    columns = [c for c in private_columns if c in meta.column_names and c not in meta.private_column_names]
    histogram_column_names = [c['name'] for c in meta.histogram_columns]

    queries = []
    for name, query in PANDAS_QUERIES.items():
        for col in columns:
            # don't try to make histograms out of categorical columns
            if name == 'histogram' and col not in histogram_column_names:
                continue
            queries.append(lambda _df, query=query, col=col: query(_df, col))

    return queries

def test(ds, private_data):
    metadata = Meta(tool=None, name=ds)

    private_config = metadata.load_synthetic_configuration_with_fallback(private_data)
    private_df = metadata.load_synthetic_data(private_data)

    def run_query(meta: Meta, config):
        query = config['query']
        epsilon = config['epsilon']
        dataset_size = config['dataset_size']

        if epsilon == '-1':
            res = query(meta.parsed_dataset.head(dataset_size))
        else:
            res = query(private_df)
            res['tool'] = private_config['tool']

        return list(res.columns), res.to_numpy()

    configuration = {
        'epsilon': ['-1', private_config['epsilon']],
        'dataset_size': [private_df.shape[0]],
        'number_of_runs': [1],
        'query': query_closures(metadata, private_df.columns)
    }

    results, _ = run_tests(run_query, metadata, configuration, merge_result=True)
    metadata.save_synthetic_SQ_result(results, private_data)

    return results


def test_dataset(ds):
    # needs to use a sq tool in order to use the without values method, have a look in it to see why
    meta = Meta('google_dp', ds, load_data=False)
    synthetic_data_dirs = meta.best_performing_synthetic_datasets()
    
    for i, sdd in enumerate(synthetic_data_dirs):
        test(ds, sdd)
        print_progress_extended(i, len(synthetic_data_dirs))

