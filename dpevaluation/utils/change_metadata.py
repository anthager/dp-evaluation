from dpevaluation.utils.meta import Meta


def change_metadata(ds, epsilon, dataset_size, allowlisted_query):
    meta = Meta(None, ds, load_data=False)
    _meta = meta._meta
    _meta['params']['epsilons'] = [float(epsilon)]
    _meta['params']['dataset_sizes'] = [int(dataset_size)]
    _meta['statistical_queries']['allowlisted_queries'] = [allowlisted_query]
    meta.save_metadata(_meta)
