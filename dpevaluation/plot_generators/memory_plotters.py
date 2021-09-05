from dpevaluation.statistical_queries.extractor import relative_result
from dpevaluation.plot_builders.line_plot_builder import plot_line
from dpevaluation.aggregators.time_aggregator import times
from dpevaluation.aggregators.memory_aggregator import all
from dpevaluation.aggregators.results_aggregator import aggregated_results
from dpevaluation.utils.log import log
from dpevaluation.plot_builders.violin_plot_builder import plot_violin
from dpevaluation.plot_builders.contour_plot_builder import plot_contour
import os
from dpevaluation.utils.meta import Meta
import numpy as np
import pandas as pd

from dpevaluation.plot_generators.utils import epsilon_dim, name_epsilon_dataset, name_tool_dataset, result_dim, dataset_size_dim,query_col_to_query_family_col, tool_dim, time_dim, add_domain_col, memory_dim_mem, memory_percentage_dim, ticks_5499, ticks_9358, ticks_2000

# note that this function generate all mem plots
def memory_plots():
    data = all(cache=True)
    # to mb
    data['memory'] = data['memory'] * ((1/1024)**2)
    data = query_col_to_query_family_col(data)

    # just temp meta
    meta = Meta(None, 'medical-survey', load_data=False)
    data = add_domain_col(data, meta)
    # overwrite with none so we dont accidentilly reuse it
    meta = None

    groupby = ['dataset', 'tool', 'dataset_size', 'container_name', 'domain', 'query']

    aggregated_by_max = data \
                .groupby([*groupby, 'epsilon'], dropna=False) \
                .max() \
                .reset_index() \

    __synthetics(aggregated_by_max)
    quit()


    priv = aggregated_by_max[aggregated_by_max['epsilon'] != -1]
    non_priv = aggregated_by_max[aggregated_by_max['epsilon'] == -1]

    _results = priv.merge(non_priv, on=groupby)
    _results['memory'] = (_results['memory_x'] - _results['memory_y']) / _results['memory_y'] * 100
    _results = _results.rename(columns={'epsilon_x': 'epsilon'})[[*groupby, 'epsilon', 'memory']]

    __contour_by_container(_results)
    __misc_memory_plots(_results)

def __synthetics(data):
    data = data[data['epsilon'] != -1]

    for ds, data_for_dataset in data.groupby(['dataset']):
        synth_tools = ['smartnoise_dpctgan', 'smartnoise_patectgan', 'smartnoise_mwem']
        synth_data = data_for_dataset[data_for_dataset['tool'].isin(synth_tools)]

        for (tool), by_tool in synth_data.groupby(['tool']):
            plot_contour(
                by_tool,
                epsilon_dim,
                dataset_size_dim,
                memory_dim_mem,
                name_tool_dataset(ds, tool),
            )

        epsilons = [3, 2, 1, 0.1]

        y_ticks = ticks_2000
        if ds == 'parkinson':
            x_ticks = ticks_5499
        else:
            x_ticks = ticks_9358


        for epsilon in epsilons:
            by_epsilon = synth_data[synth_data['epsilon'] == epsilon]

            plot_line(
                by_epsilon,
                dataset_size_dim,
                memory_dim_mem,
                tool_dim,
                name_epsilon_dataset(ds, epsilon),
                y_ticks=y_ticks,
                x_ticks=x_ticks
            )
        
        gretel_data = data_for_dataset[data_for_dataset['tool'] == 'gretel']
        
        plot_line(
                gretel_data,
                dataset_size_dim,
                memory_dim_mem,
                tool_dim,
                name_tool_dataset(ds, 'gretel'),
                y_ticks=y_ticks,
                x_ticks=x_ticks
            )

def __mem_violin_plots(data):
    for ds, ds_data in data.groupby(['dataset']):
        meta = Meta(None, ds, load_data=False)


        for domain, domain_data in ds_data.groupby(['domain']):
            for tool, _data in domain_data.groupby(['tool']):

                if domain == 'statistical_queries':
                    for (container_name, query, dataset_size), __data in _data.groupby(['container_name', 'query', 'dataset_size']):
                        plot_violin(
                            __data,
                            epsilon_dim,
                            memory_dim_mem,
                            f'mem-{tool}-{query}-{container_name}-size-{dataset_size}-{ds}',
                            meta.plots_path(domain),
                        )


                if domain == 'machine_learning':
                    for (container_name, dataset_size), __data in _data.groupby(['container_name', 'dataset_size']):
                        plot_violin(
                            __data,
                            epsilon_dim,
                            memory_dim_mem,
                            f'mem-{tool}-{container_name}-size-{dataset_size}-{ds}',
                            meta.plots_path(domain),
                        )

def __misc_memory_plots(data):

    for ds, ds_data in data.groupby(['dataset']):
        meta = Meta(None, ds, load_data=False)
        
        for domain, domain_data in ds_data.groupby(['domain']):
            if domain == "machine_learning":

                # for dataset_size, _data in domain_data.groupby(['dataset_size']):
                # for (size, tool, container), _data in domain_data.groupby(['dataset_size', 'tool', "container_name"]): # SQ
                for dataset_size, _data in domain_data.groupby(['dataset_size']): # ML
                    plot_line(
                        _data,
                        epsilon_dim,
                        memory_dim_mem,
                        tool_dim, 
                        # f'mem-{domain}-tools-size-{dataset_size}-{ds}',
                        f'{ds.replace("-", " ").replace("medical", "health").title()}, size={dataset_size}',
                        # f'{container.replace("-evaluation", "").replace("-postgres", ", Postgres").replace("-", " ").replace("_", " ").title().replace("Dp", "DP")}, {ds.replace("-", " ").replace("medical", "health").title()}, size={size}',
                        meta.plots_path(domain)
                    )



    

def __contour_by_container(data):

    for (container_name, ds, domain), _data in data.groupby(['container_name', 'dataset', 'domain']):
        if domain == "machine_learning":
            meta = Meta(None, ds, load_data=False)
            _data = _data.groupby(['epsilon', 'dataset_size']) \
                .min() \
                .reset_index() \


            # print(_data[_data['memory'] < 0])
            # quit()
            
            # print(_data['container_name'])
            plot_contour(_data, 
                epsilon_dim, 
                dataset_size_dim,
                memory_percentage_dim, 
                # f'mem-{container_name}-contour-{ds}'
                f'{container_name.replace("-evaluation", "").replace("-postgres", ", Postgres").replace("-", " ").replace("_", " ").title().replace("Dp", "DP")}, {ds.replace("-", " ").replace("medical", "health").title()}', 
                meta.plots_path(domain))

