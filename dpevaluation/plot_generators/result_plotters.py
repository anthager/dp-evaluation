import numpy as np
from dpevaluation.plot_builders.line_plot_builder import plot_line
from dpevaluation.aggregators.results_aggregator import aggregated_results
from dpevaluation.plot_builders.contour_plot_builder import plot_contour
from dpevaluation.utils.meta import Meta

from dpevaluation.plot_generators.utils import clip_dataframe, epsilon_dim, name_tool_dataset, name_epsilon_dataset, name_query_tool_dataset, name_query_epsilon_dataset, result_dim, dataset_size_dim,query_col_to_query_family_col, tool_dim, query_dim, ticks_200, ticks_5499, ticks_9358, ticks_3000

import pandas as pd

def result_plots(ds):
    data = aggregated_results(ds, cache=True)

    # __ml_result_plots(ds, data)
    # __sq_result_plots(ds, data)
    # __domain_line_plots(ds, data)
    __synth_SQ_plots(ds, data)
    # __synth_ML_plots(ds, data)


def __synth_SQ_plots(ds, data: pd.DataFrame):
    data = query_col_to_query_family_col(data)

    synth_tools = ['smartnoise_dpctgan', 'smartnoise_patectgan', 'smartnoise_mwem']

    synth_data = data[data['tool'].isin(synth_tools) & ((data['query'] == 'histogram') | (data['query'] == 'avg'))] \
        .groupby(['query', 'tool', 'dataset_size', 'epsilon']) \
        .mean() \
        .reset_index()
        

    epsilons = [3, 2, 1, 0.1]
    queries = ['avg', 'histogram']

    for (tool, query), _data in synth_data.groupby(['tool', 'query']):
        plot_contour(
            _data,
            epsilon_dim,
            dataset_size_dim,
            result_dim,
            name_query_tool_dataset(ds, tool, query),
        )

    for query in queries:
        _data = synth_data[synth_data['query'] == query]
        for epsilon in epsilons:
            __data = _data[synth_data['epsilon'] == epsilon]


            y_ticks = ticks_200
            if ds == 'parkinson':
                x_ticks = ticks_5499
            else:
                x_ticks = ticks_9358
                if query == 'histogram':
                    y_ticks = ticks_3000

            plot_line(
                __data,
                dataset_size_dim,
                result_dim,
                tool_dim,
                name_query_epsilon_dataset(ds, epsilon, query),
                x_ticks,
                y_ticks,
            )

    gretel_data = data[(data['tool'] == 'gretel') & ((data['query'] == 'histogram') | (data['query'] == 'avg'))] \
        .groupby(['query', 'tool', 'dataset_size', 'epsilon']) \
        .mean() \
        .reset_index()

    
    for query, _data in gretel_data.groupby(['query']):

        y_ticks = ticks_200
        if ds == 'parkinson':
            x_ticks = ticks_5499
        else:
            x_ticks = ticks_9358
            if query == 'histogram':
                y_ticks = ticks_3000

        plot_line(
                _data,
                dataset_size_dim,
                result_dim,
                tool_dim,
                name_query_tool_dataset(ds, 'gretel', query),
                x_ticks,
                y_ticks,
        )


def __synth_ML_plots(ds, data: pd.DataFrame):
    synth_tools = ['smartnoise_dpctgan', 'smartnoise_patectgan', 'smartnoise_mwem']

    synth_data = data[data['tool'].isin(synth_tools) & (data['query'].isnull()) & (data['evaluation_tool'] == 'tensorflow_privacy')]


    epsilons = [3, 2, 1, 0.1]

    for (tool), _data in synth_data.groupby(['tool']):
        plot_contour(
            _data,
            epsilon_dim,
            dataset_size_dim,
            result_dim,
            name_tool_dataset(ds, tool),
        )

    for epsilon in epsilons:
        _data = synth_data[synth_data['epsilon'] == epsilon]


        if ds == 'parkinson':
            x_ticks = ticks_5499
            y_ticks = ticks_3000
        else:
            x_ticks = ticks_9358
            y_ticks = ticks_200

        plot_line(
            _data,
            dataset_size_dim,
            result_dim,
            tool_dim,
            name_epsilon_dataset(ds, epsilon),
            x_ticks,
            y_ticks,
        )

    if ds == 'medical-survey':
        gretel_data = data[data['tool'].isin(['gretel']) & (data['query'].isnull())]
        
        x_ticks = ticks_9358
        y_ticks = ticks_200

        plot_line(
                gretel_data,
                dataset_size_dim,
                result_dim,
                tool_dim,
                name_tool_dataset(ds, 'gretel'),
                x_ticks,
                y_ticks,
        )

    
def __ml_result_plots(ds, data):
    domain = 'machine_learning'
    meta = Meta(None, ds, load_data=False)
    
    # machine_learning_tools = [name for name, _domain in meta._domains.items() if _domain == domain]

    # interactive_data = data[data['tool'].isin(machine_learning_tools)]
    # interactive_data = interactive_data.groupby(['dataset_size', 'epsilon', 'tool']).mean().reset_index()

    synthetic_data = data[data['evaluation_tool'] == 'tensorflow_privacy']


    for (tool), _data in synthetic_data.groupby(['tool']):
        # _data = clip_dataframe(_data, 'result', upper=.95)
        plot_contour(
            _data,
            epsilon_dim,
            dataset_size_dim,
            result_dim,
            f'{tool.replace("_", " ").title().replace("Dp", "DP")}, {ds.replace("-", " ").replace("medical", "health").title()}',
            meta.plots_path(domain),
        )

    epsilons = [3, 2, 1, 0.1]

    for epsilon in epsilons:
        _data = synthetic_data[synthetic_data['epsilon'] == epsilon]

        if epsilon == 3:
            _data = _data.append(synthetic_data[synthetic_data['tool'] == 'gretel'], ignore_index=True)

        plot_line(_data, dataset_size_dim, result_dim, tool_dim, f'Epsilon={epsilon}, {ds.replace("-", " ").replace("medical", "health").title()}',)

    # for tool, _data in interactive_data.groupby(['tool']):
    #     # _data = clip_dataframe(_data, 'result', upper=.95)
    #     plot_contour(
    #         _data,
    #         epsilon_dim,
    #         dataset_size_dim,
    #         result_dim,
    #         f'{tool.replace("_", " ").title()}, {ds.replace("-", " ").replace("medical", "health").title()}',
    #         # f'result-{tool}-{ds}',
    #         meta.plots_path(domain)
    #     )



def __sq_result_plots(ds, data: pd.DataFrame): 
    domain = 'statistical_queries'
    meta = Meta(None, ds, load_data=False)

    data = data.dropna(subset=['query']).copy()

    data = query_col_to_query_family_col(data)
    data = data.groupby(['query', 'dataset_size', 'epsilon', 'tool']).mean().reset_index()

    synthetic_tools = [name for name, _domain in meta._domains.items() if _domain == 'synthetic_data']

    for (tool, query), _data in data.groupby(['tool', 'query']):
        # _data = clip_dataframe(_data, 'result', upper=.95)
        if _data.shape[0] == 0:
            continue

        # since we always make sure to generate the same number of lines to the synth data as the original dataset, 
        # counting rows is pointless since they will be the same
        # if query == "count" and tool in synthetic_tools:
        if tool in synthetic_tools:

            plot_contour(
                _data,
                epsilon_dim,
                dataset_size_dim,
                result_dim,
                # f'result-{tool}-{query}-{ds}',
                f'{query.upper().replace("HISTOGRAM", "HIST")}, {tool.replace("_", " ").title().replace("Dp", "DP")}, {ds.replace("-", " ").replace("medical", "health").title()}',
                meta.plots_path(domain)
            )


def __domain_line_plots(ds, data: pd.DataFrame):
    meta = Meta(None, ds, load_data=False)


    # ml_tools = [name for name, _domain in meta._domains.items() if _domain == 'machine_learning']
    # data_ml = data[data['tool'].isin(ml_tools)]
    # for dataset_size, _data in data_ml.groupby(['dataset_size']):
    #     plot_line(
    #         _data,
    #         epsilon_dim,
    #         result_dim,
    #         tool_dim,
    #         # f'result-ml-tools-{dataset_size}-size-{ds}',
    #         f'{ds.replace("-", " ").replace("medical", "health").title()}, size={int(dataset_size)}',
    #         meta.plots_path('machine_learning'),
    #         ymax=30,
    #     )


    # generate plots for only the interactive sq tools to show how they do between themselfs, that is hard to 
    # see when synth are included
    # sq_tools = [name for name, _domain in meta._domains.items() if _domain == 'statistical_queries']
    # data_sq = data[data['tool'].isin(sq_tools)]
    # data_sq = query_col_to_query_family_col(data_sq)
    # # for (dataset_size, query), _data in data_sq.groupby(['dataset_size', 'query']):
    # for (dataset_size, tool), _data in data_sq.groupby(['dataset_size', 'tool']):
    #     ymax = 10 if ds == 'medical-survey' else 25
    #     plot_line(
    #         _data,
    #         epsilon_dim,
    #         result_dim,
    #         tool_dim,
    #         # f'result-sq-tools-{query}-size-{int(dataset_size)}-{ds}',
    #         # f'{query.upper().replace("HISTOGRAM", "HIST")}, {ds.replace("-", " ").replace("medical", "health").title()}, size={int(dataset_size)}',
    #         f'{tool.replace("_", " ").title().replace("Dp", "DP")}, {ds.replace("-", " ").replace("medical", "health").title()}, size={int(dataset_size)}',
    #         meta.plots_path('statistical_queries'),
    #         ymax=ymax,
    #     )

    # both interactive and synth sq
    data_sq_and_synth = data.dropna(subset=['query'])
    data_sq_and_synth = data_sq_and_synth[data_sq_and_synth['tool'] != 'gretel'].copy()
    synthetic_tools = [name for name, _domain in meta._domains.items() if _domain == 'synthetic_data']
    data_sq_and_synth = data_sq_and_synth[data_sq_and_synth['tool'].isin(synthetic_tools)]

    data_sq_and_synth = query_col_to_query_family_col(data_sq_and_synth)
    # for (dataset_size, query), _data in data_sq_and_synth.groupby(['dataset_size', 'query']):
    for (dataset_size, tool), _data in data_sq_and_synth.groupby(['dataset_size', 'tool']):
        plot_line(
            _data,
            epsilon_dim,
            result_dim,
            tool_dim,
            # f'result-sq-tools-inc-synth-{query}-size-{dataset_size}-{ds}',
            f'{tool.replace("_", " ").title()}, {ds.replace("-", " ").replace("medical", "health").title()}, size={int(dataset_size)}',
            meta.plots_path('statistical_queries'),
            ymax=100,
        )


    # synthetic_tools = [name for name, _domain in meta._domains.items() if _domain == 'synthetic_data']
    # data_sq = data[data['tool'].isin(synthetic_tools)]
    # data_sq['query'] = data_sq['query'].apply(lambda row: row.split('_')[0])
    # for (dataset_size, query), _data in data_sq.groupby(['dataset_size', 'query']):
    #     plot_line(_data, epsilon_dim, result_dim, tool_dim, f'result-sq-tools-{query}-size-{dataset_size}-{ds}', meta.plots_path('synthetic_data'), ymax=10)
        
