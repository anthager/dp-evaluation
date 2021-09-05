from matplotlib.pyplot import title
from dpevaluation.plot_builders.line_plot_builder import plot_line
from dpevaluation.aggregators.time_aggregator import times

from dpevaluation.plot_builders.violin_plot_builder import plot_violin
from dpevaluation.plot_builders.contour_plot_builder import plot_contour
from dpevaluation.utils.meta import Meta

import pandas as pd
from dpevaluation.plot_generators.utils import clip_dataframe, epsilon_dim, name_epsilon_dataset, name_tool_dataset, prettify_ds_name, prettify_name, result_dim, dataset_size_dim,query_col_to_query_family_col, tool_dim, time_dim, add_domain_col, ticks_5499, ticks_2000, ticks_9358, ticks_1000

def time_plots(ds):
    data = times(ds, cache=True)
    # __ml_time_plots(ds, data)
    # __sq_time_plots(ds, data)
    # __domain_time_line_plots(ds, data)
    # __overall_sq_time(ds, data)
    __synth_time(ds, data)

def __synth_time(ds, data: pd.DataFrame):

    synth_tools = ['smartnoise_dpctgan', 'smartnoise_patectgan', 'smartnoise_mwem']
    synth_data = data[data['tool'].isin(synth_tools)]

    for (tool), _data in synth_data.groupby(['tool']):
        plot_contour(
            _data,
            epsilon_dim,
            dataset_size_dim,
            time_dim,
            name_tool_dataset(ds, tool),
        )

    epsilons = [3, 2, 1, 0.1]

    y_ticks = ticks_2000
    if ds == 'parkinson':
        x_ticks = ticks_5499
    else:
        x_ticks = ticks_9358

    for epsilon in epsilons:
        _data = synth_data[synth_data['epsilon'] == epsilon]
        plot_line(
            _data,
            dataset_size_dim,
            time_dim,
            tool_dim,
            name_epsilon_dataset(ds, epsilon),
            x_ticks,
            y_ticks,
        )

    if ds == 'medical-survey':
        gretel_data = data[data['tool'] == 'gretel']
        plot_line(
                gretel_data,
                dataset_size_dim,
                time_dim,
                tool_dim,
                name_tool_dataset(ds, 'gretel'),
                x_ticks,
                y_ticks,
            )


def __ml_time_plots(ds, data):
    meta = Meta(None, ds)

    domain = 'machine_learning'
    meta = Meta(None, ds, load_data=False)
    
    machine_learning_tools = [name for name, _domain in meta._domains.items() if _domain == domain]

    data = data[data['tool'].isin(machine_learning_tools)]

    data_for_contour = data.groupby(['tool', 'epsilon', 'dataset_size']).mean().reset_index().groupby(['tool'])
    for tool, _data in data_for_contour:
        plot_contour(
            _data,
            epsilon_dim,
            dataset_size_dim,
            time_dim,
            # f'time-{tool}-contour-{ds}',
            f'{tool.replace("_", " ").title()}, {ds.replace("-", " ").replace("medical", "health").title()}',
            meta.plots_path(domain)
        )

    data_for_violin = data.groupby(['tool', 'dataset_size'])
    for (tool, dataset_size), _data in data_for_violin:
        plot_violin(_data, 
        epsilon_dim, 
        time_dim, 
        # f'time-{tool}-{dataset_size}-{ds}', 
        f'time-{tool}-{dataset_size}-{ds.replace("-", " ").replace("medical", "health").title()}',
        meta.plots_path(domain))


def __sq_time_plots(ds, data):
    domain = 'statistical_queries'
    meta = Meta(None, ds)

    data = data.dropna(subset=['query']).copy()

    data = query_col_to_query_family_col(data)


    # violin where epsilon is on the x axis
    # data_by_size = data.groupby(['tool', 'query', 'dataset_size'])
    # for (tool, query, dataset_size), _data in data_by_size:
    #     plot_violin(
    #         _data,
    #         epsilon_dim,
    #         time_dim,
    #         f'time-{tool}-{query}-size-{dataset_size}-{ds}',
    #         meta.plots_path(domain)
    #     )

    # violin where dataset_size is on the x axis
    # data_by_epsilon = data.copy().groupby(['tool', 'query', 'epsilon'])
    # for (tool, query, epsilon), _data in data_by_epsilon:
    #     plot_violin(
    #         _data,
    #         dataset_size_dim,
    #         time_dim,
    #         f'time-{tool}-{query}-epsilon-{epsilon}-{ds}',
    #         meta.plots_path(domain)
    #     )

    contour_data = data \
        .groupby(['tool', 'query', 'dataset_size', 'epsilon']) \
        .median() \
        .reset_index() \
        .groupby(['tool', 'query'])


    for (tool, query), _data in contour_data:
        # _data = clip_dataframe(_data, 'time', upper=.95)
        plot_contour(
            _data,
            epsilon_dim,
            dataset_size_dim,
            time_dim,
            # f'time-{tool}-{query}-contour-{ds}',
            f'{query.upper().replace("HISTOGRAM", "HIST")}, {tool.replace("_", " ").title().replace("Dp", "DP")}, {ds.replace("-", " ").replace("medical", "health").title()}',
            meta.plots_path(domain)
        )

def __domain_time_line_plots(ds, data: pd.DataFrame):
    meta = Meta(None, ds, load_data=False)
    data = add_domain_col(data, meta)
    data = query_col_to_query_family_col(data)

    for domain, domain_data in data.groupby(['domain']):

        if domain == 'machine_learning':
            for dataset_size, _data in domain_data.groupby(['dataset_size']):

                plot_line(
                    _data,
                    epsilon_dim,
                    time_dim,
                    tool_dim,
                    # f'time-{domain}-tools-{dataset_size}-size-{ds}',
                    f'{ds.replace("-", " ").replace("medical", "health").title()}, size={int(dataset_size)}',
                    meta.plots_path(domain),
                )

        # if domain == 'statistical_queries':
        #     # for (dataset_size, query), _data in domain_data.groupby(['dataset_size', 'query']):
        #     for (dataset_size, tool), _data in domain_data.groupby(['dataset_size', 'tool']):
        #         plot_line(
        #             _data,
        #             epsilon_dim,
        #             time_dim,
        #             tool_dim,
        #             # f'time-{domain}-tools-{query}-{dataset_size}-size-{ds}',
        #             f'{tool.replace("_", " ").title().replace("Dp", "DP")}, {ds.replace("-", " ").replace("medical", "health").title()}, size={int(dataset_size)}',
        #             meta.plots_path(domain),
        #         )

def __overall_sq_time(ds, data):
    meta = Meta('google_dp', ds)

    data = query_col_to_query_family_col(data)
    data = data[data['tool'] \
        .isin(['google_dp', 'smartnoise'])] \

    # for (query), _data in data.groupby(['query']):
    _data = data.groupby(['tool', 'dataset_size']).mean()
    plot_line(_data,
              dataset_size_dim,
              time_dim, 
              tool_dim,
              f'{ds.replace("-", " ").replace("medical", "health").title()}',
              meta.plots_path('statistical_queries'))
