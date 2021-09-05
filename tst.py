import pandas as pd
from dpevaluation.utils.meta import Meta
import numpy as np
import os
import re
# base =... '/Users/antonhagermalm/Projects/dp-tools-evaluation/results/synthetic_data/medical-survey/'
# dirs = os.listdir(base)

# for d in dirs:
#     path = base + d
#     old_result = path + '/results.csv'
#     old_conf = path + '/configuration.yml'

#     new_result = path + '/SQ_results.csv'
#     new_conf = path + '/SQ_configuration.yml'


#     try:
#         os.rename(old_conf, new_conf)
#     except:
#         pass
#     try:
#         os.rename(old_result, new_result)
#     except:
#         pass


# df = pd.DataFrame({'a': [1,2,3,4], 'b': [5,6,7,8], 'c': ['hej', 'cool' ,'hej', 'cool']})
# print(df)
# agg = df.groupby('c').aggregate({'a': 'min'}).reset_index()
# print(agg)


# df = pd.DataFrame({'a': [1,2,3,4], 'b': [5,6,7,8], 'c': ['hej', 'cool' ,'hej', 'cool']})
# df['min_a'] = df.groupby('c').a.transform(np.min)
# print(df[df.min_a == df.a])
# agg = df.groupby('c').aggregate({'a': 'min'}).reset_index()
# print(agg)

# synth_df = pd.read_csv('/Users/antonhagermalm/Projects/dp-tools-evaluation/results/synthetic_data/medical-survey/dpctgan_0.5_21-04-28T22:08:26/SQ_results.csv')
# smartnoise_df = pd.read_csv('/Users/antonhagermalm/Projects/dp-tools-evaluation/results/statistical_queries/medical-survey/smartnoise-medical-survey-21-05-03T11:55:58.csv')
# google_dp_df = pd.read_csv('/Users/antonhagermalm/Projects/dp-tools-evaluation/results/statistical_queries/medical-survey/google_dp-medical-survey-21-05-03T12:00:57.csv')
# print(smartnoise_df.groupby('epsilon').mean())
# print(google_dp_df.groupby('epsilon').mean())

# disallowed_histograms = ['histogram_test_time',
#                          'histogram_motor_updrs',
#                          'histogram_total_updrs',
#                          'histogram_jitter_percent',
#                          'histogram_jitter_abs',
#                          'histogram_jitter_rap',
#                          'histogram_jitter_ppq5',
#                          'histogram_jitter_ddp',
#                          'histogram_shimmer',
#                          'histogram_shimmer_db',
#                          'histogram_shimmer_apq3',
#                          'histogram_shimmer_apq5',
#                          'histogram_shimmer_apq11',
#                          'histogram_shimmer_dda',
#                          'histogram_nhr',
#                          'histogram_hnr',
#                          'histogram_rpde',
#                          'histogram_dfa',
#                          'histogram_ppe',
#                          ]

# df = pd.read_csv(
#     '/Users/antonhagermalm/Projects/dp-tools-evaluation/data/parkinson/dataset.csv')
# print('\n'.join(df.columns))


# print(df)
# df.to_csv('/Users/antonhagermalm/Projects/dp-tools-evaluation/data/parkinson/dataset.csv', index=False)




# a = df['a'].idxmin()
# print(df.iloc[a])

# from dpevaluation.utils.meta import Meta
# from dpevaluation.statistical_queries.sq_utils import print_progress_extended

# ds = 'medical-survey'
# meta = Meta(None, ds)
# synthetic_data_dirs = meta.synthetic_datasets()

# for i, sdd in enumerate(synthetic_data_dirs):
#     print_progress_extended(i, len(synthetic_data_dirs))
#     path = f'/Users/antonhagermalm/Projects/dp-tools-evaluation/medical-survey/{sdd}/configuration.yml'
#     with open(path, 'w') as f:
#         print(f)

#     quit









# random_state = 1234

# df1 = pd.DataFrame({'a': [1,2,3], 'b': [1,2,3]})
# df2 = pd.DataFrame({'a': [4,5,6], 'b': [4,5,6]})

# df = df1.append(df2, ignore_index=True)
# size = df.shape[0]

# print(df)

# df = df.sample(size, random_state=random_state)

# print('-------------')
# print(df)

# import os 
# import yaml

# base = '/Users/antonhagermalm/Projects/dp-tools-evaluation/parkinson/'
# for p in os.listdir(base):
#     if  p.startswith('.'):
#         continue

#     config = None
#     with open(base + 'configuration.yml', "r") as f:
#         config = yaml.safe_load_all(f.read())
        
#     df = pd.read_csv(p)

# df1 = pd.DataFrame([{'a': [1,2,3], 'b': 1}])
# df2 = pd.DataFrame({'a': [4,5,6], 'b': [4,5,6]})

# print(df2['a'].to_numpy())




# data = pd.read_csv('/Users/antonhagermalm/Projects/dp-tools-evaluation/results/machine_learning/parkinson/tensorflow_privacy-parkinson-2021-05-18T22:07:41.csv')

# data['time'] = data['elapsed_time']
# data['result'] = data['mse']

# data.to_csv('/Users/antonhagermalm/Projects/dp-tools-evaluation/results/machine_learning/parkinson/tensorflow_privacy-parkinson-2021-05-18T22:07:41.csv', index=False)


# src_base = '/Users/antonhagermalm/Projects/dp-tools-evaluation/medical-survey/'
# dist_base = '/Users/antonhagermalm/Projects/dp-tools-evaluation/results/synthetic_data/medical-survey2/'
# # /Users/antonhagermalm/Projects/dp-tools-evaluation/medical-survey/smartnoise_patectgan_2.0_2021-05-13T16:12:20/opacus_result.csv

# for sds in os.listdir(src_base):
#     src = src_base + sds + '/opacus_result.csv'
#     exists = os.path.exists(src)
#     if not exists:
#         continue
#     dist = dist_base + sds + '/opacus_result.csv'

#     os.rename(src, dist)



# df = pd.read_csv('/Users/antonhagermalm/Projects/dp-tools-evaluation/results/machine_learning/medical-survey/opacus-medical-survey-2021-05-18T15:57:59.csv')
# df = df[['dataset_size', 'epsilon', 'result', 'time']]
# df.to_csv('/Users/antonhagermalm/Projects/dp-tools-evaluation/results/machine_learning/medical-survey/opacus-medical-survey-2021-05-18T15:57:59.csv', index=False)


# print(df['result'].quantile(0.75))


# df = pd.read_csv('./hehe.csv')





# meta = Meta(None, 'turbo')
# sds = meta.synthetic_datasets()

# for sd in sds:
#     path = f'/dp-tools-evaluation/results/synthetic_data/turbo/{sd}/dataset.csv'
#     df = pd.read_csv(path)
#     df = df.round(0)
#     df.to_csv(path, index=False)

# df = pd.DataFrame([
#     {'a': 1, 'b': 1},
#     {'a': 0.2, 'b': 2},
#     {'a': 3, 'b': 3, 'query': 'sum_ahah'},
# ])

# df = df.round({'d': 0})

# # print(df['a'][1].dtype)

# def s(row):
#     return row.split('_')[0] if type(row) == str else row

# def query_col_to_query_family_col(data) -> pd.DataFrame:
#     # works even if row doesnt contain a query
#     if not 'query' in data:
#         return data
#     data['query'] = data['query'].apply(s)
#     return data


# print(query_col_to_query_family_col(df))


# df = pd.DataFrame([
#     {'result': 0, 'dataset_size': 4},
#     {'result': 1, 'dataset_size': 4},
#     {'result': 2, 'dataset_size': 4},
#     {'result': 2, 'dataset_size': 4},
#     {'result': 2, 'dataset_size': 4},
#     {'result': 2, 'dataset_size': 4},
#     {'result': 2, 'dataset_size': 4},
#     {'result': 2, 'dataset_size': 4},
#     {'result': 3, 'dataset_size': 5},
#     {'result': 3, 'dataset_size': 5},
#     {'result': 3, 'dataset_size': 5},
# ])



# import pandas as pd

# def __remove_percentile_outliers(data: pd.DataFrame, col, lower=0, upper=0) -> pd.DataFrame:
#     lower_values_to_remove = round(data.shape[0] * lower)
#     upper_values_to_remove = round(data.shape[0] * upper)

#     data = data.sort_values(col)
#     data = data.head(data.shape[0] - upper_values_to_remove)
#     data = data.tail(data.shape[0] - lower_values_to_remove)

#     return data


# def remove_percentile_outliers_from_group(data: pd.DataFrame, dimensions_for_grouping=['dataset_size', 'epsilon', 'query'], metric='result', lower=0, upper=0) -> pd.DataFrame: 
#     '''
#     no order will be perserved
#     '''
#     new_data = pd.DataFrame()
#     for groups, _data in data.groupby(dimensions_for_grouping):
#         new_data = new_data.append(__remove_percentile_outliers(_data, col=metric, lower=lower, upper=upper))
#     return new_data





# print('-----------------')
# res = remove_percentile_outliers_from_group(df, dimensions_for_grouping=['dataset_size'], metric='result', lower=0.1, upper=0.1)
# print(res)


# def query_col_to_query_family_col(data) -> pd.DataFrame:
#     # works even if row doesnt contain a query
#     if not 'query' in data:
#         return data
#     data['query'] = data['query'].apply(lambda row: row.split('_')[1] if type(row) == str else row)
#     return data

# data = query_col_to_query_family_col(data)

# print(data['query'].unique())


data = pd.read_csv('/Users/antonhagermalm/Projects/dp-tools-evaluation/results/machine_learning/parkinson/tensorflow_privacy-parkinson-2021-05-25T19:00:28.csv')
data = data[['epsilon', 'dataset_size', 'mse', 'result', 'time']]
data.to_csv('/Users/antonhagermalm/Projects/dp-tools-evaluation/results/machine_learning/parkinson/tensorflow_privacy-parkinson-2021-05-25T19:00:28.csv', index=False)






# from sdv.evaluation import evaluate



# evaluate()