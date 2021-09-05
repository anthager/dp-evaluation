import pandas as pd
import re

epsilon_dim = {'name': 'Epsilon', 'dim': 'epsilon'}
dataset_size_dim = {'name': 'Dataset size', 'dim': 'dataset_size'}

# time_dim = {'name': "Time (% from benchmark)", 'dim': 'time'}
# time_dim = {'name': "Avg. RMSPE (all ε & data size)", 'dim': 'time'} # SQ comparison tools
time_dim = {'name': "Time (s)", 'dim': 'time'}

# result_dim = {'name': 'Result (% from benchmark)', 'dim': 'result'}
result_dim = {'name': 'RMSPE', 'dim': 'result'}

memory_dim_mem = {'name': 'Memory (MB)', 'dim': 'memory'}
memory_dim_delta = {'name': 'delta (δ)', 'dim': 'memory'}
memory_percentage_dim = {'name': 'delta (δ)', 'dim': 'memory'}

tool_dim = {'name': 'Tool', 'dim': 'tool'}
container_name_dim = {'name': 'Container name', 'dim': 'container_name'}

query_dim = {'name': 'Query', 'dim': 'query'}

def clip_dataframe(data: pd.DataFrame, col, lower=0, upper=1) -> pd.DataFrame:
    upper = float(data[col].quantile(upper))
    lower = float(data[col].quantile(lower))
    data[col] = data[col].clip(lower, upper)

    return data

def add_domain_col(data, meta):
    data['domain'] = data['tool'].apply(lambda tool: meta._domains[tool])
    return data

def query_col_to_query_family_col(data) -> pd.DataFrame:
    # works even if row doesnt contain a query
    if not 'query' in data:
        return data
    data['query'] = data['query'].apply(lambda row: row.split('_')[0] if type(row) == str else row)
    return data

def prettify_name(ugly_name):
    return re.sub(r'-|_', ' ', ugly_name).title()

def prettify_ds_name(ugly_name):
    return ugly_name.replace("-", " ").replace("medical", "health").title()

def prettify_tool_name(ugly_name):
    synth_names = {'gretel': 'Gretel', 'smartnoise_dpctgan': 'Smartnoise DPCTGAN', 'smartnoise_patectgan': 'Smartnoise PATECTGAN', 'smartnoise_mwem': 'Smartnoise MWEM'}
    return synth_names[ugly_name]

def prettify_query_name(ugly_name):
    return ugly_name.upper().replace("HISTOGRAM", "HIST")


# sq
def name_query_tool_dataset(ds, tool, query):
    pretty_query = prettify_query_name(query)
    pretty_ds = prettify_ds_name(ds)
    pretty_tool = prettify_tool_name(tool)

    return f'{pretty_query}, {pretty_tool}, {pretty_ds}'

def name_query_epsilon_dataset(ds, epsilon, query):
    pretty_query = prettify_query_name(query)
    pretty_ds = prettify_ds_name(ds)

    return f'{pretty_query}, ε={str(epsilon)}, {pretty_ds}'

def name_tool_dataset(ds, tool):
    pretty_ds = prettify_ds_name(ds)
    pretty_tool = prettify_tool_name(tool)

    return f'{pretty_tool}, {pretty_ds}'

def name_epsilon_dataset(ds, epsilon):
    pretty_ds = prettify_ds_name(ds)

    return f'ε={str(epsilon)}, {pretty_ds}'


ticks_100 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ticks_200 = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
ticks_1000 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
ticks_2000 = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
ticks_3000 = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]

# used for medical-survey
ticks_9358 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9358]
# used for parkinson
ticks_5499 = [1000, 2000, 3000, 4000, 5000, 5499]

