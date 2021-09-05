import re
import pandas as pd

QUIERIES = {
    'histogram' :'''
        SELECT COUNT({column_name}) as result, \'histogram_{column_name}\' AS query, {column_name} AS result_bin
        FROM data_%s
        GROUP BY {column_name}
        ORDER BY result_bin
    ''',
    'sum': "SELECT SUM({column_name}) AS result, 'sum_{column_name}' AS query FROM public.data_%s WHERE {column_name} != -1",
    'count': "SELECT COUNT({column_name}) AS result, 'count_{column_name}' AS query FROM public.data_%s WHERE {column_name} != -1",
    'avg': "SELECT AVG({column_name}) AS result, 'avg_{column_name}' AS query FROM public.data_%s WHERE {column_name} != -1",
}

def histogram(df: pd.DataFrame, column_name):
    res = df.groupby([column_name]).aggregate('size').to_frame('result').reset_index().rename(columns={column_name: 'result_bin'})
    res['query'] = f'histogram_{column_name}'
    return res

def _sum(df: pd.DataFrame, column_name):
    res = df.aggregate({column_name: 'sum'}).to_frame('result')
    res['query'] = f'sum_{column_name}'
    return res

def count(df: pd.DataFrame, column_name):
    res = df.aggregate({column_name: 'size'}).to_frame('result')
    res['query'] = f'count_{column_name}'
    return res

def avg(df: pd.DataFrame, column_name):
    res = df.aggregate({column_name: 'mean'}).to_frame('result')
    res['query'] = f'avg_{column_name}'
    return res

# while it works to use the pandas sql package for running the raw sql queries they are magnitudes slower than native pandas.
# 3s vs 3ms on my macbook
PANDAS_QUERIES = {
    'histogram': histogram,
    'sum': _sum,
    'count': count,
    'avg': avg,
}


# stolen from pandasql. It wasn't exported so it's copy pasted here
def extract_table_names(query):
    """ Extract table names from an SQL query. """
    # a good old fashioned regex. turns out this worked better than actually parsing the code
    tables_blocks = re.findall(
        r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    tables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    return tables


def sanitize_query(query: str) -> str:
    # pandasql dont like dots in the from clause so we just merge schema and table to one str
    # could be nice to do something better with the sql ast
    return query.replace('.', '')
