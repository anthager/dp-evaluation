import pandas as pd

def __remove_percentile_outliers(data: pd.DataFrame, col, lower=0, upper=0) -> pd.DataFrame:
    lower_values_to_remove = round(data.shape[0] * lower)
    upper_values_to_remove = round(data.shape[0] * upper)

    data = data.sort_values(col)
    data = data.head(data.shape[0] - upper_values_to_remove)
    data = data.tail(data.shape[0] - lower_values_to_remove)

    return data


def remove_percentile_outliers_from_group(data: pd.DataFrame, dimensions_for_grouping=['dataset_size', 'epsilon', 'query'], metric='result', lower=0, upper=0) -> pd.DataFrame: 
    '''
    no order will be perserved
    '''
    new_data = pd.DataFrame()

    for _, _data in data.groupby(dimensions_for_grouping, dropna=False):
        new_data = new_data.append(__remove_percentile_outliers(_data, col=metric, lower=lower, upper=upper))

    return new_data