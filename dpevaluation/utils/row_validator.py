import pandas as pd
import numpy as np
from functools import reduce


def rows_validator(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    '''
    Okay this is a handful. Given a dataframe and an array of conditions with name, upper and lower
    returns only the rows that are between the lower and upper bound (both inclusive).
    This is done by generating a series containing boolean values for each row for each of the columns.
    When we have a series for with the shape (,<rows>) for each of the columns, these are converted
    to numpy arrays of ints and reduced togheter to a single (,<rows>) vector. If a row in this
    vector has the value of len(columns) it fullfills all conditions and should be included in the 
    resulting dataframe
    '''
    column_conditions = [((df[column['name']] <= column['upper']) & (
        df[column['name']] >= column['lower'])).to_numpy(np.int) for column in columns if column['name'] in df.columns]
    conditions = np.array(list(map(lambda num: num == len(
        column_conditions), reduce(lambda a, c: a + c, column_conditions))))
    return df[conditions]

