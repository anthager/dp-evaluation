import pandas as pd
import numpy as np


dfs = pd.read_csv('data.txt').groupby(
    'age_group').apply(lambda x: len(x) / 1000)
print(dfs)
