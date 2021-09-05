from dpevaluation.utils.meta import Meta
import pandas as pd

path = '/dp-tools-evaluation/results/synthetic_data/turbo/smartnoise_patectgan_10_2021-05-26T07:30:36/dataset.csv'
df = pd.read_csv(path)

df = df.round(0)

df.to_csv(path, index=False)