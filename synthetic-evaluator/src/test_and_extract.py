import os
from dpevaluation.synthetic.statistical_extractor import dataset_scores
from dpevaluation.synthetic.statistical_tester import test_dataset


if __name__ == '__main__':
    if 'DATASET' in os.environ:
        ds = os.environ['DATASET']
        test_dataset(ds)
        dataset_scores(ds)
