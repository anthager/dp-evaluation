import os
from dpevaluation.synthetic.statistical_extractor import all_dataset_scores, dataset_scores
from dpevaluation.synthetic.statistical_tester import test_all, test_dataset
from dpevaluation.utils.meta import Meta

def main(ds, priv_ds): 
  meta = Meta(None, ds)
  meta.reconstruct_synth_dataset(priv_ds)
  



if __name__ == '__main__':
    ds = os.environ['DATASET']
    priv_ds = os.environ['PRIVATE_DATA']
    main(ds, priv_ds)
    