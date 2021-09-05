from dpevaluation.utils.log import log
import os
from dpevaluation.utils.meta import Meta
from dpevaluation.statistical_queries.sq_utils import print_progress_extended

ds = 'medical-survey'
meta = Meta(None, ds)
synthetic_data_dirs = meta.synthetic_datasets()

for i, sdd in enumerate(synthetic_data_dirs):
    print_progress_extended(i, len(synthetic_data_dirs))
    path = meta._synthetic_configuration_path(sdd)
    if not os.path.exists(path):
        log('warn', f'{path} does not exist')
        continue

    dataset_size = int(meta.load_synthetic_data(sdd).shape[0])

    log('debug', path)
    with open(path, 'r+') as f:
        new_file = []
        ignore = False
        for l in f.readlines():
            if l.startswith('dataset_size'):
                ignore = True
            elif l.startswith('epochs'):
                ignore = False
                new_file.append(f'dataset_size: {dataset_size}\n')

            if not ignore:
                new_file.append(l)
        f.seek(0)
        f.write(''.join(new_file))
        f.truncate()



