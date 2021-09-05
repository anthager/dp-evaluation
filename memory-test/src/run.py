from dpevaluation.utils.meta import Meta
import subprocess
from dpevaluation.monitor.monitor import Monitor
from dpevaluation.utils.change_metadata import change_metadata

import os

monitor_process = None


def start_evaluation(tool, ds, host_repo_path, _tool=''):
    path = 'src/{}_memory_test.sh'.format(tool)
    subprocess.run([path], env={'DATASET': ds,
                                'HOST_REPO_PATH': host_repo_path,
                                'TOOL': _tool,
                                })


def main(tool, ds, host_repo_path):
    meta = Meta(None, ds, load_data=False)

    epsilons = meta.epsilons
    dataset_sizes = meta.dataset_sizes
    queries =  meta.queries if tool != 'smartnoise_synthetic' and meta._domains[tool] == 'synthetic_queries' else ['']
    # the algorithms in smartnoise synth is called "tools"
    _tools =  ['smartnoise_patectgan', 'smartnoise_dpctgan'] if tool == 'smartnoise_synthetic' else ['']
    monitor = Monitor(tool, ds)

    for epsilon in epsilons:
        for dataset_size in dataset_sizes:
            for query in queries:
                for _tool in _tools:
                    change_metadata(ds, epsilon, dataset_size, query)

                    if query != '' and _tool != '':
                        raise Exception('both query and _tool should not be defined at the same time')
                    additional_name = ''
                    if query != '':
                        additional_name = query
                    if _tool != '':
                        additional_name = _tool

                    monitor.start(additional_name)
                    start_evaluation(tool, ds, host_repo_path, _tool)
                    monitor.stop()


if __name__ == "__main__":
    datasets = os.environ['DATASET'].split(',')
    tools=os.environ['TOOL'].split(',')
    # to make sure our volumes work. At this point the lengths we go for protability approaches self harm
    host_repo_path = os.environ['HOST_REPO_PATH']

    # multiple tests can be run after eachother
    for ds in datasets:
        for tool in tools:
            main(tool, ds, host_repo_path)

