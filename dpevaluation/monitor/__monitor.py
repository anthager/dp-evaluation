import os
import time
import subprocess
import json

import pandas as pd
import time
from dpevaluation.utils.meta import Meta
from dpevaluation.utils.log import log
from dpevaluation.utils.timestamp import timestamp
import signal


def main():
    if not 'CONTAINER_NAMES' in os.environ:
        raise Exception(
            'you need to provide an comma seperated list of container names to monitor in the "CONTAINER_NAMES" env')

    results = []
    tool = os.environ['TOOL']
    dataset = os.environ['DATASET']
    additional_name = os.environ.get('ADDITIONAL_NAME')
    containers = [{'name': name}
                  for name in os.environ['CONTAINER_NAMES'].split(',')]
    meta = Meta(tool, dataset, load_data=False)

    # we don't care about the args
    def handle_sigint(_a1, _a2):
        meta.save_monitoring_data(pd.DataFrame(results), additional_name)
        exit(0)

    # sigint is sent on ctrl + c of it felt natural to use for stopping
    signal.signal(signal.SIGINT, handle_sigint)

    for container in containers:
        process = subprocess.Popen(
            ['curl', '--no-buffer', '-s', '--unix-socket', '/var/run/docker.sock', 'http://localhost/containers/{}/stats'.format(container["name"])], stdout=subprocess.PIPE, universal_newlines=True)
        container['process'] = process

    while True:
        for container in containers:
            process = container['process']
            output = process.stdout.readline().strip()
            if output is not None:
                res = {}
                parsed_output = None

                try:
                    parsed_output = json.loads(output)

                    # this is how docker stats do it
                    res['memory'] = parsed_output['memory_stats']['usage'] - \
                        parsed_output['memory_stats']['stats']['cache']

                    res['docker_timestamp'] = parsed_output['read']
                    res['process_timestamp'] = timestamp()
                    res['container_name'] = container['name']

                    results.append(res)
                except:
                    log('debug', "{} is not up yet".format(container['name']))
                    continue

        time.sleep(0.1)


if __name__ == "__main__":
    main()