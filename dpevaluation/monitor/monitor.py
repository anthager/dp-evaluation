import time
import os
import subprocess
import signal


class Monitor:
    __container_names_dict = {
        'smartnoise': 'smartnoise-postgres,smartnoise-evaluation',
        'google_dp': 'google-dp-postgres,google_dp-evaluation',
        'tensorflow_privacy': 'tensorflow_privacy-evaluation',
        'opacus': 'opacus-evaluation',
        'diffprivlib': 'diffprivlib-evaluation',
        'smartnoise_synthetic': 'smartnoise-synthetic-evaluation',
        'gretel': 'gretel-evaluation',
    }
    __monitor_script_path = ABS_PATH = os.path.dirname(
        os.path.abspath(__file__)) + '/__monitor.py'

    __process: subprocess.Popen = None

    def __init__(self, tool, ds):
        self.tool = tool
        self.ds = ds
        self.container_names = self.__container_names_dict[tool]

    # extra data that you want in the file name
    def start(self, additional_name=''):
        self.__process = subprocess.Popen([
            'python', 
            self.__monitor_script_path,
        ], env={
            'TOOL': self.tool,
            'DATASET': self.ds,
            'CONTAINER_NAMES': self.container_names,
            'ADDITIONAL_NAME': additional_name,
            'PATH': os.environ['PATH'],
        })

    def stop(self):
        self.__process.send_signal(signal.SIGINT)
        self.__process.wait()

def main():
    m = Monitor('smartnoise', 'medical-survey')
    m.start()
    time.sleep(10)
    m.stop()

if __name__ == "__main__":
    main()