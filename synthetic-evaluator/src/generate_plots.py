from dpevaluation.plot_generators.memory_plotters import memory_plots
from dpevaluation.plot_generators.time_plotters import time_plots
from dpevaluation.plot_generators.result_plotters import result_plots

import os


if __name__ == '__main__':
    ds = "medical-survey"#os.environ['DATASET']
    # result_plots(ds)
    # time_plots(ds)
    memory_plots()

    ds = "parkinson"
    # result_plots(ds)
    # time_plots(ds)
    memory_plots()
