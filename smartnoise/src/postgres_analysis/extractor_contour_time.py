from dpevaluation.statistical_analysis.utils import utils
from numpy.lib.arraysetops import unique
from sqlalchemy.sql.expression import label
import sqlalchemy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np


METRICS = ['accuracy']
DIMENSIONS = ['dataset_size', 'epsilon']
QUERIES = ['avg', 'count', 'sum']


def empty_result():
    result = {}
    for query in QUERIES:
        result[query] = {}
        for m in METRICS:
            result[query][m] = []
        for d in DIMENSIONS:
            result[query][d] = []


engine = sqlalchemy.create_engine(utils.POSTGRES_DB)
path = utils.SQL_QUERIES_PATH + '/extractor-contour-time.sql'
text = open(path, 'r').read()


def run():
    query = sqlalchemy.text(text)
    rows = engine.execute(query)

    results = {}
    epsilons_set = set()
    dataset_sizes_set = set()

    for row in rows:
        q = row.query
        epsilons_set.add(row.epsilon)
        dataset_sizes_set.add(row.dataset_size)
        if (not results.get(q)):
            results[q] = {}

        if not results[q].get(row.dataset_size):
            results[q][row.dataset_size] = {}

        results[q][row.dataset_size][row.epsilon] = row.time

    epsilons = np.array(sorted(list(epsilons_set)))
    dataset_sizes = np.array(sorted(list(dataset_sizes_set)))

    times = {}
    for q in QUERIES:
        time = np.zeros((len(dataset_sizes), len(epsilons)))
        for i, ds in enumerate(dataset_sizes):
            for j, e in enumerate(epsilons):
                time[i][j] = results[q].get(ds).get(e)
        times[q] = time

    plt.figure()
    for dv in QUERIES:
        matplotlib.rcParams.update({
            'axes.titlesize': 22,
            'axes.titlepad': 4.0,
            'axes.labelsize': 20,
            'axes.labelpad': 3.0,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18})
        fig, ax = plt.subplots()
        cs = ax.contourf(epsilons, dataset_sizes,
                         times[dv], [200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260,
                                     265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325,
                                     330, 335, 340, 345], extent=(-0, 2, 0, 10000), cmap="terrain")
        norm = matplotlib.colors.Normalize(
            vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        fig.colorbar(cs, label='Time (RMSPE)')
        ax.set_title(
            'Overhead for ' + dv)
        plt.xlabel('Epsilon')
        plt.ylabel('Dataset size')
        plt.savefig(utils.RESULTS_PATH + '/' + 'plot' + '-' + 'contour' + '-' + 'time' + '-' + dv + '.png', dpi=200,
                    facecolor='w', edgecolor='w', orientation='portrait', bbox_inches='tight')
        plt.clf()
