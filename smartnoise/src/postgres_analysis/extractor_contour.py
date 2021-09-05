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
path = utils.SQL_QUERIES_PATH + '/extractor-contour.sql'
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

        results[q][row.dataset_size][row.epsilon] = row.accuracy

    epsilons = np.array(sorted(list(epsilons_set)))
    dataset_sizes = np.array(sorted(list(dataset_sizes_set)))

    accuracies = {}
    for q in QUERIES:
        accuracy = np.zeros((len(dataset_sizes), len(epsilons)))
        for i, ds in enumerate(dataset_sizes):
            for j, e in enumerate(epsilons):
                accuracy[i][j] = results[q].get(ds).get(e)
        accuracies[q] = accuracy
    
    plt.figure()
    for dv in QUERIES:
        matplotlib.rcParams.update({
            'axes.titlesize': 22,
            'axes.titlepad': 12.0,
            'axes.labelsize': 20,
            'axes.labelpad': 2.0,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18})

        fig, ax = plt.subplots()

        cs = plt.contourf(
            epsilons,
            dataset_sizes,
            accuracies[dv], [0.5, 0.75,  1, 2, 3, 5, 10],
            extent=(-0, 2, 0, 10000),
            cmap="terrain")

        norm = matplotlib.colors.Normalize(
            vmin=cs.cvalues.min(),
            vmax=cs.cvalues.max())

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        fig.colorbar(cs, label='Accuracy (RMSPE)')
        ax.set_title('Accuracy for ' + dv)
        plt.xlabel('Epsilon')
        plt.ylabel('Dataset size')

        # plt.legend()

        plt.savefig(
            utils.RESULTS_PATH + '/' + 'plot' + '-' + 'contour' + '-' + dv + '.png',
            dpi=200,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            bbox_inches='tight')

        plt.clf()
