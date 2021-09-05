from dpevaluation.statistical_analysis.utils import utils
import sqlalchemy
import matplotlib.pyplot as plt

METRICS = ['epsilon', 'time_rmspe']
DIMENSION_VALUES = ['avg', 'count', 'sum']


def empty_result():
    result = {}
    for dv in DIMENSION_VALUES:
        result[dv] = {}
        for m in METRICS:
            result[dv][m] = []


engine = sqlalchemy.create_engine(utils.POSTGRES_DB)
path = utils.SQL_QUERIES_PATH + 'extractor-time.sql'
text = open(path, 'r').read()


def run():
    query = sqlalchemy.text(text)
    rows = engine.execute(query)

    results = {}
    for row in rows:
        q = row.query
        if (not results.get(q)):
            results[q] = {}

        for m in METRICS:
            if (not results[q].get(m)):
                results[q][m] = []
            results[q][m].append(row.__getitem__(m))

    plt.figure()
    for dv in DIMENSION_VALUES:
        plt.plot(results[dv]['epsilon'], results[dv]['time_rmspe'], label=dv)
        plt.xlabel('Epsilon')
        plt.ylabel('Time RMSPE')
        plt.legend()
        plt.savefig(utils.RESULTS_PATH + '/' + 'plot' + '-' + 'time' + '-' + dv + '.png', dpi=200,
                    facecolor='w', edgecolor='w', orientation='portrait', pad_inches=0.1)
        plt.clf()
