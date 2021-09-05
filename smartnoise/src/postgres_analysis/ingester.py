from dpevaluation.statistical_analysis.utils import utils
from sqlalchemy import create_engine, FLOAT, TEXT
import pandas as pd
import os


def run():
    utils.log('info', "Ingesting results (csv) to Postgres db...")

    latest_test = utils.get_latest_test()
    data = pd.read_csv(utils.RESULTS_PATH + latest_test)
    try:
        engine = create_engine(utils.POSTGRES_DB)
        table_name = "test_results"

        if not engine.has_table(table_name):
            data.to_sql(table_name, engine,
                        if_exists='replace',
                        dtype={'time': FLOAT,
                               'reader': TEXT,
                               'result': FLOAT,
                               'query': TEXT,
                               'epsilon': FLOAT})
        else:
            utils.log("debug",
                      "Table %s containing test results %s already exists" % (table_name, latest_test))
            return
    except Exception as e:
        utils.log('error',
                  "Could not store csv file to SQL database: %s" % os.path.realpath(__file__))
        print(e)
        exit(1)

    utils.log('info', "Ingestion done")


if __name__ == '__main__':
    run()
