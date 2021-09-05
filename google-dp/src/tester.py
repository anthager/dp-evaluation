#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

from dpevaluation.statistical_queries.runner import run_test
from dpevaluation.statistical_queries import sq_utils
from dpevaluation.utils.meta import Meta
import psycopg2 as pg
import os


def main(ds='medical-survey',
         rewrite=False):

    dataset_metadata = Meta('google_dp', ds)

    # Build metadata for PrivateReader
    metadata = sq_utils.build_metadata(dataset_metadata)

    postgres_connection = \
        '''
            dbname='postgres'
            host='postgres'
            port='5432'
            user='postgres'
            password='password'
        '''

    # the function is decleared here so it has database connections and other misc stuff in it's closure
    def run_query(epsilon, size, query):
        # non-private
        if (epsilon == -1.0):
            format_query = query % size
        # private
        else:
            private_query, func, column = sq_utils.parse_query(query)
            bounds = metadata["Database"]["public"]["data"][column]
            # ANON_COUNT(q1, %f)
            if func == "COUNT":
                format_query = private_query % (epsilon, size)
            # ANON_AVG_WITH_BOUNDS(column, lower, upper, epsilon)
            # ANON_SUM_WITH_BOUNDS(column, lower, upper, epsilon)
            else:
                format_query = \
                    private_query % (bounds['lower'], bounds['upper'], epsilon, size)

        connection = pg.connect(postgres_connection)
        cursor = connection.cursor()

        cursor.execute(format_query)
        header = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        return header, rows

    run_test(run_query, 'google_dp', ds, rewrite)


if __name__ == "__main__":
    ds = os.environ['DATASET']
    if not ds:
        raise('make sure the environment variable DATASET is set')
    main(ds)
