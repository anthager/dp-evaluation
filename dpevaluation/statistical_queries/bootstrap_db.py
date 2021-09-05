#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

from sqlalchemy.sql.sqltypes import FLOAT
from dpevaluation.statistical_queries.sq_utils import POSTGRES_DB
from dpevaluation.utils.log import log
from dpevaluation.utils.meta import Meta
from sqlalchemy import create_engine


def run(ds, dataset_sizes, rewrite=False):
    log("info", "Bootstraping Postgres db...")

    meta = Meta(None, ds)
    datasets, dataset_sizes = meta.split_dataset()

    log("debug", "Dataset sizes: %s" % str(dataset_sizes)[1:-1])

    for dataset in datasets:
        try:
            engine = create_engine(POSTGRES_DB)
            size = dataset.shape[0]
            table = "data_%s" % size
            write = True if rewrite else not engine.has_table(table)

            if write:
                dataset.to_sql(table, engine, if_exists="replace", dtype=FLOAT)
                log("debug", "Dataset size: %s done" % size)
            else:
                log("debug", "Table \"%s\" already exists" % table)
        except Exception as e:
            log("error", "Could not write DataFrame to SQL database")
            log("warn", "Make sure Postgres docker is running")
            print(e)
            exit(1)

    log("info", "Bootstrap done")

    return dataset_sizes

if __name__ == "__main__":
    run()
