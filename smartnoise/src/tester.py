#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# from postgres_analysis import ingester, extractor_contour, extractor_contour_time
from dpevaluation.statistical_queries.runner import run_test
import os
from dpevaluation.statistical_queries import sq_utils
from dpevaluation.utils.meta import Meta

import numpy as np
import time

from opendp.smartnoise.sql import PrivateReader, PostgresReader
from opendp.smartnoise.metadata import CollectionMetadata


def main(ds='medical-survey',
         rewrite=False):


    dataset_metadata = Meta('smartnoise', ds)
    
    # Build metadata for PrivateReader
    metadata = sq_utils.build_metadata(dataset_metadata)

    # Init PostgresReader
    postgres_reader = PostgresReader(host="postgres",
                                     database="postgres",
                                     user="postgres",
                                     password="password",
                                     port="5432")


    def run_query(epsilon, size, query):
        sq_utils.format_metadata(metadata, size)
        meta = CollectionMetadata.from_dict(metadata)

        # non-private
        if (epsilon == -1.0):
            reader = postgres_reader
        # private
        else:
            reader = PrivateReader(postgres_reader, meta, epsilon)

        format_query = query % size

        res = reader.execute(format_query)

        header = res[0]
        rows = res[1:]

        return header, rows

    run_test(run_query, 'smartnoise', ds, rewrite)
    

if __name__ == "__main__":
    ds = os.environ['DATASET']
    if not ds:
        raise('make sure the environment variable DATASET is set')
    main(ds)
