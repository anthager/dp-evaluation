#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

from dpevaluation.utils.log import log, style
from dpevaluation.utils.meta import Meta
import pandas as pd
import numpy as np
import re
import os

# Paths
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
SQL_QUERIES_PATH = ABS_PATH + "/../../../smartnoise/src/postgres_analysis/queries/"
POSTGRES_DB = "postgres://postgres:password@postgres:5432/postgres"


def build_metadata(meta: Meta):
    bounds = meta.columns

    metadata = {
        "Database": {
            "public": {
                "data": {}
            }
        }
    }

    for i in bounds:
        column = i.pop("name")
        if i.get('private'):
            metadata["Database"]["public"]["data"][column] = {
                "private_id": True,
                "type": "int"
            }
        else:
            metadata["Database"]["public"]["data"][column] = i.copy()

    return metadata


def format_metadata(metadata, size):
    prev_size = next(iter(metadata["Database"]["public"]))
    metadata["Database"]["public"]["data_%s" % size] = \
        metadata["Database"]["public"].pop(prev_size)


def parse_query(query):
    match = re.search(
        r'\b' + '(SUM|AVG|COUNT)' + r'\b', query, re.IGNORECASE)

    func = match.group(0)
    index = match.start()
    length = (index + len(func))

    head = query[0:index]
    tail = query[length:]

    _index = tail.find(")")
    _head = tail[0:_index]
    column = _head[1:]
    _tail = tail[_index + 1:]

    if func == "COUNT":
        new_tail = _head + ", %f)" + _tail
        parsed_query = head + "ANON_COUNT" + new_tail
    else:
        new_tail = _head + ", %f, %f, %.3f)" + _tail
        if func == "SUM":
            parsed_query = head + "ANON_SUM_WITH_BOUNDS" + new_tail
        else:
            parsed_query = head + "ANON_AVG_WITH_BOUNDS" + new_tail

    return parsed_query, func, column


def print_progress(epsilon, dataset_size, count):
    string = "%s #%s: e:%s, s:%s" % (
        style("[DBUG]", "blue"), count, epsilon, dataset_size)
    print(f"{string}\r", end="")


def print_progress_extended(count, total_count, entries={}):
    string = ', '.join(["{key}: {value}".format(key=key, value=value) for key, value in entries.items() if not callable(value)])
    log('debug', f'{string} ({count}/{total_count}) \r')


# given an array of rows, merge all the rows to one. If all the cells in a column contain
# the same value, just keep it as it is. If the values are different, concatinate them 
# as strings. This makes this function work for both histograms and regular queries.
def merge_rows(rows):
    if len(rows) == 0:
        return

    # creates an empty array with the same len as the number of columns returned from the query
    results = [None] * len(rows[0])
    for row in rows:
        for i, value in enumerate(row):
            if results[i] is None:
                results[i] = value
            # value is the same as value in accumulator, continue
            elif results[i] == value:
                continue
            else:
                results[i] = str(results[i]) + ',' + str(value)


    return results
