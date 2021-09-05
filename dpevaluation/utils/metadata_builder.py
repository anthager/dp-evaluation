#!/usr/bin/python3.7
# -*- coding: utf-8 -*-


from dpevaluation.statistical_queries.sq_utils import print_progress_extended
from dpevaluation.utils.log import log
from dpevaluation.utils.date import timestamp_to_epoch
from dpevaluation.utils.meta import Meta
from dpevaluation.utils.timestamp import timestamp
import pandas as pd
import numpy as np
import yaml
import os

# https://stackoverflow.com/questions/17315737/split-a-large-pandas-dataframe/28882020#28882020
def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def build(ds):
    meta = Meta(None, ds)
    build_meta_and_numbers_dataset(meta)

def build_in_chunks(ds, chunk_size: int = 10000):
    meta = Meta(None, ds)
    data = meta.raw_dataset
    chunks = split_dataframe(data, chunk_size)
    file_tag = timestamp("%f")  

    for i, chunk in enumerate(chunks):
        # if (i == 10):
        #     quit()
        meta.raw_dataset = chunk
        new_metadata = build_meta_and_numbers_dataset(meta, file_tag)
        meta._meta = new_metadata
        print_progress_extended(i, len(chunks))


def build_meta_and_numbers_dataset(meta: Meta, file_tag = timestamp("%f")):
    data = meta.raw_dataset
    new_metadata = meta._meta if meta._meta is not None else {
        'ordinal_columns': [],
        'categorical_columns': [],
        'other_columns': [],
    }
    parsed_data = {}


    for col in data.columns:

        def handle_number(_data, col_type):
            _old_col_obj = [c for c in new_metadata[col_type] if c['name'] == col]
            old_col_obj = _old_col_obj[0] if len(_old_col_obj) else None

            col_max = round(float(_data[col].max()), 6)
            col_min = round(float(_data[col].min()), 6)

            if old_col_obj is None:
                new_metadata[col_type].append({
                    'name': col,
                    'upper': col_max,
                    'lower': col_min
                })
            else:
                old_col_obj['lower'] = min(old_col_obj['lower'], col_min)
                old_col_obj['upper'] = max(old_col_obj['upper'], col_max)

            parsed_data[col] = _data[col].to_numpy()

        # Set floats as other_columns and write to metadata.
        # Look up min and max values for bounds
        if data[col].dtype == np.float:
            handle_number(data, 'other_columns')

        # Set ints as categorical. If this is wrong -> change the metadata manually
        # Look up min and max values for bounds
        elif data[col].dtype == np.int:
            handle_number(data, 'categorical_columns')

        # Make strings to categorical columns denotes as numbers.
        # Write a map with number -> string in .metadata
        elif data[col].dtype == np.object:
            # the column is a date column, the format is hard coded after az date format
            try:
                # test if the first value is a date, if it is, assume all are. If it's not, continue with string
                timestamp_to_epoch(data[col].iloc[0])
                data[col] = data[col].apply(timestamp_to_epoch)
                handle_number(data, 'other_columns')
                continue
            except:
                pass


            _old_col = [c for c in new_metadata['categorical_columns'] if c['name'] == col]
            old_col = _old_col[0] if len(_old_col) != 0 else None
            old_col_meta_values = list(old_col['meta'].values()) if old_col is not None else []
            
            # Drop sort all the unique string values in an array
            # We sort to make sure that '10-19' & '20-29' keep their ordinal property
            # Note that this isn't true when merging with another metadata file but that is inevitable
            new_col_meta_values = data[col].astype(str).drop_duplicates().to_numpy()
            meta_values_exclusive_for_new = sorted(np.setdiff1d(new_col_meta_values, old_col_meta_values))
            categories =  old_col_meta_values + meta_values_exclusive_for_new
            category_n = range(len(categories))


            if old_col is not None:
                old_col['upper'] = max(old_col['upper'], category_n[-1])
                old_col['lower'] = min(old_col['lower'], category_n[0])
                old_col['meta'] = dict(zip(category_n, categories))
            else: 
                new_metadata['categorical_columns'].append({
                    'name': col,
                    'upper': category_n[-1],
                    'lower': category_n[0],
                    'meta': dict(zip(category_n, categories))
                })

            parsed_data[col] = \
                data.replace(categories, value=category_n)[col].to_numpy()

        else:
            raise('Uncaught type: ' + str(data[col].dtype))

    new_df = pd.DataFrame(parsed_data)

    # Write metadata to .yml file
    metadata_path = remove_extension_from_path(meta.metadata_path)
    with open(metadata_path + "-" + file_tag + ".yml", 'w') as file:
        yaml.dump(new_metadata, file)

    file_exists = os.path.isfile(meta.dataset_path)
    if file_exists:
        new_df.to_csv(meta.dataset_path, index=False, mode='a', header=False)
    else:
        new_df.to_csv(meta.dataset_path, index=False)

    return new_metadata


def remove_extension_from_path(path):
    """
    Removes the file extension. split on ".", take all but last element 
    and join all the elements together with "."
    """
    return '.'.join(path.split('.')[:-1])

