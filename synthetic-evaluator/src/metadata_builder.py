from dpevaluation.utils.metadata_builder import build_in_chunks
import os

build_in_chunks(os.environ['DATASET'], 1000)