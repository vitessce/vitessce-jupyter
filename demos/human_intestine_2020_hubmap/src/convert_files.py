import argparse

from vitessce import SnapWrapper
from os.path import join
from scipy.io import mmread
import numpy as np
import pandas as pd
import json

def convert_files(mtx, barcodes_df, bins_df, clusters_df, output_genomic_profiles, output_cells, output_cell_sets):
    w = SnapWrapper(mtx, barcodes_df, bins_df, clusters_df)
    w.create_genomic_multivec_zarr(output_genomic_profiles)
    with open(output_cell_sets, 'w') as f:
        f.write(json.dumps(w.create_cell_sets_json()))
    with open(output_cells, 'w') as f:
        f.write(json.dumps(w.create_cells_json()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-im', '--input_mtx', type=str, required=True)
    parser.add_argument('-iba', '--input_barcodes', type=str, required=True)
    parser.add_argument('-ibi', '--input_bins', type=str, required=True)
    parser.add_argument('-ia', '--input_annotations', type=str, required=True)
    parser.add_argument('-ogp', '--output_genomic_profiles', type=str, required=True)
    parser.add_argument('-oc', '--output_cells', type=str, required=True)
    parser.add_argument('-ocs', '--output_cell_sets', type=str, required=True)
    args = parser.parse_args()

    mtx = mmread(args.input_mtx)
    barcodes_df = pd.read_csv(args.input_barcodes, header=None)
    bins_df = pd.read_csv(args.input_bins, header=None)
    clusters_df = pd.read_csv(args.input_annotations, index_col=0)

    convert_files(
        mtx, barcodes_df, bins_df, clusters_df,
        args.output_genomic_profiles, args.output_cells, args.output_cell_sets
    )
