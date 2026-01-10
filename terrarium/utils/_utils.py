'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import pandas as pd

__all__ = [
    'to_list',
    'load_data'
]

def to_list(length, *items):
    lists = []

    for i in items:
        if isinstance(i, (tuple, list)):
            lists.append(i[:length] + i[-1:]*(length - len(i)))
        else:
            lists.append([i]*length)

    if len(lists) == 1:
        return lists[0]
    return lists

def load_data(ann_path, otu_path):
    ann_df = pd.read_csv(ann_path)
    otu_df = pd.read_csv(otu_path)
    data = pd.concat([ann_df, otu_df], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    return data
