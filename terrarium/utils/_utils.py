'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

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
