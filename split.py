import numpy as np

from helpers import *

JET_INDEX = 22

JET0_COLUMNS = [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]
JET1_COLUMNS = [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25]
JET2_COLUMNS = JET3_COLUMNS = [i for i in range(29) if i != JET_INDEX]

COLUMNS = [JET0_COLUMNS, JET1_COLUMNS, JET2_COLUMNS, JET3_COLUMNS]


def split(y, tx, method):
    """
    serves as public api function

    Params:
        method (str): could be 'jet' or 'mass'
    """
    if method == 'jet':
        return split_jets(y, tx)
    elif method == 'mass':
        return split_jets_and_mass(y, tx)
    else:
        raise Exception('no split method')


def split_jets(y, tx):
    """
    split y and tx based on jet column

    Return:
        a tuple of each group
        a group is a tuple consists of y and tx
    """
    groups = _split(y, tx, JET_INDEX, [0, 1, 2, 3])
    return groups[0], groups[1], groups[2], groups[3]


def split_jets_and_mass(y, tx):
    """
    split y and tx based on jet column and mass value in the first feature

    Return:
        a list of 8 groups
        a group is a tuple consists of y, tx and their ids
    """
    groups = split_jets(y, tx)
    res = []
    for group in groups:
        jet_y, jet_x, jet_indices = group
        subgroups = _split(jet_y, jet_x, 0, -999, jet_indices[0])  # split based on mass for the first column
        res.append(subgroups[0])
        res.append(subgroups[1])

    return res


def _split(y, tx, col_index, col_value, ids=None):
    """
    split y and tx into different groups, based on column index and col_value
    each group has col_value in column col_index

    Params:
        y (ndarray): target variable
        tx (ndarray): feature matrix
        col_index (int): feature index in tx
        col_value (int/list): split criteria based on this value
    """
    if isinstance(col_value, int):
        value_rows = np.where(tx[:, col_index] == col_value)
        non_value_rows = np.delete(np.arange(tx.shape[0]), value_rows)
        value_tx, non_value_tx = tx[value_rows], tx[non_value_rows]
        if len(y) != 0:
            value_y, non_value_y = y[value_rows], y[non_value_rows]
        else:
            value_y = non_value_y = []
        return [[value_y, np.delete(value_tx, col_index, axis=1), ids[value_rows]],
                [non_value_y, non_value_tx, ids[non_value_rows]]]
    else:  # col_value is a list
        res = {}
        for v, columns in zip(col_value, COLUMNS):
            rows_index = np.where(tx[:, col_index] == v)
            y_value = y[rows_index] if any(y) else []
            res[v] = [np.array(y_value), np.array(tx[rows_index][:, columns]), rows_index]
            
        return res

