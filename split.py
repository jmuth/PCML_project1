import numpy as np

from helpers import *

JET_INDEX = 22


def split(y, tx, col_index, col_value):
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
        value_y, non_value_y = y[value_rows], y[non_value_rows]
        return {col_value: [value_y, value_tx, value_rows],
                '-1': [non_value_y, non_value_tx, non_value_rows]} # -1 is used becasue there could be different non_col_value
    else:  # col_value is a list
        res = {}
        for v in col_value:
            rows_index = np.where(tx[:, col_index] == v)
            res[v] = [y[rows_index], tx[rows_index], rows_index]
        return res


def split_jets(y, tx):
    """
    split y and tx based on jet column

    Return:
        a tuple of each group
        a group is a tuple consists of y and tx
    """
    groups = split(y, tx, JET_INDEX, [0, 1, 2, 3])
    return groups[0], groups[1], groups[2], groups[3]

