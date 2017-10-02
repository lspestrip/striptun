#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from typing import Any, Dict
from collections import namedtuple, OrderedDict
import xlrd
import numpy as np


def read_worksheet_table(wks: xlrd.book.Book) -> Dict[str, Any]:
    '''Read a table of numbers from an Excel file saved by Keithley.

    This function reads the first worksheet in the Excel file passed as
    argument and returns a dictionary associating NumPy arrays with their
    names.
    '''

    sheet = wks.sheet_by_index(0)
    result = OrderedDict()
    nrows = sheet.nrows
    for cur_col in range(sheet.ncols):
        name = sheet.cell(0, cur_col).value
        if len(name) >= len('START') and name[:5] == 'START':
            # This column and the following are not useful
            break
        result[name] = np.array([sheet.cell(i, cur_col).value
                                 for i in range(1, nrows)])

    return result


def read_worksheet_settings(wks: xlrd.book.Book) -> Dict[str, Any]:
    sheet = wks.sheet_by_name('Settings')
    result = {}
    for cur_row in range(sheet.nrows):
        key = str(sheet.cell(cur_row, 0).value)
        if key == '':
            continue

        if key == 'Formulas':
            break

        values = []
        for cur_col in range(1, sheet.ncols):
            cur_value = sheet.cell(cur_row, cur_col).value
            if str(cur_value) == '':
                break

            values.append(cur_value)

        if len(values) == 1:
            values = values[0]

        result[key] = values

    return result
