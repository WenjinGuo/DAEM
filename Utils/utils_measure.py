import numpy as np
import xlrd
from xlutils.copy import copy


def read_value(xlsx_path):
    readbook = xlrd.open_workbook(xlsx_path)
    sheets = readbook.sheet_names()
    image_index = {}
    sheet_num = 0
    sheet_names = []
    for k in range(len(sheets)):
        sheet = readbook.sheet_by_index(k)
        nrows = sheet.nrows
        ncols = sheet.ncols
        if nrows == 0:
            continue
        index_values = np.zeros([nrows-1, ncols], dtype=np.float32)
        index_names = []
        for j in range(ncols):
            index_names.append(sheet.cell(0, j).value)
            for i in range(nrows-1):
                index_values[i, j] = sheet.cell(i+1, j).value
        image_index[sheet_num] = index_values
        sheet_names.append(sheet.name)
        sheet_num = sheet_num + 1

    return index_names, sheet_names, image_index


def xlsx_mean(xlsx_path, sheet_list, col):
    index_names, sheet_names, image_index = read_value(xlsx_path)
    value_sheet = np.zeros(len(sheet_list))
    for i in range(len(sheet_list)):
        location = sheet_names.index(str(sheet_list[i]))
        value_sheet[i] = image_index[location][10, col]

    return np.mean(value_sheet)
