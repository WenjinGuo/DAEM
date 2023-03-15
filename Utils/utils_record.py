import numpy as np
import xlrd
from xlutils.copy import copy


def xlsx_record(index_name, index_value, index_better_condition, index_img, xlsx_path):
    wb = xlrd.open_workbook(xlsx_path)
    newb = copy(wb)
    sheet = newb.add_sheet(str(index_img), cell_overwrite_ok=False)
    for i in range(len(index_name)):
        sheet.write(0, i, index_name[i])

    for i in range(index_value.shape[0]):
        for j in range(index_value.shape[1]):
            sheet.write(i+1, j, index_value[i, j])

    # for i in range(index_value.shape[1]):
    #     if index_better_condition == 0:
    #         index_value_i = index_value[:, i]
    #         index_best_i = np.min(index_value_i)
    #         index_best_i_pos = np.where(index_value_i == index_best_i)
    #         sheet.write(index_value.shape[0] + 2, i, index_best_i)
    #         for j in range(len(index_best_i_pos)):
    #             sheet.write(index_value.shape[0] + 3 + j, i, str(index_best_i_pos[j][0]))
    #     else:
    #         index_value_i = index_value[:, i]
    #         index_best_i = np.max(index_value_i)
    #         index_best_i_pos = np.where(index_value_i == index_best_i)
    #         sheet.write(index_value.shape[0] + 2, i, index_best_i)
    #         for j in range(len(index_best_i_pos)):
    #             sheet.write(index_value.shape[0] + 3 + j, i, str(index_best_i_pos[j][0]))

    newb.save(xlsx_path)

