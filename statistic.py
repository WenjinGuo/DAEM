from Utils.utils_measure import xlsx_mean

path = './index_record_test.xls'
sheet_list = [i for i in range(16, 32)]
psnr_mean = xlsx_mean(path, sheet_list, 4)
sam_mean = xlsx_mean(path, sheet_list, 5)
ergas_mean = xlsx_mean(path, sheet_list, 8)
uqi_mean = xlsx_mean(path, sheet_list, 9)


print('PSNR: {0}, SAM: {1}, ERGAS: {2}, UIQI: {3}'.format(psnr_mean, sam_mean, ergas_mean, uqi_mean))