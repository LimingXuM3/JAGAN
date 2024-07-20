import pandas as pd
import numpy as np
csv_g = pd.read_csv("mimic/output-ground-1.csv")
header = csv_g.columns.tolist()
print(header[1])
csv_r = pd.read_csv("mimic/precdition.csv")
print(len(csv_r))
ex = Exception('Inconsistent table length')
if len(csv_g) != len(csv_r):
    raise ex
else:
    print("Consistent table length, Let's strat")
macro_precision = 0 ; macro_recall = 0 ; macro_f1 = 0; AUCs = []
for n in header:
    if n == "Reports" or n == "No Finding":
        continue
    print("The traversal list:" + n)
    data_g_n = np.array(csv_g[n])
    data_r_n = np.array(csv_r[n])
    print("-------------------------------------------------")

    TP = 0 ; TN = 0 ; FP = 0 ; FN = 0
    for i in range(1,len(data_g_n)):
        if data_g_n[i] == data_r_n[i]:
            TP = TP + 1
        elif (data_g_n[i] == 1 or data_g_n[i] ==0) and np.isnan(data_r_n[i]):
            FN = FN + 1
        elif np.isnan(data_g_n[i]) and np.isnan(data_r_n[i]): 
            TN = TN + 1
        elif (np.isnan(data_g_n[i]) and data_r_n[i] == 1 ) or (data_g_n[i] == 1 and data_r_n[i] == 0 ) \
                or (data_g_n[i] == 0 and data_r_n[i] == 1 ): 
            FP = FP + 1
    if TP ==0 :
        continue
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*(precision*recall)/(precision+recall)
    print(precision)
    print(recall)
    macro_precision = macro_precision + precision
    macro_recall = macro_recall + recall
    macro_f1 = macro_f1 + f1
    print(1)
    print(TP)
print("macro_precision：")
print(macro_precision/(len(header)-2))
print("macro_recall：")
print(macro_recall/(len(header)-2))
print("macro_f1：")
print(macro_f1/(len(header)-2))
