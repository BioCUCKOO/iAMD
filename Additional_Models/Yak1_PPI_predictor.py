# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import joblib


def auc_curve(y, prob, tag, auc):
    fpr, tpr, threshold = roc_curve(y, prob)

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.3f)' % auc)  # 假阳率为横坐标，真阳率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig(tag + '.png')


Pho_data = open('Data/Yak1_PPI_phosphoproteomics_predictor_result.txt', 'r')
Pse_data = open('Data/Yak1_PPI_PseAAC_predictor_result.txt', 'r')


pho_dict = {}
Pse_dict = {}
Hybrid_dict = {}


for line in Pho_data:
    line = line.strip().split('\t')
    pho_dict[line[0]] = [float(line[1]), int(line[2])]

for line in Pse_data:
    line = line.strip().split('\t')
    Pse_dict[line[0]] = [float(line[1]), int(line[2])]


weight_pho = 6
weight_pse = 1


for key, value in Pse_dict.items():
    Hybrid_dict[key] = [pho_dict[key][0] * weight_pho + value[0] * weight_pse]


X_data = []
X_target = []
data_list = []


for key, value in Pse_dict.items():
    if value[1] == 0:
        data_list.append(key)
        X_data.append(Hybrid_dict[key])
        X_target.append(0)
    elif value[1] == 1:
        data_list.append(key)
        X_data.append(Hybrid_dict[key])
        X_target.append(1)


total_score = []
TPs = []
TNs = []
FPs = []
FNs = []
Thresholds = []


lr = joblib.load('Model/Yak1_PPI_predictor_model.pkl')

total_score = list(lr.predict_proba(np.array(X_data))[:, 1])


# 计算 Sp, Sn, MCC 
for threshold in total_score:
    Thresholds.append(threshold)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(total_score)):
        if total_score[i] >= threshold:
            if X_target[i] == 1:
                tp += 1
            else:
                fp += 1
        elif total_score[i] < threshold:
            if X_target[i] == 1:
                fn += 1
            else:
                tn += 1
    TPs.append(tp)
    FPs.append(fp)
    FNs.append(fn)
    TNs.append(tn)



total_auc = roc_auc_score(X_target, total_score)
auc_curve(X_target, total_score, 'Yak1_PPI_predictor_model', total_auc)
print("Yak1_PPI_predictor_model AUC : ", total_auc)
