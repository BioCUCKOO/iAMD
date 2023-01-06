# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve


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



X_data = []
X_target = []
data_list = []
data_file = open('Data/Substrate_phosphoproteomics_predictor_data.txt', 'r')

for line in data_file:
    express = []
    line = line.strip().split('\t')
    data = line[0].strip().split(', ')
    for i in range(len(data)):
        express.append(float(data[i]))
    if line[1] == '0':
        data_list.append(line[0])
        X_data.append(express)
        X_target.append(0)
    elif line[1] == '1':
        data_list.append(line[0])
        X_data.append(express)
        X_target.append(1)


total_score = []
TPs = []
TNs = []
FPs = []
FNs = []
Thresholds = []


model = load_model('Model/Substrate_phosphoproteomics_predictor_model.h5')

for x in X_data:
    total_score.extend(model.predict(np.array(x).reshape(1, -1))[:, 1])


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


for c in range(len(Thresholds)):
    if (((TPs[c]+FPs[c])*(TPs[c]+FNs[c])*(TNs[c]+FPs[c])*(TNs[c]+FNs[c]))**0.5) == 0:
        continue
    else:
        Sp = TNs[c]/(TNs[c] + FPs[c])
        Sn = TPs[c]/(TPs[c] + FNs[c])
        MCC = (TPs[c]*TNs[c] - FPs[c]*FNs[c])/(((TPs[c]+FPs[c])*(TPs[c]+FNs[c])*(TNs[c]+FPs[c])*(TNs[c]+FNs[c]))**0.5)


total_auc = roc_auc_score(X_target, total_score)
auc_curve(X_target, total_score, 'Substrate_phosphoproteomics_predictor_model', total_auc)
print("Substrate_phosphoproteomics_predictor_model AUC : ", total_auc)