from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt



def calculate_metrics(y_true, y_pred, y_score):
    # 初始化计数器
    TP = FP = TN = FN = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            TP += 1
        elif yt == 0 and yp == 0:
            TN += 1
        elif yt == 0 and yp == 1:
            FP += 1
        elif yt == 1 and yp == 0:
            FN += 1

    # 计算指标
    acc = (TP + TN) / (TP + FP + TN + FN)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    ba = ((TP / (TP + FN)) + (TN / (TN + FP))) / 2
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print(TP)
    print(TN)
    return acc, auc_score, ba, recall, precision, f1






# 使用示例
# acc, ba, recall, precision, f1 = calculate_metrics(np.array(y_true), np.array(y_pred))
