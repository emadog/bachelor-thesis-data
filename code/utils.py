import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc

def caculate_metric(pred_y, labels, pred_prob):
    """
    计算各种评估指标，包括准确率 (ACC)、精确率 (Precision)、召回率 (Recall/Sensitivity)、特异性 (Specificity)、F1 分数、AUC（曲线下面积）、MCC（马修斯相关系数）等。
    还计算 ROC 曲线和 PR 曲线的数据。

    参数：
    pred_y (list or array): 模型预测的类别标签
    labels (list or array): 真实的标签
    pred_prob (list or array): 模型预测的概率值

    返回：
    metric (torch.Tensor): 计算得到的评估指标，包括 ACC、Precision、Recall、Specificity、F1、AUC 和 MCC
    roc_data (list): ROC 曲线的数据，包括假阳性率 (fpr)、真正率 (tpr) 和 AUC
    prc_data (list): PR 曲线的数据，包括召回率 (recall)、精确率 (precision) 和 AP（平均精度）
    """

    test_num = len(labels)  # 测试样本的数量
    tp = 0  # 真阳性数量
    fp = 0  # 假阳性数量
    tn = 0  # 真阴性数量
    fn = 0  # 假阴性数量

    # 遍历每个样本，计算 TP, FP, TN, FN
    for index in range(test_num):
        if int(labels[index]) == 1:  # 真实标签为正样本
            if labels[index] == pred_y[index]:  # 预测结果也为正样本
                tp = tp + 1  # 正确的正样本预测
            else:
                fn = fn + 1  # 错误的正样本预测
        else:  # 真实标签为负样本
            if labels[index] == pred_y[index]:  # 预测结果也为负样本
                tn = tn + 1  # 正确的负样本预测
            else:
                fp = fp + 1  # 错误的负样本预测

    # 计算准确率 (Accuracy)
    ACC = float(tp + tn) / test_num

    # 计算精确率 (Precision)
    if tp + fp == 0:
        Precision = 0  # 防止除以零
    else:
        Precision = float(tp) / (tp + fp)

    # 计算召回率 (Recall/Sensitivity)
    if tp + fn == 0:
        Recall = Sensitivity = 0  # 防止除以零
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # 计算特异性 (Specificity)
    if tn + fp == 0:
        Specificity = 0  # 防止除以零
    else:
        Specificity = float(tn) / (tn + fp)

    # 计算马修斯相关系数 (MCC)
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0  # 防止除以零
    else:
        # MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        MCC = float(tp * tn - fp * fn) / np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))

    # 计算 F1 分数
    if Recall + Precision == 0:
        F1 = 0  # 防止除以零
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    # 计算 ROC 曲线数据
    labels = list(map(int, labels))  # 转换标签为整数
    pred_prob = list(map(float, pred_prob))  # 转换预测概率为浮点数
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  # 计算假阳性率、真正率和阈值
    AUC = auc(fpr, tpr)  # 计算 ROC 曲线下面积

    # 计算 PR 曲线数据
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)  # 计算精确率、召回率和阈值
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)  # 计算平均精度

    # 返回计算得到的指标和曲线数据
    metric = [ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC]
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data
