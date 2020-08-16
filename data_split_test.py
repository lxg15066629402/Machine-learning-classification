#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xlrd
import numpy as np
from sklearn import svm  # svm支持向量机

from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier  # 多层感知机
from sklearn.linear_model import LogisticRegression  #

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.ensemble import BaggingClassifier

from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path


# read data
def get_data(path, x):

    # 打开excel文件，创建一个workbook对象,book对象也就是fruits.xlsx文件,表含有sheet名
    rbook = xlrd.open_workbook(path)
    # sheets方法返回对象列表,[<xlrd.sheet.Sheet object at 0x103f147f0>]
    rbook.sheets()
    # xls默认有3个工作簿,Sheet1,Sheet2,Sheet3
    rsheet = rbook.sheet_by_index(0)  # 取第一个工作簿

    data_c = []
    for i in x:
        data_c1 = []
        for row in rsheet.get_rows():
            # for product_column in row:
            product_column = row[i]  # 品名所在的列
            product_value = product_column.value  # 项目名
            data_c1.append(product_value)
        data_c.append(data_c1)
    data_c = np.array(data_c)
    return data_c


# 数据清洗
def _no_dele(path, x):
    data = get_data(path, x)
    w, h = data.shape
    no_dele_data = []
    for i in range(h):
        for j in range(w):
            # if data[j][i] == 69:
            #      #print(data[:,i])
            if ',1' in data[j][i]:
                data[j][i] = '1'
            if '24-36' in data[j][i]:
                data[j][i] = '30'

            if data[j][i] in ['', '0正在查', '1正在查', '·', '/', '>10', '正在检查', '正在查', '一年以上', '记不清', '无', '12045.0', '12050.0', '12060.0', '12064.0', '12065.0', '12043.0', '12052.0', '＞10']:
                # print(data[:,i])
                break
            if j == w-1:
                no_dele_data.append(data[:, i])
    no_dele_data = np.array(no_dele_data)

    return no_dele_data


def _no_rows(path, x):
    data = _no_dele(path, x)
    for i in range(int(data.shape[0])):
        if '～' in data[i][3] or '~' in data[i][3]:
            data[i][3] = (int(data[i][3][0]) + int(data[i][3][2]))/2.0
    n_data = data[1:, :].astype(np.float)

    return n_data


def _get_data_tainandtext():
    x = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 25, 42]  # data + label
    path = '上海新华医院 - 儿童哮喘数据整理（380例+198例）.xlsx'
    train = _no_rows(path, x)

    return train


def pretty_print_linear(coefs, names=None, sort = False):
    if names == None:
        names = ["%X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))

    return '+'.join("%s * %s" % (round(coef, 3), name) for coef, name in lst)


def train():
    train = _get_data_tainandtext()
    lengh = train.shape[0]
    m = 5   # 5 cross-validation 
    k = int(lengh/m)
    print(lengh)
    auc_ = []
    acc = []
    # m cross validation
    k_5 = [k*x for x in range(m+1)]

    # feature choose method
    names = []
    for i in range(len(train[0, :])):
        names.append(i)
    model = LassoCV(alphas=np.linspace(1e-4, 1, 1000), fit_intercept=True, normalize=False)  # error

    model.fit(train[:, 0:19], np.squeeze(train[:, 19:]))

    print("Lasso model:", pretty_print_linear(model.coef_, names, sort=True))

    # lasso method
    alphas_lasso, coefs_lasso, _ = lasso_path(train[:, 0:19], np.squeeze(train[:, 19:]),
                                              fit_intercept=False,
                                              alphas=np.linspace(model.alpha_, 0.5, 100))  # model.alpha_
    plt.figure(1)
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    for coef_l in coefs_lasso:
        plt.plot(neg_log_alphas_lasso, coef_l)
    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Lasso Paths')
    plt.axis('tight')
    plt.savefig("figure_LR/lasso_path.png", dpi=600)
    plt.show()

    data = np.concatenate([train[:, 1:4], train[:, 6:8], train[:, 13:15], train[:, 18:19], train[:, 19:]], axis=1)
    print(data.shape)

    # plot ROC curve and area the curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 5K cross-validation
    for i in range(m):
        test = data[k_5[i]:k_5[i + 1]]
        train = np.concatenate([data[:k_5[i]], data[k_5[i+1]:]], axis=0)

        test_y, test_x, train_y, train_x = test[:, -1], test[:, :-1], train[:, -1], train[:, :-1]



        # LR
        # clf = LogisticRegression()  # 逻辑回归

        # RF
        # clf = RandomForestClassifier(n_estimators=1000, max_features=8)  #

        # SVM
        # clf = svm.SVC(kernel='rbf', C=10, gamma=0.01, probability=True)  # svm
        
        # MLPC
        clf = MLPClassifier(activation='relu', solver='adam', alpha=0.0001)


        clf.fit(preprocessing.scale(train_x), train_y)  # 训练模型
        result = clf.predict(preprocessing.scale(test_x))  # 使用模型预测值 auc_score
        result_ = clf.predict_proba(preprocessing.scale(test_x))  # auc value

        # auc_score
        result1 = np.rint(result)  # 四舍五入
        test_auc = metrics.roc_auc_score(test_y, result)  # 验证集上的auc值
        test_acc = metrics.accuracy_score(test_y, result1)  # 准确率
        auc_.append(test_auc)
        acc.append(test_acc)

        # plot ROC cure and area the curve
        fpr, tpr, threshold = metrics.roc_curve(test_y, result_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label="ROC fold %d AUC = %0.3f" % (i, roc_auc))

    # figure
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='binary class', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=0.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel("True Positive Rate")
    plt.xlabel('False Positive Rate')
    plt.title("Receiver operating characteristic example")
    plt.legend(loc='lower right')
    # save file
    # plt.savefig("figure_LR/filename_lr.png")
    # plt.savefig("figure_RF/filename_rf.png")
    # plt.savefig("figure_SVM/filename_svm.png")
    plt.savefig("figure_MLPC/filename_mlpn.png")
    plt.show()

    print("auc_:", np.mean(auc_, axis=0))
    print("acc:", np.mean(acc))


if __name__ == "__main__":
    train()


# Lasso model:  right
# 0.121 * 7 + =====
# 0.074 * 6 + =====
# 0.07 * 18 + =====
# -0.055 * 8 +
# -0.047 * 10 +
# 0.042 * 3 +  ====
# -0.032 * 0 +
# -0.028 * 16 +
# -0.021 * 15 +
# 0.02 * 14 +  =====
# -0.018 * 17 +
# 0.014 * 1 +  =====
# 0.008 * 13 + =====
# -0.006 * 4 +
# -0.005 * 12 +
# -0.005 * 11 +
# 0.005 * 2 + =====
# -0.004 * 5 +
# -0.002 * 9

# Lasso model: 0.089 * 7+0.041 * 3+0.039 * 18+0.037 * 6+-0.026 * 8+-0.023 * 10+0.014 * 1+-0.006 * 4+0.004 * 2+-0.003 * 0+-0.0 * 5+-0.0 * 9+-0.0 * 11+0.0 * 12+0.0 * 13+0.0 * 14+-0.0 * 15+-0.0 * 16+-0.0 * 17

