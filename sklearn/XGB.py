import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# 数据索引
positive_train_index = pd.read_csv(r'..\..\p_train_index.csv', encoding='utf-8', header=None)
positive_test_index = pd.read_csv(r'..\..\p_test_index.csv', encoding='utf-8', header=None)

# 导入相似性数据
sm = pd.read_csv(r'..\..\..\data\sm.csv', encoding='utf-8', header=None)
sd = pd.read_csv(r'..\..\..\data\sd.csv', encoding='utf=8', header=None)
sm = np.array(sm)
sd = np.array(sd)

negative = pd.read_csv(r'..\..\..\data\unlabeled.csv', encoding='utf-8', header=None)
negative = np.array(negative.iloc[:, 0:878])


# 构建特征向量
def data_construct(index):
    md = np.empty((index.shape[0], 878))
    index = index.astype(np.int32)
    for i in range(index.shape[0]):
        m = index.iloc[i, 0]
        d = index.iloc[i, 1]
        m_d = np.append(sm[m-1, :].reshape(1, 495), sd[d-1, :].reshape(1, 383))
        md[i] = m_d

    return md


# 生成训练数据
positive_train = data_construct(positive_train_index)
positive_test = data_construct(positive_test_index)


def ROC_curve(y,pred):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pred_sort = np.sort(pred)[::-1]  #从大到小排序
    index = np.argsort(pred)[::-1]#从大到小排序
    y_sort = y[index]
    print(y_sort)
    tpr = []
    fpr = []
    thr = []
    for i,item in enumerate(pred_sort):
        tpr.append(np.sum((y_sort[:i] == 1)) / pos)
        fpr.append(np.sum((y_sort[:i] == 0)) / neg)
        thr.append(item)
    '''print(len(fpr))
    print(len(tpr))
    print(len(thr))
    print('++++++++++++')'''
    return (fpr, tpr)

def PR_curve(y,pred):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pred_sort = np.sort(pred)[::-1]  # 从大到小排序
    index = np.argsort(pred)[::-1]  # 从大到小排序
    y_sort = y[index]
    print(y_sort)

    Pre = []
    Rec = []
    for i, item in enumerate(pred_sort):
        if i == 0:#因为计算precision的时候分母要用到i，当i为0时会出错，所以单独列出
            Pre.append(1)
            Rec.append(0)

        else:
            Pre.append(np.sum((y_sort[:i] == 1)) /i)
            Rec.append(np.sum((y_sort[:i] == 1)) / pos)
    '''print(len(Pre))
    print(len(Rec))'''

    return (Pre, Rec)


# 五折交叉验证
def cv(positive_train, positive_test, reliable_negative, fold):
    positive_train = positive_train[(fold - 1) * 4344:fold * 4344, :]
    positive_test = positive_test[(fold - 1) * 1086:fold * 1086, :]

    np.random.seed(100)
    a = np.random.choice(reliable_negative.shape[0], 5430, replace=False)

    negative = np.empty((5430, 878))
    for i in range(5430):
        index = a[i]
        negative[i] = reliable_negative[index]

    negative_train = negative[0: 4344, :]
    negative_test = negative[4344: 5430, :]

    label1 = np.ones((4344, 1))
    label2 = np.ones((1086, 1))
    label3 = np.zeros((4344, 1))
    label4 = np.zeros((1086, 1))

    positive_train = np.concatenate((positive_train, label1), axis=1)
    positive_test = np.concatenate((positive_test, label2), axis=1)
    negative_train = np.concatenate((negative_train, label3), axis=1)
    negative_test = np.concatenate((negative_test, label4), axis=1)

    train = np.concatenate((positive_train, negative_train))
    test = np.concatenate((positive_test, negative_test))

    train = shuffle(train, random_state=1)
    test = shuffle(test, random_state=1)

    train_data = train[:, :-1]
    train_target = train[:, -1]
    test_data = test[:, :-1]
    test_target = test[:, -1]

    # xgboost
    xgboost = XGBClassifier(n_estimators=1000, learning_rate=0.1, random_state=1)
    xgboost.fit(train_data, train_target)

    predict_results_proba = xgboost.predict_proba(test_data)[:, 1]
    predict_results = xgboost.predict(test_data)

    # roc
    fpr, tpr = ROC_curve(test_target, predict_results_proba)
    ROC_AUC = metrics.auc(fpr, tpr)

    # pr
    precision_, recall_ = PR_curve(test_target, predict_results_proba)
    PR_AUC = metrics.auc(recall_, precision_)

    # 通过标签计算
    p = precision_score(test_target, predict_results)
    r = recall_score(test_target, predict_results)
    f1 = f1_score(test_target, predict_results)
    acc = accuracy_score(test_target, predict_results)

    print('++++++++++++++')
    print(ROC_AUC)
    print(PR_AUC)
    print(p)
    print(r)
    print(f1)
    print(acc)

    return (ROC_AUC, PR_AUC, p, r, f1, acc, fpr, tpr, precision_, recall_)


roc_auc1, pr_auc1, p1, r1, f11, acc1, fpr1, tpr1, pre1, rec1 = cv(positive_train, positive_test, negative, 1)
roc_auc2, pr_auc2, p2, r2, f12, acc2, fpr2, tpr2, pre2, rec2 = cv(positive_train, positive_test, negative, 2)
roc_auc3, pr_auc3, p3, r3, f13, acc3, fpr3, tpr3, pre3, rec3 = cv(positive_train, positive_test, negative, 3)
roc_auc4, pr_auc4, p4, r4, f14, acc4, fpr4, tpr4, pre4, rec4 = cv(positive_train, positive_test, negative, 4)
roc_auc5, pr_auc5, p5, r5, f15, acc5, fpr5, tpr5, pre5, rec5 = cv(positive_train, positive_test, negative, 5)


# 绘图
paint_fpr = np.vstack((fpr1, fpr2, fpr3, fpr4, fpr5))
paint_tpr = np.vstack((tpr1, tpr2, tpr3, tpr4, tpr5))
paint_pre = np.vstack((pre1, pre2, pre3, pre4, pre5))
paint_rec = np.vstack((rec1, rec2, rec3, rec4, rec5))

paint_fpr = np.mean(paint_fpr, axis=0)
paint_tpr = np.mean(paint_tpr, axis=0)
paint_pre = np.mean(paint_pre, axis=0)
paint_rec = np.mean(paint_rec, axis=0)

LGBM = np.vstack((paint_fpr, paint_tpr, paint_pre, paint_rec))

np.savetxt('XGBdata.csv', LGBM, delimiter=',')


# 输出结果
roc_auc = (roc_auc1 + roc_auc2 + roc_auc3 + roc_auc4 + roc_auc5)/5
pr_auc = (pr_auc1 + pr_auc2 + pr_auc3 + pr_auc4 + pr_auc5)/5
p = (p1 + p2 + p3 + p4 + p5)/5
r = (r1 + r2 + r3 + r4 + r5)/5
f1score = (f11 + f12 + f13 + f14 + f15)/5
acc = (acc1 + acc2 + acc3 + acc4 + acc5)/5

print('++++++++++++++++++')
print('最终结果：')
print('++++++++++++++++++')
print("the average of the AUC is ", roc_auc)
print("the average of the AUPR is ", pr_auc)
print("the average of pre is ", p)
print("the average of rec is ", r)
print("the average of f1_score is ", f1score)
print("the average of acc is", acc)


plt.figure(figsize=(5, 5))  # 创建画布
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200
# 子网格1：ROC曲线
'''plt.subplot(1, 2, 1)  # 第一个子网格
  # 画出ROC曲线'''
'''plt.subplot(1, 2, 2)'''
plt.plot(paint_fpr, paint_tpr, label='ROC')
plt.xlabel('fpr')
plt.ylabel('tpr')
'''plt.plot(paint_fpr, paint_tpr, label='ROC')'''
plt.show()