import os

import numpy as np
import pandas as pd

def gets(labels,pred,step):

    lb = labels.cpu().numpy()
    out_pred_1 = pred.cpu().numpy()
    p_name = './tsne/pre/' + str(step) + '.txt'
    l_name = './tsne/lb/' + str(step) + '.txt'
    np.savetxt(p_name, out_pred_1, fmt='%.3f')
    np.savetxt(l_name, lb, fmt='%d')


def get_cat_pre() :
    meragefiledir_1 = os.getcwd() + '/tsne/pre/'
    filenames_1 = os.listdir(meragefiledir_1)
    file_1 = open('./tsne/pre/pre_total.txt', 'w')

    for filename in filenames_1:
        filepath = meragefiledir_1 + '/'
        filepath = filepath + filename
        for line in open(filepath):
            file_1.writelines(line)

    file_1.close()

def get_cat_lb():
    meragefiledir_2 = os.getcwd() + '/tsne/lb/'
    filenames_2 = os.listdir(meragefiledir_2)
    file_2 = open('./tsne/lb/lb_total.txt', 'w')

    for filename in filenames_2:
        filepath = meragefiledir_2 + '/'
        filepath = filepath + filename
        for line in open(filepath):
            file_2.writelines(line)

    file_2.close()

    df = pd.read_csv('./tsne/pre/pre_total.txt', delimiter=' ')
    df.to_csv('./tsne/pre/pre_total.csv', encoding='utf-8', index=False)


