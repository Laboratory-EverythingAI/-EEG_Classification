# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 14:57:27 2018

@author: Winham

# CPSC_combine_leads.py: 组合各导联神经网络的输出特征

"""

import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as bk
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from CPSC_config import Config
import CPSC_utils as utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
config = Config()
config.MODEL_PATH = 'D:/CPSC_Scheme/Net_models/'

records_name = np.array(os.listdir(config.DATA_PATH))
#减1 是想让 正常为 0 ，数组坐标一般由0开始，并且独热编码会少一位。
records_label = np.load(config.REVISED_LABEL) - 1
class_num = len(np.unique(records_label))

train_val_records, test_records, train_val_labels, test_labels = train_test_split(
    records_name, records_label, test_size=0.2, random_state=config.RANDOM_STATE)

train_records, val_records, train_labels, val_labels = train_test_split(
    train_val_records, train_val_labels, test_size=0.2, random_state=config.RANDOM_STATE)

train_records, train_labels = utils.oversample_balance(train_records, train_labels, config.RANDOM_STATE)
val_records, val_labels = utils.oversample_balance(val_records, val_labels, config.RANDOM_STATE)

for i in range(config.LEAD_NUM):  # 分别载入各个导联对应的模型进行概率预测，并拼接
    #数据处理
    TARGET_LEAD = i
    train_x = utils.Fetch_Pats_Lbs_sLead(train_records, Path=config.DATA_PATH,
                                     target_lead=TARGET_LEAD, seg_num=config.SEG_NUM, 
                                     seg_length=config.SEG_LENGTH)
    train_y = to_categorical(train_labels, num_classes=class_num)
    val_x = utils.Fetch_Pats_Lbs_sLead(val_records, Path=config.DATA_PATH,
                                     target_lead=TARGET_LEAD, seg_num=config.SEG_NUM, 
                                     seg_length=config.SEG_LENGTH)
    val_y = to_categorical(val_labels, num_classes=class_num)
    for j in range(train_x.shape[0]):
        train_x[j, :, :] = scale(train_x[j, :, :], axis=0)  
    for j in range(val_x.shape[0]):
        val_x[j, :, :] = scale(val_x[j, :, :], axis=0) 
    model_name = 'net_lead_' + str(TARGET_LEAD) + '.hdf5'
    #加载模型
    model = load_model(config.MODEL_PATH + model_name)
    pred_nnet_rt = model.predict(train_x, batch_size=64, verbose=1)
    del train_x
    pred_nnet_vt = model.predict(val_x, batch_size=64, verbose=1)
    del val_x
    
    test_x = utils.Fetch_Pats_Lbs_sLead(test_records, Path=config.DATA_PATH,
                                     target_lead=TARGET_LEAD, seg_num=config.SEG_NUM, 
                                     seg_length=config.SEG_LENGTH)
    test_y = to_categorical(test_labels, num_classes=class_num)
    for j in range(test_x.shape[0]):
        test_x[j, :, :] = scale(test_x[j, :, :], axis=0)
        
    pred_nnet_tt = model.predict(test_x, batch_size=64, verbose=1)
    del test_x

#拼接输出结果。 最终的结果会是 12 * 8 列数据，行数为 输入模型数据的行数。LEAD_NUM = 12 。独热编码 0 0 0 0 0 0 0 0 8 位。
    if i == 0:
        pred_nnet_r = pred_nnet_rt[:, 1:]
        pred_nnet_v = pred_nnet_vt[:, 1:]
        pred_nnet_t = pred_nnet_tt[:, 1:]
    else:
        pred_nnet_r = np.concatenate((pred_nnet_r, pred_nnet_rt[:, 1:]), axis=1)
        pred_nnet_v = np.concatenate((pred_nnet_v, pred_nnet_vt[:, 1:]), axis=1)
        pred_nnet_t = np.concatenate((pred_nnet_t, pred_nnet_tt[:, 1:]), axis=1)
    del model
    bk.clear_session()
    tf.reset_default_graph()
#保存输出，可以理解为：原始数据 经过 神经网络后 得出的 更高维度的特征，供后续xgboost分类使用
np.save(config.MODEL_PATH + 'pree_nnet_r.npy', pred_nnet_r)
np.save(config.MODEL_PATH + 'pred_nnet_v.npy', pred_nnet_v)
np.save(config.MODEL_PATH + 'pred_nnet_t.npy', pred_nnet_t)
