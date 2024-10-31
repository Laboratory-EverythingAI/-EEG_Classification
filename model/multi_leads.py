# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 14:41:17 2018

@author: Winham

# CPSC_train_multi_leads.py: 针对每个导联训练网络并保存模型，主体与CPSC_train_single_lead.py基本一致

"""

import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as bk
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from CPSC_model import Net
from CPSC_config import Config
import CPSC_utils as utils
#日志等级ERROR + FATAL
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
config = Config()

#X
records_name = np.array(os.listdir(config.DATA_PATH))
#Y
records_label = np.load(config.REVISED_LABEL) - 1
#几分类
class_num = len(np.unique(records_label))

#划分训练集、测试集，20%作为测试集
train_val_records, test_records, train_val_labels, test_labels = train_test_split(
    records_name, records_label, test_size=0.2, random_state=config.RANDOM_STATE)
del test_records, test_labels

#训练集又划分为：真实训练集和验证集（一般用作调参），上述的测试集用作后续验证模型
train_records, val_records, train_labels, val_labels = train_test_split(
    train_val_records, train_val_labels, test_size=0.2, random_state=config.RANDOM_STATE)

train_records, train_labels = utils.oversample_balance(train_records, train_labels, config.RANDOM_STATE)
val_records, val_labels = utils.oversample_balance(val_records, val_labels, config.RANDOM_STATE)

for i in range(config.LEAD_NUM):
    TARGET_LEAD = i
    print('Fetching data for Lead ' + str(TARGET_LEAD) + ' ...-----------------\n')
    train_x = utils.Fetch_Pats_Lbs_sLead(train_records, Path=config.DATA_PATH,
                                     target_lead=TARGET_LEAD, seg_num=config.SEG_NUM, 
                                     seg_length=config.SEG_LENGTH)
    #独热编码
    train_y = to_categorical(train_labels, num_classes=class_num)
    #特征提取
    val_x = utils.Fetch_Pats_Lbs_sLead(val_records, Path=config.DATA_PATH,
                                     target_lead=TARGET_LEAD, seg_num=config.SEG_NUM, 
                                     seg_length=config.SEG_LENGTH)
    val_y = to_categorical(val_labels, num_classes=class_num)
    
    model_name = 'net_lead_' + str(TARGET_LEAD) + '.hdf5'
    
    print('Scaling data ...-----------------\n')
    #标准化
    for j in range(train_x.shape[0]):
        train_x[j, :, :] = scale(train_x[j, :, :], axis=0)
    for j in range(val_x.shape[0]):
        val_x[j, :, :] = scale(val_x[j, :, :], axis=0)

    batch_size = 64
    epochs = 100
    momentum = 0.9
    keep_prob = 0.5

#清缓存
    bk.clear_session()
    tf.reset_default_graph()

#训练模型
    inputs = Input(shape=(config.SEG_LENGTH, config.SEG_NUM))
    net = Net()
    outputs, _ = net.nnet(inputs, keep_prob, num_classes=class_num)
    model = Model(inputs=inputs, outputs=outputs)
    
    opt = optimizers.SGD(lr=config.lr_schedule(0), momentum=momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    #权重参数
    checkpoint = ModelCheckpoint(filepath=config.MODEL_PATH+model_name,
                                 monitor='val_categorical_accuracy', mode='max',
                                 save_best_only='True')
    #损失函数
    lr_scheduler = LearningRateScheduler(config.lr_schedule)
    callback_lists = [checkpoint, lr_scheduler]
    model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=2,
              validation_data=(val_x, val_y), callbacks=callback_lists)
    
    del train_x, train_y
    #加载模型，用save()
    model = load_model(config.MODEL_PATH + model_name)
    
    pred_vt = model.predict(val_x, batch_size=batch_size, verbose=1)

    #上述独热编码。 0 ， 0 ， 1 ，0 =》 2
    pred_v = np.argmax(pred_vt, axis=1)
    true_v = np.argmax(val_y, axis=1)
    del val_x, val_y

    #混淆矩阵
    Conf_Mat_val = confusion_matrix(true_v, pred_v)
    print('\nResult for Lead ' + str(TARGET_LEAD) + '-----------------------------\n')
    print(Conf_Mat_val)
    F1s_val = []
    #算F1-mean
    for j in range(class_num):
        f1t = 2 * Conf_Mat_val[j][j] / (np.sum(Conf_Mat_val[j, :]) + np.sum(Conf_Mat_val[:, j]))
        print('| F1-' + config.CLASS_NAME[j] + ':' + str(f1t) + ' |')
        F1s_val.append(f1t)
    print('F1-mean: ' + str(np.mean(F1s_val)))
