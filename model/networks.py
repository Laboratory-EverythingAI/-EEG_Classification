# -*- coding: utf-8 -*-
"""

# CPSC_model.py:深度学习网络模型和人工HRV特征提取

"""
import warnings
import numpy as np
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, AveragePooling1D, Dense, Conv2D
from tensorflow.keras.layers import Dropout, Concatenate, Flatten, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Reshape,Bidirectional
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from biosppy.signals import ecg
from pyentrp import entropy as ent
from torch.nn.utils.spectral_norm import SpectralNorm

import CPSC_utils as utils

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.python.keras import activations
# from tensorflow.python.keras import constraints
# from tensorflow.python.keras import initializers
# from tensorflow.python.keras import regularizers

warnings.filterwarnings("ignore")


class Net(object):


    def __init__(self):
            pass

    @staticmethod
    def __slice(x, index):
        return x[:, :, index]

    @staticmethod
    def __backbone(inp, C=0.001, initial='he_normal'):
        """
        # 用于信号片段特征学习的卷积层组合
        :param inp:  keras tensor, 单个信号切片输入
        :param C:   double, 正则化系数， 默认0.001
        :param initial:  str, 初始化方式， 默认he_normal
        :return: keras tensor, 单个信号切片经过卷积层后的输出
        """
        net = Conv1D(4, 31, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(inp)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(8, 11, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(8, 7, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(16, 5, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(int(net.shape[1]), int(net.shape[1]))(net)

        return net

    @staticmethod
    def nnet(inputs, keep_prob, num_classes):
        """
        # 适用于单导联的深度网络模型
        :param inputs: keras tensor, 切片并堆叠后的单导联信号.
        :param keep_prob: float, dropout-随机片段屏蔽概率.
        :param num_classes: int, 目标类别数.
        :return: keras tensor， 各类概率及全连接层前自动提取的特征.
        """
        branches = []
        for i in range(int(inputs.shape[-1])):
            ld = Lambda(Net.__slice, output_shape=(int(inputs.shape[1]), 1), arguments={'index': i})(inputs)
            ld = Reshape((int(inputs.shape[1]), 1))(ld)
            bch = Net.__backbone(ld)
            branches.append(bch)
        features = Concatenate(axis=1)(branches)
        features = Dropout(keep_prob, [1, int(inputs.shape[-1]), 1])(features)
        # features = Bidirectional(CuDNNLSTM(1, return_sequences=True), merge_mode='concat')(features)
        features = Flatten()(features)
        net = Dense(units=num_classes, activation='softmax')(features)
        return net, features


# class GraphAttentionLayer(keras.layers.Layer):
#     def compute_output_signature(self, input_signature):
#         pass
#
#     def __init__(self,
#                  input_dim,
#                  output_dim,
#                  adj,
#                  nodes_num,
#                  dropout_rate=0.0,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  coef_dropout=0.0,
#                  **kwargs):
#         """
#         :param input_dim: 输入的维度
#         :param output_dim: 输出的维度，不等于input_dim
#         :param adj: 具有自环的tuple类型的邻接表[coords, values, shape]， 可以采用sp.coo_matrix生成
#         :param nodes_num: 点数量
#         :param dropout_rate: 丢弃率，防过拟合，默认0.5
#         :param activation: 激活函数
#         :param use_bias: 偏移，默认True
#         :param kernel_initializer: 权值初始化方法
#         :param bias_initializer: 偏移初始化方法
#         :param kernel_regularizer: 权值正则化
#         :param bias_regularizer: 偏移正则化
#         :param activity_regularizer: 输出正则化
#         :param kernel_constraint: 权值约束
#         :param bias_constraint: 偏移约束
#         :param coef_dropout: 互相关系数丢弃，默认0.0
#         :param kwargs:
#         """
#         super(GraphAttentionLayer, self).__init__()
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.support = [tf.SparseTensor(indices=adj[0][0], values=adj[0][1], dense_shape=adj[0][2])]
#         self.dropout_rate = dropout_rate
#         self.coef_drop = coef_dropout
#         self.nodes_num = nodes_num
#         self.kernel = None
#         self.mapping = None
#         self.bias = None
#
#     def build(self, input_shape):
#         """
#         只执行一次
#         """
#         self.kernel = self.add_weight(shape=(self.input_dim, self.output_dim),
#                                       initializer=self.kernel_initializer,
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint,
#                                       trainable=True)
#
#         if self.use_bias:
#             self.bias = self.add_weight(shape=(self.nodes_num, self.output_dim),
#                                         initializer=self.kernel_initializer,
#                                         regularizer=self.kernel_regularizer,
#                                         constraint=self.kernel_constraint,
#                                         trainable=True)
#         print('[GAT LAYER]: GAT W & b built.')
#
#     def call(self, inputs, training=True):
#         # 完成输入到输出的映射关系
#         # inputs = tf.nn.l2_normalize(inputs, 1)
#         raw_shape = inputs.shape
#         inputs = tf.reshape(inputs, shape=(1, raw_shape[0], raw_shape[1]))  # (1, nodes_num, input_dim)
#         mapped_inputs = keras.layers.Conv1D(self.output_dim, 1, use_bias=False)(inputs)  # (1, nodes_num, output_dim)
#         # mapped_inputs = tf.nn.l2_normalize(mapped_inputs)
#
#         sa_1 = keras.layers.Conv1D(1, 1)(mapped_inputs)  # (1, nodes_num, 1)
#         sa_2 = keras.layers.Conv1D(1, 1)(mapped_inputs)  # (1, nodes_num, 1)
#
#         con_sa_1 = tf.reshape(sa_1, shape=(raw_shape[0], 1))  # (nodes_num, 1)
#         con_sa_2 = tf.reshape(sa_2, shape=(raw_shape[0], 1))  # (nodes_num, 1)
#
#         con_sa_1 = tf.cast(self.support[0], dtype=tf.float32) * con_sa_1  # (nodes_num, nodes_num) W_hi
#         con_sa_2 = tf.cast(self.support[0], dtype=tf.float32) * tf.transpose(con_sa_2, [1, 0])  # (nodes_num, nodes_num) W_hj
#
#         weights = tf.sparse.add(con_sa_1, con_sa_2)  # concatenation
#         weights_act = tf.SparseTensor(indices=weights.indices,
#                                       values=tf.nn.leaky_relu(weights.values),
#                                       dense_shape=weights.dense_shape)  # 注意力互相关系数
#         attention = tf.sparse.softmax(weights_act)  # 输出注意力机制
#         inputs = tf.reshape(inputs, shape=raw_shape)
#         if self.coef_drop > 0.0:
#             attention = tf.SparseTensor(indices=attention.indices,
#                                         values=tf.nn.dropout(attention.values, self.coef_dropout),
#                                         dense_shape=attention.dense_shape)
#         if training and self.dropout_rate > 0.0:
#             inputs = tf.nn.dropout(inputs, self.dropout_rate)
#         if not training:
#             print("[GAT LAYER]: GAT not training now.")
#
#         attention = tf.sparse.reshape(attention, shape=[self.nodes_num, self.nodes_num])
#         value = tf.matmul(inputs, self.kernel)
#         value = tf.sparse.sparse_dense_matmul(attention, value)
#
#         if self.use_bias:
#             ret = tf.add(value, self.bias)
#         else:
#             ret = tf.reshape(value, (raw_shape[0], self.output_dim))
#         return self.activation(ret)
#
# class SelfAttention():
#     def __init__(self):
#         super(SelfAttention, self).__init__()
#
#     def build(self, input_shape):
#         n, h, w, c = input_shape
#         self.n_feats = h * w
#         self.conv_theta = Conv2D(c // 8, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_Theta')
#         self.conv_phi = Conv2D(c // 8, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_Phi')
#         self.conv_g = Conv2D(c // 8, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_g')
#         self.conv_attn_g = Conv2D(c // 8, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_AttnG')
#         self.sigma = self.add_weight(shape=[1], initializer='zeros', trainable=True, name='sigma')
#
#
#     def call(self, x):
#         n, h, w, c = x.shape
#         theta = self.conv_theta(x)
#         theta = tf.reshape(theta, (-1, self.n_feats, theta.shape[-1]))
#         phi = self.conv_phi(x)
#         phi = tf.nn.max_pool2d(phi, ksize=2, strides=2, padding='VALID')
#         phi = tf.reshape(phi, (-1, self.n_feats//4, phi.shape[-1]))
#         g = self.conv_g(x)
#         g = tf.nn.max_pool2d(g, ksize=2, strides=2, padding='VALID')
#         g = tf.reshape(g, (-1, self.n_feats//4, g.shape[-1]))
#         attn_g = tf.matmul(g)
#         attn_g = tf.reshape(attn_g, (-1, h, w, attn_g.shape[-1]))
#         attn_g = self.conv_attn_g(attn_g)
#         output = x + self.sigma * attn_g
#         return output

class ManFeat_HRV(object):
    """
        针对一条记录的HRV特征提取， 以II导联为基准
    """
    FEAT_DIMENSION = 9

    def __init__(self, sig, fs=250.0):
        assert len(sig.shape) == 1, 'The signal must be 1-dimension.'
        assert sig.shape[0] >= fs * 6, 'The signal must >= 6 seconds.'
        self.sig = utils.WTfilt_1d(sig)
        self.fs = fs
        self.rpeaks, = ecg.hamilton_segmenter(signal=self.sig, sampling_rate=self.fs)
        self.rpeaks, = ecg.correct_rpeaks(signal=self.sig, rpeaks=self.rpeaks,
                                         sampling_rate=self.fs)
        self.RR_intervals = np.diff(self.rpeaks)
        self.dRR = np.diff(self.RR_intervals)

    def __get_sdnn(self):  # 计算RR间期标准差
        return np.array([np.std(self.RR_intervals)])

    def __get_maxRR(self):  # 计算最大RR间期
        return np.array([np.max(self.RR_intervals)])

    def __get_minRR(self):  # 计算最小RR间期
        return np.array([np.min(self.RR_intervals)])

    def __get_meanRR(self):  # 计算平均RR间期
        return np.array([np.mean(self.RR_intervals)])

    def __get_Rdensity(self):  # 计算R波密度
        return np.array([(self.RR_intervals.shape[0] + 1) 
                         / self.sig.shape[0] * self.fs])

    def __get_pNN50(self):  # 计算pNN50
        return np.array([self.dRR[self.dRR >= self.fs*0.05].shape[0] 
                         / self.RR_intervals.shape[0]])

    def __get_RMSSD(self):  # 计算RMSSD
        return np.array([np.sqrt(np.mean(self.dRR*self.dRR))])
    
    def __get_SampEn(self):  # 计算RR间期采样熵
        sampEn = ent.sample_entropy(self.RR_intervals, 
                                  2, 0.2 * np.std(self.RR_intervals))
        for i in range(len(sampEn)):
            if np.isnan(sampEn[i]):
                sampEn[i] = -2
            if np.isinf(sampEn[i]):
                sampEn[i] = -1
        return sampEn

    def extract_features(self):  # 提取HRV所有特征
        features = np.concatenate((self.__get_sdnn(),
                self.__get_maxRR(),
                self.__get_minRR(),
                self.__get_meanRR(),
                self.__get_Rdensity(),
                self.__get_pNN50(),
                self.__get_RMSSD(),
                self.__get_SampEn(),
                ))
        assert features.shape[0] == ManFeat_HRV.FEAT_DIMENSION
        return features

