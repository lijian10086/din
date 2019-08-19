# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 8:43
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : model.py
# @Software: PyCharm

import random
import numpy as np
import tensorflow as tf
# import tensorflow.python.keras as keras
# import tensorflow.python.keras.backend as K

from tensorflow import keras as keras

from tensorflow import keras

# import tensorflow.keras.backend as K
from keras import backend as K

# 设置随机种子，方便复现
seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


class Attention(keras.layers.Layer):
    def __init__(self, attention_hidden_units=(80, 40, 1), attention_activation="sigmoid", supports_masking=True):
        super(Attention, self).__init__()
        self.attention_hidden_units = attention_hidden_units
        self.attention_activation = attention_activation
        self.supports_masking = supports_masking

    def build(self, input_shape):
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):  ##把【（10,128），(10,max_len,128)，(10,1)】 —>(batch_size, hidden_units)=(10,128)
        '''
        i_emb:     [Batch_size, Hidden_units]
        hist_emb:        [Batch_size, max_len, Hidden_units]
        hist_len: [Batch_size]
        '''
        assert len(x) == 3  #Attention()([i_emb, hist_emb, hist_len])

        i_emb, hist_emb, hist_len = x[0], x[1], x[2] #（10,128），(10,max_len,128)，(10,1)
        hidden_units = K.int_shape(hist_emb)[-1]
        max_len = tf.shape(hist_emb)[1]

        i_emb = tf.tile(i_emb, [1, max_len])  # 生成新维度的tensor, [1, max_len]表示i_emb的第一维保持不变，第二维扩展为原来的max_len倍。（10,128）—>(batch_size, max_len * hidden_units)=(10,128*max_len)
        i_emb = tf.reshape(i_emb, [-1, max_len, hidden_units]) #(10,128*max_len)—>重构为 (batch_size, max_len, hidden_units)=(10,max_len,128) 相当于把候选item特征复制max_len份
        concat = K.concatenate([i_emb, hist_emb, i_emb - hist_emb, i_emb * hist_emb],
                               axis=2)  #对候选item向量和序列item向量进行特征融合 (10,max_len,128) —> # (batch_size, max_len, hidden_units * 4)=(10,max_len,512)

        for i in range(len(self.attention_hidden_units)): #(10,max_len,512)->经过dense(80,"sigmoid")=(10,max_len,80)->经过dense(40,"sigmoid")=(10,max_len,40)->经过dense(1)=(10,max_len,1)
            activation = None if i == 2 else self.attention_activation
            outputs = keras.layers.Dense(self.attention_hidden_units[i], activation=activation)(concat)
            concat = outputs

        outputs = tf.reshape(outputs, [-1, 1, max_len])#(10,max_len,1)—> 重构为(batch_size, 1, max_len)=(10,1,max_len)

        if self.supports_masking:
            mask = tf.sequence_mask(hist_len, max_len)  #以（10个item序列长度）—>生成10个每行元素值=ture或false的张量且true个数为item序列长度的max_len张量=(batch_size, 1, max_len)
            padding = tf.ones_like(outputs) * (-1e12)  #创建一个用于覆盖填充outputs，元素值为很大很大的值张量=(batch_size, 1, max_len) 注意：(-1e12) 也可换为(-2 ** 32 + 1)
            outputs = tf.where(mask, outputs, padding) #对outputs进行覆盖，如果mask元素=false,则用padding对应元素（经过后面的sofmax以后，元素值约等于0）覆盖。

        # 对outputs进行scale
        outputs = outputs / (hidden_units ** 0.5)
        outputs = K.softmax(outputs)   #对权重向量元素进行缩放，被padding的元素值约等于0


        outputs = tf.matmul(outputs, hist_emb)  # 元素权重向量和历史item序列向量进行元素点乘,使得每个item的所有向量元素乘以相同的权重后得以weight-sum。 (10,1,max_len) *(10,max_len,128)—>（batch_size, 1, hidden_units)=(10,1,128)

        outputs = tf.squeeze(outputs)  # 降维度(10,1,128)—>(batch_size, hidden_units)=(10,128)

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


def share_weights(hidden_units=63930):
    '''
    reuse a group of keras layers(封装多层，同时可以共享)
    '''
    layers_units = (80, 40, 1)
    share_input = keras.layers.Input(shape=(hidden_units, )) #(10,63930)
    share_layer = share_input
    for i in range(len(layers_units)):
        activation = None if i == 2 else "sigmoid"
        share_layer = keras.layers.Dense(layers_units[i], activation=activation)(share_layer)
    out_layer = share_layer #(10,63930)—》(10,80)—》(10,40)—》(10,1）
    model = keras.models.Model(share_input, out_layer)
    return model


def din(item_count, cate_count, hidden_units=128): #din(item_count=63001, cate_count=801, hidden_units=128) #若batchsize=10
    '''
    :param item_count: 商品数
    :param cate_count: 类别数
    :param hidden_units: 隐藏单元数
    :return: model
    '''
    target_item = keras.layers.Input(shape=(1,), name='target_item', dtype="int32")  # 点击的item =（10,1）
    target_cate = keras.layers.Input(shape=(1,), name='target_cate', dtype="int32")  # 点击的item对应的所属类别 =（10,1）
    label = keras.layers.Input(shape=(1,), name='label', dtype="float32")  # 是否点击 =（10,1）

    hist_item_seq = keras.layers.Input(shape=(None,), name="hist_item_seq", dtype="int32")  # 点击item序列 =（10,?）
    hist_cate_seq = keras.layers.Input(shape=(None,), name="hist_cate_seq", dtype="int32")  # 点击item序列对应的类别序列 =（10,?）

    hist_len = keras.layers.Input(shape=(1,), name='hist_len', dtype="int32")  # item序列本来的长度

    item_emb = keras.layers.Embedding(input_dim=item_count,
                                      output_dim=hidden_units // 2,
                                      embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1e-4,
                                                                                             seed=seed)) #emb层。利用w=(63001,64)，把（10,1)—>（10,63001）->(10,64)
    cate_emb = keras.layers.Embedding(input_dim=cate_count,
                                      output_dim=hidden_units // 2,
                                      embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1e-4,
                                                                                             seed=seed)) #emb层。利用w=(801,64)，把（10,1)—>（10,801）->(10,64)
    item_b = keras.layers.Embedding(input_dim=item_count, output_dim=1,
                                    embeddings_initializer=keras.initializers.Constant(0.0))#emb层。利用w=(63001,1)，把（10,1)—>（10,63001）->(10,1)

    # get target bias embedding
    target_item_bias_emb = item_b(target_item)  #把（10,63001）->(batch_size,1, 1)=(10,1,1)
    #
    target_item_bias_emb = keras.layers.Lambda(lambda x: K.squeeze(x, axis=1))(target_item_bias_emb) # 把(10,1,1)——>(10,1)

    # get target embedding
    target_item_emb = item_emb(target_item)  # 把（10,1)—>（10,63001）->(batch_size,1,hidden_units//2)=(10,1,64)
    target_cate_emb = cate_emb(target_cate)  # 把（10,1)—>（10,801）->(batch_size,1 ,hidden_units//2)=(10,1,64)
    i_emb = keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))(
        [target_item_emb, target_cate_emb])  # 把商品(10,1,164)和类别(10,1,64)—>融合向量(batch_size,1, hidden_units)=(10,1,128)
    i_emb = keras.layers.Lambda(lambda x: K.squeeze(x, axis=1))(i_emb)  #(batch_size,1, hidden_units)=(10,1,128)—> 降维(batch_size, hidden_units)=(10,128)

    # get history item embedding
    hist_item_emb = item_emb(hist_item_seq)  #把（batch_size, items_len）=(10,items_len)—>（10,items_len,63001）—>(batch_size, max_len, hidden_units//2)=(10,max_len,64)
    hist_cate_emb = cate_emb(hist_cate_seq)  #把（batch_size, cates_len）=(10,cates_len)—>（10,items_len,801）—>(batch_size, max_len, hidden_units//2)=(10,max_len,64)
    hist_emb = keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))(
        [hist_item_emb, hist_cate_emb])  # # 把商品(10,1,164)和类别(10,1,64)—>融合向量(batch_size, max_len, hidden_units)=(10,max_len,128)

    # 构建点击序列与候选的attention关系
    din_attention = Attention()([i_emb, hist_emb, hist_len])  #把【（10,128），(10,max_len,128)，(10,1)】 —>weight sum得到用户兴趣向量(batch_size, hidden_units)=(10,128)
    din_attention = keras.layers.Lambda(lambda x: tf.reshape(x, [-1, hidden_units]))(din_attention) #确保维度(10,128)—>(10,128)

    # keras.layers.BatchNormalization实现暂时有坑，借用paddle相关代码实现
    din_attention_fc = keras.layers.Dense(63802)(din_attention)  # 用户兴趣向量(10,128)—>(batch_size, item_count + cate_count)=(10,63802)？？不对吧。应该 用户兴趣向量(10,128)concatenate候选item向量（10,128）得到总向量（10,256）—>经过fc(80)-fc(40)-fc(1)得到点击率
    # item_count:  63001   cate_count:  801         hidden_units:  128   (batch_size, item_count + cate_count + hidden_units)
    din_item = keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([i_emb, din_attention_fc]) #(10,128)+(10,63802)->(10,63930)
    din_item = share_weights()(din_item)  # (10,63930)—》(batch_size, 1)=(10,1)

    print("logits:", din_item, target_item_bias_emb)
    logits = keras.layers.Add()([din_item, target_item_bias_emb]) #(10,1)+(10,1)—》加层得偏置=(10,1) 得点击概率

    label_model = keras.models.Model(inputs=[hist_item_seq, hist_cate_seq, target_item, target_cate, hist_len], outputs=[logits])

    train_model = keras.models.Model(inputs=[hist_item_seq, hist_cate_seq, target_item, target_cate, hist_len, label],
                               outputs=logits) #输入：历史商品序列，历史商品类别序列，候选商品，候选商品类别，序列长度，标签    输出：点击概率

    # 计算损失函数
    loss = K.binary_crossentropy(target=label, output=logits, from_logits=True)
    train_model.add_loss(loss)
    train_model.compile(optimizer=keras.optimizers.SGD(1e-3), metrics=["accuracy"])

    return train_model, label_model

if __name__ == "__main__":
    # 构造模型
    train_model, label_model = din(item_count=63001, cate_count=801, hidden_units=128)
    print('123')
