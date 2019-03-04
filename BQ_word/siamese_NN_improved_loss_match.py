# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, Input, Lambda,LSTM,add,dot,Dense,Activation,subtract,Add,multiply,concatenate,merge,Dropout,BatchNormalization,multiply,Bidirectional
from keras.models import Model,Sequential
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam,Adadelta
from keras.preprocessing.sequence import pad_sequences
import data_helper
from attention import AttentionWithContext

input_dim = data_helper.MAX_SEQUENCE_LENGTH
emb_dim = data_helper.EMB_DIM
model_path = './model/siameselstm.hdf5'
tensorboard_path = './model/ensembling'

embedding_matrix = data_helper.load_pickle('embedding_matrix.h5')

embedding_layer = Embedding(embedding_matrix.shape[0],
                            emb_dim,
                            weights=[embedding_matrix],
                            input_length=input_dim,
                            trainable=False)


def manhattan_distance(vects):
    x, y = vects
    return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))

def distance_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1

def base_network(input_shape):
    input = Input(shape=input_shape)
    p = embedding_layer(input)
    p = Bidirectional(LSTM(400, return_sequences=False, dropout=0.1, recurrent_dropout=0.1,name='f_input'))(p)
    #p = AttentionWithContext()(p)
    return Model(input, p, name='review_base_nn')

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    return precision


def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    recall = c1 / c3

    return recall

def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

margin = 0.7
theta = lambda t: (K.sign(t)+1.)/2.
def loss(y_true, y_pred):
    return  (1 - theta(y_true - margin) * theta(y_pred - margin) 
              - theta(1 - margin - y_true) * theta(1 - margin - y_pred)
              ) *(K.mean(K.square(y_true - y_pred)))
def siamese_model():
    input_shape = (input_dim,)
    
    # Creating LSTM Encoder

    base_net = base_network(input_shape)

    
    # Creating LSTM Encoder layer for frist Sentence
    input_q1 = Input(shape=input_shape, dtype='int32', name='sequence1')
    processed_q1 = base_net([input_q1])
    
    
    # Creating LSTM Encoder layer for Second Sentence
    input_q2 = Input(shape=input_shape, dtype='int32', name='sequence2')
    processed_q2 = base_net([input_q2])
    
    #doing matching
    D1_diff = Lambda(lambda x: K.abs((x[0]-K.mean(x[0],axis=1,keepdims=True)) - (x[1]-K.mean(x[1],axis=1,keepdims=True))))([processed_q1,processed_q2])
    D2_diff = Lambda(lambda x: K.abs((x[0]-K.mean(x[1],axis=1,keepdims=True)) + (x[1]-K.mean(x[0],axis=1,keepdims=True))))([processed_q1,processed_q2])


    all_diff = concatenate([D1_diff,D2_diff])
    
    all_diff = Dropout(0.5)(all_diff)

    similarity = Dense(600)(all_diff)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('relu')(similarity)
    similarity = Dense(600)(similarity)
    similarity = Dropout(0.5)(similarity)
    similarity = Activation('relu')(similarity)
    similarity = Dense(1)(similarity)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('sigmoid')(similarity)
    
    distance = Lambda(manhattan_distance, output_shape=distance_output_shape, name='distance')([processed_q1, processed_q2])
    model = Model([input_q1, input_q2], [distance])
    #model = Model([input_q1, input_q2], [similarity])
    #loss:binary_crossentropy,contrastive_loss;optimizer:adm,Adadelta
    adm = Adam(lr=0.002)
    model.compile(loss=mse_loss, optimizer=adm, metrics=['accuracy',precision, recall, f1_score])
    return model


def train():
    
    data = data_helper.load_pickle('model_data.h5')

    train_q1 = data['train_q1']
    train_q2 = data['train_q2']
    train_y = data['train_label']

    dev_q1 = data['dev_q1']
    dev_q2 = data['dev_q2']
    dev_y = data['dev_label']
    
    test_q1 = data['test_q1']
    test_q2 = data['test_q2']
    test_y = data['test_label']
    
    model = siamese_model()
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir=tensorboard_path)    
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max')
    callbackslist = [checkpoint, tensorboard,earlystopping,reduce_lr]

    history = model.fit([train_q1, train_q2], train_y,
              batch_size=512,
              epochs=200,
              validation_data=([dev_q1, dev_q2], dev_y),
              callbacks=callbackslist)

    loss, accuracy, precision, recall, f1_score = model.evaluate([test_q1, test_q2],test_y,verbose=1,batch_size=512)
    print("Test best model word=loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (loss, accuracy, precision, recall, f1_score))

if __name__ == '__main__':
    train()
    
 