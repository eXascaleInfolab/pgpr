from keras.models import Model
from keras.layers import Input, RepeatVector, Permute, Dense, Embedding, Conv1D, MaxPooling1D, Dropout, \
    GlobalMaxPooling1D, Activation, LSTM, merge, Bidirectional, GRU, Flatten, Reshape
from keras.preprocessing.text import Tokenizer
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import concatenate
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.regularizers import l2
from keras.layers import Dropout
import keras
import numpy as np
import random
import torch
seed_val = 13270
random.seed(seed_val)
np.random.seed(int(seed_val / 100))
torch.manual_seed(int(seed_val / 10))
torch.cuda.manual_seed_all(seed_val)

def log_reg(dim):
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(units=1,kernel_initializer='glorot_uniform',
                    activation='sigmoid',
                    kernel_regularizer=l2(0.),
                    input_dim=dim))
    return model

def create_mlp(dim, classification = False):
    # define our MLP network
    np.random.seed(10)
    model = Sequential()
    model.add(Dense(52, input_dim=dim, activation="tanh"))
    model.add(Dense(52, activation="relu"))
    # model.add(Dense(4, activation="relu"))
    model.add(Dropout(0.03))
    if classification:
        model.add(Dense(2, activation='softmax'))
    # return our model
    return model

def mil_model(nbr_filters,MAX_SEQUENCE_LENGTH,MAX_SENTENCE_NUM,grading_classes):
    print('Build model...')
    np.random.seed(10)
    ############# input representation layer #######################
    inputs = Input(shape=(MAX_SENTENCE_NUM, MAX_SEQUENCE_LENGTH))
    ############ Sentence classification layer ##########################
    pi = TimeDistributed(Dense(grading_classes, activation='softmax'))(inputs)
    #We use separate LSTM modules to produce forward and backward hidden vectors, which are then concatenated:
    a = Bidirectional(LSTM(nbr_filters, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(inputs)
    #to measure the importance of each sentence use tanh a one-layer MLP that produces an attention weight ai for the i-th
    # sentence,and Wa and ba are parameters in it
    a = TimeDistributed(Dense(1, activation='tanh'))(a)
    a = Flatten()(a) #A tensor, reshaped into 1-D
    #transform the result into a probability between 0 and 1 using the softmax function
    a = Activation('softmax',name="weights")(a)
    #Permutes the dimensions of the input according to a given pattern. e.g. (2, 1) permutes the first and second dimension of the input.
    pi = Permute((2, 1), name="pi")(pi)
    #we obtain a document-level distribution over class
    #labels as the weighted sum of sentence-level distributions:
    predictions = keras.layers.Dot(axes=(2, 1))([pi, a])
    _model = Model(inputs=inputs, outputs=predictions)
    return _model

def lstm_model(MAX_SENTENCE_NUM, MAX_SEQUENCE_LENGTH):
    np.random.seed(13270)
    inputs = Input(shape=(MAX_SENTENCE_NUM, MAX_SEQUENCE_LENGTH))
    first_lstm = Bidirectional(LSTM(200,dropout=0.5, return_sequences=True)) (inputs)
    sec_lstm = Bidirectional(LSTM(200, dropout=0.5)) (first_lstm)
    dens_lay = Dense(200, activation="relu") (sec_lstm)
    drop_lay = Dropout(0.5)(dens_lay)
    predictions = Dense(2, activation='softmax')(drop_lay)
    model = Model(inputs=inputs, outputs=predictions)
    return model

def scibert_model(MAX_SEQUENCE_LENGTH,MAX_SENTENCE_NUM):
    np.random.seed(10)
    _model =lstm_model(MAX_SENTENCE_NUM, MAX_SEQUENCE_LENGTH)
    opt = keras.optimizers.Adam(learning_rate=1)
    _model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse'])
    return _model

def define_model(nbr_filters,MAX_SEQUENCE_LENGTH,MAX_SENTENCE_NUM,grading_classes,opt):
    seed_val = 100
    random.seed(seed_val)
    np.random.seed(int(seed_val / 100))
    torch.manual_seed(int(seed_val / 10))
    torch.cuda.manual_seed_all(seed_val)
    _model = mil_model(nbr_filters,MAX_SEQUENCE_LENGTH,MAX_SENTENCE_NUM,grading_classes)
    _model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return _model

def mix_data_type(MAX_SEQUENCE_LENGTH,MAX_SENTENCE_NUM,grading_classes,x_train_stats):
    np.random.seed(10)
    model_mlp = log_reg(x_train_stats.shape[1])
    model_att =  mil_model(16,MAX_SEQUENCE_LENGTH,MAX_SENTENCE_NUM,grading_classes)
    combinedInput = concatenate([model_mlp.output, model_att.output])
    x = Dense(10, activation="tanh")(combinedInput)
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=[model_mlp.input, model_att.input], outputs=x)
    opt = keras.optimizers.SGD(learning_rate=1000)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

    return model