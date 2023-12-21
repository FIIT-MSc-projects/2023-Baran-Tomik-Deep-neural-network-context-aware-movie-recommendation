# import tensorflow as tf
from keras.layers import Activation, Dense, Dropout, LSTM, Bidirectional, Flatten
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.losses import MeanSquaredError as LossMSE
from keras.losses import MeanAbsoluteError as LossMAE
from keras.losses import CosineSimilarity as LossCosSim
from keras.losses import MeanSquaredLogarithmicError as LossLog
from keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError


def create_model(architecture=None, lr=0.001, train_data_shape=None):
    if architecture == 0:
        print("Model 0")
        return model_0(lr=lr, train_data_shape=train_data_shape)
    elif architecture == 1:
        print("Model 1")
        return model_1(lr=lr, train_data_shape=train_data_shape)
    elif architecture == 2:
        print("Model 2")
        return model_2(lr=lr, train_data_shape=train_data_shape)
    elif architecture == 3:
        print("Model 3")
        return model_3(lr=lr, train_data_shape=train_data_shape)
    elif architecture == 4:
        print("Model 4")
        return model_4(lr=lr, train_data_shape=train_data_shape)
    else:
        return None


def model_0(lr=0.001, train_data_shape=None):
    model = Sequential()
    model.add(Dense(64, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.2))

    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.2))

    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Dense(16))
    model.add(Activation('relu'))

    model.add(Dense(8))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.compile(optimizer=Adam(), loss=LossMSE(), metrics=[MeanSquaredError(), MeanAbsoluteError()])

    return model

def model_1(lr, train_data_shape=None):
    model = Sequential()
    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))

    # model.compile(optimizer=Adam(), loss=LossListMLE(), metrics=[MeanAbsoluteError(), RootMeanSquaredError(), NDCGMetric(name="ndcg_metric")])
    model.compile(optimizer=Adam(), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    return model


def model_2(lr=0.001, train_data_shape=None):
    model = Sequential()

    # b4 LSTM data, input_shape = (train_data_shape,)
    model.add(LSTM(256, input_shape=(train_data_shape), return_sequences=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(LSTM(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(optimizer=Adam(learning_rate=lr), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    return model

def model_3(lr=0.001, train_data_shape=None):
    model = Sequential()

    # b4 LSTM data, input_shape = (train_data_shape,)
    model.add(Bidirectional(LSTM(128), input_shape=(train_data_shape)))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(optimizer=Adam(learning_rate=lr), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    return model


def model_4(lr, train_data_shape=None):
    model = Sequential()
    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    # model.compile(optimizer=Adam(), loss=LossListMLE(), metrics=[MeanAbsoluteError(), RootMeanSquaredError(), NDCGMetric(name="ndcg_metric")])
    model.compile(optimizer=Adam(), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    return model






# def create_model_run2(lr=0.001):
#     model = Sequential([
#         Dense(32, input_shape=(13,)),
#         Activation('relu'),

#         Dense(16, input_shape=(32,)),
#         Activation('relu'),
#         Dense(8, input_shape=(16,)),
#         Activation('relu'),

#         Dense(1, input_shape=(8,))
#     ])
#     model.compile(optimizer=Adam(learning_rate=lr),
#                   loss=LossMSE(),
#                   metrics=[MeanSquaredError(), MeanAbsoluteError()])
#                                                                                                     # tfa.metrics.RSquare()
#     return model


# def create_model_run3(lr=0.001):
#     model = Sequential([
#         Dense(128, input_shape=(13,)),
#         Activation('relu'),

#         Dense(32, input_shape=(128,)),
#         Activation('relu'),
#         Dense(8, input_shape=(32,)),
#         Activation('relu'),

#         Dense(1, input_shape=(8,))
#     ])
#     model.compile(optimizer=Adam(learning_rate=lr),
#                   loss=LossMSE(),
#                   metrics=[MeanSquaredError(), MeanAbsoluteError()])
#                                                                                                     # tfa.metrics.RSquare()
#     return model


# def create_model_run4(lr=0.001):
#     model = Sequential([
#         Dense(32, input_shape=(13,)),
#         Activation('relu'),

#         Dense(64, input_shape=(32,)),
#         Activation('relu'),
#         Dense(64, input_shape=(64,)),
#         Activation('relu'),
#         Dense(128, input_shape=(64,)),
#         Activation('relu'),
#         Dense(64, input_shape=(128,)),
#         Activation('relu'),
#         Dense(32, input_shape=(64,)),
#         Activation('relu'),
#         Dense(16, input_shape=(32,)),
#         Activation('relu'),
#         Dense(8, input_shape=(16,)),
#         Activation('relu'),

#         Dense(1, input_shape=(8,))
#     ])
#     model.compile(optimizer=Adam(learning_rate=lr),
#                   loss=LossMSE(),
#                   metrics=[MeanSquaredError(), MeanAbsoluteError()])
#                                                                                                     # tfa.metrics.RSquare()
#     return model


##--------------------------------------------------------------------------------------------------------------------##

# def create_model_run5(lr=0.001):
#     model = Sequential([
#         Dense(128, input_shape=(13,)),
#         Activation('leaky_relu'),

#         Dense(32, input_shape=(128,)),
#         Activation('leaky_relu'),
#         Dense(8, input_shape=(32,)),
#         Activation('leaky_relu'),

#         Dense(1, input_shape=(8,))
#     ])
#     model.compile(optimizer=Adam(learning_rate=lr),
#                   loss=LossMSE(),
#                   metrics=[MeanSquaredError(), MeanAbsoluteError()])
#                                                                                                     # tfa.metrics.RSquare()
#     return model


# # # run6 (0.6), run7 (0.2), 
# def create_model_run6(lr=0.001):
#     model = Sequential([
#         Dense(128, input_shape=(13,)),
#         Activation('relu'),
#         Dropout(0.2),

#         Dense(32, input_shape=(128,)),
#         Activation('relu'),
#         Dense(8, input_shape=(32,)),
#         Activation('relu'),

#         Dense(1, input_shape=(8,))
#     ])
#     model.compile(optimizer=Adam(learning_rate=lr),
#                   loss=LossMSE(),
#                   metrics=[MeanSquaredError(), MeanAbsoluteError()])
#                                                                                                     # tfa.metrics.RSquare()
#     return model

# def create_model_run8(lr=0.001):
#     model = Sequential([
#         Dense(128, input_shape=(13,)),
#         Activation('relu'),
#         Dropout(0.2),

#         Dense(32, input_shape=(128,)),
#         Activation('relu'),
#         Dropout(0.2),
#         Dense(8, input_shape=(32,)),
#         Activation('relu'),
#         Dropout(0.2),

#         Dense(1, input_shape=(8,))
#     ])
#     model.compile(optimizer=Adam(learning_rate=lr),
#                   loss=LossMSE(),
#                   metrics=[MeanSquaredError(), MeanAbsoluteError()])
#                                                                                                     # tfa.metrics.RSquare()
#     return model


# def create_model_run(lr=0.001):
#     model = Sequential([
#         Dense(128, input_shape=(13,)),
#         Activation('relu'),

#         Dense(32, input_shape=(128,)),
#         Activation('relu'),
#         Dense(8, input_shape=(32,)),
#         Activation('relu'),

#         Dense(1, input_shape=(8,))
#     ])
#     model.compile(optimizer=RMSprop(learning_rate=lr),
#                   loss=LossMSE(),
#                   metrics=[MeanSquaredError(), MeanAbsoluteError()])
#                                                                                                     # tfa.metrics.RSquare()
#     return model


# def create_model_run(lr=0.001):
#     model = Sequential([
#         Dense(128, input_shape=(13,)),
#         Activation('relu'),

#         Dense(32, input_shape=(128,)),
#         Activation('relu'),
#         Dense(8, input_shape=(32,)),
#         Activation('relu'),

#         Dense(1, input_shape=(8,))
#     ])
#     model.compile(optimizer=Adadelta(learning_rate=lr),
#                   loss=LossMSE(),
#                   metrics=[MeanSquaredError(), MeanAbsoluteError()])
#                                                                                                     # tfa.metrics.RSquare()
#     return model

# def create_model_run(lr=0.001):
#     model = Sequential([
#         Dense(128, input_shape=(13,)),
#         Activation('relu'),

#         Dense(32, input_shape=(128,)),
#         Activation('relu'),
#         Dense(8, input_shape=(32,)),
#         Activation('relu'),

#         Dense(1, input_shape=(8,))
#     ])
#     model.compile(optimizer=Adam(learning_rate=lr),
#                   loss=LossMAE(),
#                   metrics=[MeanSquaredError(), MeanAbsoluteError()])
#                                                                                                     # tfa.metrics.RSquare()
#     return model

# def create_model_run(lr=0.001):
#     model = Sequential([
#         Dense(128, input_shape=(13,)),
#         Activation('relu'),

#         Dense(32, input_shape=(128,)),
#         Activation('relu'),
#         Dense(8, input_shape=(32,)),
#         Activation('relu'),

#         Dense(1, input_shape=(8,))
#     ])
#     model.compile(optimizer=Adam(learning_rate=lr),
#                   loss=LossCosSim(),
#                   metrics=[MeanSquaredError(), MeanAbsoluteError()])
#                                                                                                     # tfa.metrics.RSquare()
#     return model

# def create_model_run(lr=0.001):
#     model = Sequential([
#         Dense(128, input_shape=(13,)),
#         Activation('relu'),

#         Dense(32, input_shape=(128,)),
#         Activation('relu'),
#         Dense(8, input_shape=(32,)),
#         Activation('relu'),

#         Dense(1, input_shape=(8,))
#     ])
#     model.compile(optimizer=Adam(learning_rate=lr),
#                   loss=LossLog(),
#                   metrics=[MeanSquaredError(), MeanAbsoluteError()])
#                                                                                                     # tfa.metrics.RSquare()
#     return model
