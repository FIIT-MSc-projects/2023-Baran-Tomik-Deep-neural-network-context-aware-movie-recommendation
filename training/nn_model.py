from keras.layers import Activation, Dense, Dropout, LSTM, Bidirectional, Flatten, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.losses import MeanSquaredError as LossMSE
from keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, R2Score


def create_model(architecture=None, lr=0.001, train_data_shape=None, alpha=0.3):
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
    elif architecture == 5:
        print("Model 5")
        return model_5(lr=lr, train_data_shape=train_data_shape)
    elif architecture == 6:
        print("Model 6")
        return model_6(lr=lr, train_data_shape=train_data_shape)
    elif architecture == 7:
        print("Model 7")
        return model_7(lr=lr, alphaNumber=alpha, train_data_shape=train_data_shape)
    elif architecture == 8:
        print("Model 8")
        return model_8(lr=lr, train_data_shape=train_data_shape)
    elif architecture == 9:
        print("Model 9")
        return model_9(lr=lr, train_data_shape=train_data_shape)
    elif architecture == 10:
        print("Model 10")
        return model_10(lr=lr, train_data_shape=train_data_shape)
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

    model.compile(optimizer=Adam(), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    return model


def model_2(lr=0.001, train_data_shape=None):
    model = Sequential()

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

    model.compile(optimizer=Adam(), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    return model


def model_5(lr, train_data_shape=None):
    model = Sequential()
    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer=Adam(), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError(), R2Score()])

    return model


def model_6(lr, train_data_shape=None):
    model = Sequential()
    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(optimizer=Adam(), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError(), R2Score()])

    return model


def model_7(lr, alphaNumber, train_data_shape=None):
    print("LekyReLU alpha: ", alphaNumber)
    model = Sequential()
    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(LeakyReLU(alpha=alphaNumber))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(LeakyReLU(alpha=alphaNumber))
    model.add(Dropout(0.2))

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(LeakyReLU(alpha=alphaNumber))
    model.add(Dropout(0.2))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(LeakyReLU(alpha=alphaNumber))
    model.add(Dropout(0.2))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(LeakyReLU(alpha=alphaNumber))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer=Adam(), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError(), R2Score()])

    return model


def model_8(lr, train_data_shape=None):
    model = Sequential()
    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(optimizer=Adam(), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError(), R2Score()])

    return model


def model_9(lr, train_data_shape=None):
    model = Sequential()
    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(optimizer=Adam(), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError(), R2Score()])

    return model


def model_10(lr, train_data_shape=None):
    model = Sequential()
    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(128, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))

    model.add(Dense(512, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128, input_shape=(train_data_shape,)))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(optimizer=SGD(), loss=LossMSE(), metrics=[RootMeanSquaredError(), MeanAbsoluteError(), R2Score()])

    return model
