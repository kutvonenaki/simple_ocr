from tensorflow.keras.layers import Dense, Conv2D, TimeDistributed, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional


def downsample_block(input_img):
    """CNN downsampling block and flattening to time distributed layer from
    the original image of shape (width, height, 3) will be mapped to (width/4, 128)"""

    #CNN block
    x = Conv2D(8, 3, activation="relu", padding="same")(input_img)
    x = Conv2D(16, 3, activation="relu", padding="same")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, 3, activation="relu", padding="same")(x)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    x = MaxPooling2D()(x)

    # flatten the height and channels to one dimension, after this the dimension
    # is batch, width, len(height)*len(channels)
    tdist = TimeDistributed(Flatten(), name='timedistrib')(x)

    # map to encoding dimension for rnn (or just next dense if rnn skipped):
    rnn_in = Dense(128, activation="relu", name='dense_in')(tdist)

    return rnn_in

def LSTM_encoding_block(rnn_in):
    """run the inputs through lstm blocks"""

    # lstm encoding layers
    x = Bidirectional(LSTM(64, return_sequences=True))(rnn_in)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    return x


def make_standard_CRNN(img_h, num_chars, use_lstm=False):
    """Construct a standard type CRNN
    Inputs
    img_h: input image height"""

    #note the reverse order here to normal height, width, channels. 
    #it's because the widht of the image will become the time axis
    # and doesn't effect the trained weight parameters
    input_img = Input(name='the_input', shape=(None, img_h, 3))

    #extract features and downsample with CNN layers
    rnn_in = downsample_block(input_img)

    # in case of having actual words:
    # run inputs through LSTM for the language model, to correlate symbols at different positions
    if use_lstm:
        encoded = LSTM_encoding_block(rnn_in)

    # if the input data is only random strings, such a series of numbers, there's no
    # language model to be learned and the lstm layers can be skipped
    else:
         encoded = rnn_in

    #the +1 stands for a blank character needed in CTC
    y_pred = Dense(num_chars+1, name="predictions", activation='softmax')(encoded)

    #construct the model and return
    model = Model(inputs=input_img, outputs=y_pred)

    return model