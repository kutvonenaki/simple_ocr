from trdg.generators import GeneratorFromRandom
import ocr_source.batch_functions as batch_functions
import ocr_source.models as models
import ocr_source.custom_callbacks as custom_callbacks
import ocr_source.inference as inference
import ocr_source.losses as losses

import time
import importlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def train():
    """main training function"""

    # *********** MAGIC LINES ****************
    # you might need this if training crashes due GPU memory overload
    # or you get CuDNN load failure

    #check for gpu
    print(tf.config.list_physical_devices('GPU'))

    #for tf2 magic lines to prevent razer from crashing
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)



    #*************  PARAMETERS *******************
    batch_size = 12
    img_h = 32
    num_epochs = 10

    # list of all characters
    # map each color to an integer, a "label" and reverse mapping
    all_chars = "0123456789"
    num_chars = len(all_chars)
    char_to_lbl_dict = dict((char, ind) for ind, char in enumerate(all_chars))
    lbl_to_char_dict = dict((ind, char) for ind, char in enumerate(all_chars))


    # ************** DATA GENERATORS *********************
    #use the trdg for the base generator of text
    base_generator = GeneratorFromRandom(use_symbols=False, use_letters=False, background_type=1)

    #add some more augmentation with keras ImageDataGenerator
    keras_augm = ImageDataGenerator(rotation_range=2.0, width_shift_range=5.0, height_shift_range=5.0,
                                    shear_range=4.0, zoom_range=0.1)

    #the actual datagenerator for training and visualizations (and validation)
    dg_params = {"batch_size": batch_size,"img_h": img_h, "keras_augmentor": keras_augm,
                "char_to_lbl_dict": char_to_lbl_dict}

    datagen = batch_functions.OCR_generator(base_generator, **dg_params)
    val_datagen = batch_functions.OCR_generator(base_generator, **dg_params, validation=True)


    #*******MODEL******
    model = models.make_standard_CRNN(img_h, num_chars)


    #********CALLBACKS AND LOSSES****************
    # get the cool outputs
    predvis = custom_callbacks.PredVisualize(model,val_datagen, lbl_to_char_dict, printing=True)
    model_saver = custom_callbacks.make_save_model_cb()
    custom_loss = losses.custom_ctc()


    #********COMPILE, SAVE MODEL**************
    model.compile(loss=custom_loss, optimizer="Adam")
    tf.keras.models.save_model(model, "saved_models", overwrite=True, include_optimizer=False)

    H = model.fit(datagen, epochs=num_epochs, verbose=1, callbacks=[predvis, model_saver])


if __name__ == '__main__':
    train()