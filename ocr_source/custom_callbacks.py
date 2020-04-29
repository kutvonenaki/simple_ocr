import tensorflow as tf
import numpy as np
from IPython.display import display
from PIL import Image
import os

def decode_ctc(args):
    """returns a list of decoded ctc losses"""

    y_pred, input_length = args

    ctc_decoded = tf.keras.backend.ctc_decode(
        y_pred, input_length, greedy=True)

    return ctc_decoded


class PredVisualize(tf.keras.callbacks.Callback):

    def __init__(self, model, val_datagen, lbl_to_char_dict, printing=False):
        """CTC decode the results and visualize output"""
        self.model = model
        self.datagen = iter(val_datagen)
        self.printing = printing
        self.lbl_to_char_dict = lbl_to_char_dict

    def on_epoch_end(self, batch, logs=None):

        #make a batch of data
        batch_imgs, batch_labels, input_length, label_lens = next(self.datagen)

        #predict from batch
        y_preds = self.model.predict(batch_imgs)

        #call the ctc decode
        pred_tensor, _ = decode_ctc([y_preds, np.squeeze(input_length)])
        pred_labels = tf.keras.backend.get_value(pred_tensor[0])

        #map back to strings
        predictions = ["".join([self.lbl_to_char_dict[i] for i in word if i!=-1]) for word in pred_labels.tolist()]
        truths = ["".join([self.lbl_to_char_dict[i] for i in word]) for word in batch_labels.tolist()]

        #combine the images and print at screen if printing is on
        # transpose first back to original form
        if self.printing:
            imgs_list_arr_T = [img.transpose((1,0,2)) for img in batch_imgs]
            imgs_comb = np.hstack(imgs_list_arr_T) * 255
            imgs_comb = Image.fromarray(imgs_comb.astype(np.uint8),'RGB')
            display(imgs_comb)

        print('predictions {}'.format(predictions))

def make_save_model_cb(folder = "saved_models"):
    """save model weights after each epoch callback
        Should really save the whole model but for some reason doesn't work in tf2"""

    filename = "weights.h5"
    filepath = os.path.join(os.getcwd(), folder, filename)

    callback = tf.keras.callbacks.ModelCheckpoint(filepath,
                    save_weights_only=True, verbose=0, save_best_only=False)

    return callback

