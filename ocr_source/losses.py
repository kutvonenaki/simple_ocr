from tensorflow.keras.backend import ctc_batch_cost
import tensorflow as tf

def custom_ctc():
    """Custom CTC loss implementation"""

    def loss(y_true, y_pred):
        """Why you make it so complicated?
        
        Since the prediction from models is (batch, timedistdim, tot_num_uniq_chars)
        and the true target is labels (batch_size,1) but the ctc loss need some
        additional information of different sizes. And the inputs to loss y_true,
        y_pred must be both same dimensions because of keras.
        
        So I have packed the needed information inside the y_true and just made it
        to a matching dimension with y_true"""

        batch_labels = y_true[:, :, 0]
        label_length = y_true[:, 0, 1]
        input_length = y_true[:, 0, 2]

        #reshape for the loss, add that extra meaningless dimension
        label_length = tf.expand_dims(label_length, -1)
        input_length = tf.expand_dims(input_length, -1)


        return ctc_batch_cost(batch_labels, y_pred, input_length, label_length)
    return loss