import numpy as np
from PIL import Image
import itertools
from tensorflow.keras.utils import Sequence
import numpy as np



class OCR_generator(Sequence):
    """Generator for the input data to the OCR model. We're also preparing 
    arrays for the CTC loss which are related to the output dimensions"""

    def __init__(self, base_generator, batch_size, char_to_lbl_dict,
                img_h , keras_augmentor, epoch_size=500, validation=False):
        """Inputs
        base_generator: the base trdg generator
        batch_size: number of examples fed to the NN simultaneously
        char_to_lbl_dict: mapping from character to its label (int number)
        img_h: we assume that the input here is already scaled to the correct height
        keras_augmentor: Keras augmentor to add more augmenting, the current base generator doesn'
                for example zoom, translate etc"""
        self.base_generator = base_generator
        self.batch_size = batch_size
        self.char_to_lbl_dict = char_to_lbl_dict
        self.img_h = img_h
        self.epoch_size = epoch_size
        self.validation = validation
        self.keras_augmentor = keras_augmentor

        # total number of unique characters
        self.num_chars = len(char_to_lbl_dict)

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch """
        return self.epoch_size


    def __getitem__(self, index):
        """Generate one batch of data"""

        # stores the length (number of characters) of each word in a batch
        label_lens = np.zeros((self.batch_size),dtype=np.float32)

        # generate content for the batch as a list of lists
        generated_content = list(list(tup) for tup in itertools.islice(self.base_generator,self.batch_size))

        # preprocess the batch content
        generated_content, img_w, max_word_len_batch = \
                self.preprocess_batch_imgs(generated_content)

        # allocate the vectors for batch labels (integers for each character in a word)
        # and the padded + preprocessed images
        batch_labels = np.zeros((self.batch_size, max_word_len_batch),dtype=np.float32)
        batch_imgs = np.zeros((self.batch_size, img_w, self.img_h, 3),dtype=np.float32)

        # the number of time distributed values, or another words the length of the time axis in the output,
        # or equivalently the width of the image after convolutions. Needed to input in the CTC loss
        # each maxpooling halves the width dimension so in our model scaling is 1/4 with 2 maxpoolings
        t_dist_dim = int(img_w / 4)

        # we need to give it for every entry
        input_length = np.full((self.batch_size),t_dist_dim,dtype=np.float32)

        # fill the batch
        for batch_ind in range(self.batch_size):

            # get a new image and a the content word for it
            img_arr, word = generated_content[batch_ind]
            batch_imgs[batch_ind,:,:] = img_arr

            # the labels for each word, even if the max number of characters is say for example 10
            # and the word is just 5 characters, the first 5 positions are filled by the character labels
            # and the rest are whatever (zeros in our implementation), however in the real loss theyre ignored
            # because of the label_length input
            labels_arr = np.array([self.char_to_lbl_dict[char] for char in word])
            batch_labels[batch_ind,0:len(labels_arr)] = labels_arr
            label_lens[batch_ind] = len(word)

            # now the hacky part
            # keras requires in the loss function to y_pred and y_true to be the same shape
            # but the ctc losses use y_pred of shape (batchsize, tdistdim, num_chars) from NN
            # and batch_labels, input_length, label_lens which are the "y_true" but these are
            # different dimension so pack them to (batchsize, tdistdim, num_chars) and later
            # unpack in the loss to stop the whining.
            y_true = np.zeros((self.batch_size, t_dist_dim, self.num_chars),dtype=np.float32)

            y_true[:, 0:max_word_len_batch, 0] = batch_labels
            y_true[:, 0, 1] = label_lens
            y_true[:, 0, 2] = input_length


        if self.validation:
            # for validation we return slightly different things so we can do fancy
            # stuff at callback

            return batch_imgs, batch_labels, input_length, label_lens

        else: #return x, y for the model
            return batch_imgs, y_true

    def preprocess_batch_imgs(self,generated_content):
        """Function to do augmentations, padd images, return longest word len etc"""

        # check the largest image width and word len in the batch
        pil_images = [img for img, word in generated_content]
        max_width = max([img.size[0] for img in pil_images])
        max_word_len_batch = max([len(word) for img, word in generated_content])


        # expand img with to mod 4_ds so that the maxpoolings wil result into
        # well defined integer length for the mapped tdist dimension ("new width")
        if max_width % 4 == 0:
            img_w = max_width
        else:
            img_w = max_width + 4 - (max_width % 4)

        #augment batch images
        for batch_ind in range(self.batch_size):

            # pad the image width with to the largest (fixed) image width
            pil_img = pil_images[batch_ind]
            width, height = pil_img.size

            new_img = Image.new(pil_img.mode, (img_w, self.img_h), (255,255,255))
            new_img.paste(pil_img, ((img_w - width) // 2, 0))

            # convert to numpy array
            img_arr = np.array(new_img)
            
            #some additional augmentation
            img_arr = self.keras_augmentor.random_transform(img_arr)

            # scale with 255 so that the values are between 0 and 1
            # and save to batch, also transpose because the "time axis" is width
            generated_content[batch_ind][0] = img_arr.transpose((1,0,2)) / 255

        return generated_content, img_w, max_word_len_batch