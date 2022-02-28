from random import choice

import numpy as np
from tensorflow.keras.utils import Sequence

from debvader.normalize import Normalizer


class COSMOSsequence(Sequence):
    def __init__(
        self,
        list_of_samples,
        x_col_name,
        y_col_name,
        batch_size,
        num_iterations_per_epoch,
        normalizer=None,
        channel_last=False,
    ):
        """
        initializes the Data generator

        parameters:
        list_of_samples: list of paths to the datafiles.
        x_col_name: column name of data to be fed as input to the network
        y_col_name: column name of data to be fed as target to the network
        batch_size: sample sixe for each batch
        num_iterations_per_epoch: number of samples (of size = batch_size) to be drawn from the sample
        normalizer: object of Debvader.normalize.Normalize, used to perform norm and denorm operations (default is None).
        channel_last: boolean to indicate if the the clast channel corresponds to differnet bands of the input data.
        """
        self.list_of_samples = list_of_samples
        self.x_col_name = x_col_name
        self.y_col_name = y_col_name
        self.batch_size = batch_size
        self.num_iterations_per_epoch = num_iterations_per_epoch
        if (normalizer is not None) and (not isinstance(normalizer, Normalizer)):
            raise ValueError(
                "The parameter `normalizer` should be an instance of debvader.normalize.Normalizer"
            )

        self.normalizer = normalizer
        self.channel_last = channel_last

    def __len__(self):
        return self.num_iterations_per_epoch

    def __getitem__(self, idx):

        current_loop_file_name = choice(self.list_of_samples)
        current_sample = np.load(current_loop_file_name, allow_pickle=True)

        batch = np.random.choice(current_sample, size=self.batch_size, replace=False)
        x = batch[self.x_col_name]
        y = batch[self.y_col_name]

        x = np.array(x.tolist())
        y = np.array(y.tolist())

        if self.normalizer is not None:
            x = self.normalizer.forward(x)
            y = self.normalizer.forward(y)

        #  flip : flipping the image array
        # if not self.channel_last:
        #    rand = np.random.randint(4)
        #    if rand == 1:
        #        x = np.flip(x, axis=-1)
        #        y = np.flip(y, axis=-1)
        #    elif rand == 2 :
        #        x = np.swapaxes(x, -1, -2)
        #        y = np.swapaxes(y, -1, -2)
        #    elif rand == 3:
        #        x = np.swapaxes(np.flip(x, axis=-1), -1, -2)
        #        y = np.swapaxes(np.flip(y, axis=-1), -1, -2)

        # Change the shape of inputs and targets to feed the network
        x = np.transpose(x, axes=(0, 2, 3, 1))
        y = np.transpose(y, axes=(0, 2, 3, 1))

        return x, y
