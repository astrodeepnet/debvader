import numpy as np
import tensorflow.keras


class BatchGenerator_vae(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the VAE.
    """

    def __init__(
        self,
        bands,
        list_of_samples,
        sample,
        total_sample_size,
        batch_size,
        list_of_weights_e=None,
    ):
        """
        Initialization function
        parameters:
            bands: list of bands. It must be a tuple.
            list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
            sample: choice between noisy or noiseless data, the two accepted input are 'noisy' or 'noiseless'. It must be a string.
            batch_size: size of the batches to feed the network
            total_sample_size: size of the whole training, validation, or test sample
            list_of_weights_e: list of weights to apply to the images (not required)
        """
        self.bands = bands
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.sample = sample
        self.epoch = 0

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode="c")
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        # self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        # print("Produced samples", self.produced_samples)
        self.produced_samples = 0

    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        sample_filename = np.random.choice(self.list_of_samples, p=self.p)

        sample = np.load(sample_filename, mmap_mode="c")
        sample_dc2 = np.load(
            sample_filename.replace("img_noiseless_sample", "img_cropped_sample"),
            mmap_mode="c",
        )

        if self.list_of_weights_e == None:
            indices = np.random.choice(
                new_data.index, size=self.batch_size, replace=False, p=None
            )
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(
                new_data.index,
                size=self.batch_size,
                replace=False,
                p=self.weights_e / np.sum(self.weights_e),
            )

        self.produced_samples += len(indices)

        x_1 = np.tanh(np.arcsinh(sample_dc2[indices][:, :, :, self.bands]))
        x_2 = np.tanh(np.arcsinh(sample[indices][:, :, :, self.bands]))

        # flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1:
            x_1 = np.flip(x_1, axis=2)
            x_2 = np.flip(x_2, axis=2)
        elif rand == 2:
            x_1 = np.swapaxes(x_1, 2, 1)
            x_2 = np.swapaxes(x_2, 2, 1)
        elif rand == 3:
            x_1 = np.swapaxes(np.flip(x_1, axis=2), 2, 1)
            x_2 = np.swapaxes(np.flip(x_2, axis=2), 2, 1)
        if len(self.bands) == 1:
            x_1 = np.expand_dims(x_1, axis=-1)
            x_2 = np.expand_dims(x_2, axis=-1)

        if self.sample == "noiseless":
            return x_2, x_2
        if self.sample == "noisy":
            return x_1, x_2
