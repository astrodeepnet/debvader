from abc import ABC, abstractmethod

import numpy as np


class Normalization(ABC):
    @abstractmethod
    def forward(self, images):
        pass

    @abstractmethod
    def inverse(self, images):
        pass


class LinearNormCosmos(Normalization):
    """
    Performs linear normalization/denormalization on Cosmos data
    """

    def forward(self, images):
        """
        function to perform linear normalization of the data

        parameters:
            images: numpy array to be denormalzied.
        """
        return images / 80000

    def inverse(self, images):
        """
        function to perform linear denormalization of the data

        parameters:
            images: numpy array to be denormalzied.
        """
        return images * 80000


class NonLinearNormCosmos(Normalization):
    """
    Performs non-linear normalization/denormalization on Cosmos data
    """

    def forward(self, images):
        """
        non-linear normalization used for cosmos dataset

        parameters:
            images: numpy array to be normalzied.
        """
        return np.tanh(np.arcsinh(images))

    def inverse(self, images):
        """
        non-linear denormalization used for cosmos dataset

        parameters:
            images: numpy array to be denormalzied.
        """
        return np.sinh(np.arctanh(images))
