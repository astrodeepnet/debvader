from abc import ABC, abstractmethod

import numpy as np


class Normalizer(ABC):
    @abstractmethod
    def forward(self, images):
        """
        function to perform normalization of the data
        """
        pass

    @abstractmethod
    def backward(self, images):
        """
        function to perform linear denormalization of the data
        """
        pass


class IdentityNorm(Normalizer):
    """
    Performs identity normalization/denormalization
    """

    def forward(self, images):
        """
        makes no change to the data during normalization

        parameters:
            images: numpy array to be denormalzied.
        """
        return images

    def backward(self, images):
        """
        makes no change to the data during denormalization

        parameters:
            images: numpy array to be denormalzied.
        """
        return images


class LinearNormCosmos(Normalizer):
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

    def backward(self, images):
        """
        function to perform linear denormalization of the data

        parameters:
            images: numpy array to be denormalzied.
        """
        return images * 80000


class NonLinearNormCosmos(Normalizer):
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

    def backward(self, images):
        """
        non-linear denormalization used for cosmos dataset

        parameters:
            images: numpy array to be denormalzied.
        """
        return np.sinh(np.arctanh(images))
