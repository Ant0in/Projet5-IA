
import numpy as np
import matplotlib.pyplot as plt

from src.AutoEncoder import AutoEncoder
from src.utils import *



class MainExecutor:

    @staticmethod
    def main() -> None:

        x_train, _ = get_dataset("./dataset/mnist_train.csv")
        x_test, _ = get_dataset("./dataset/mnist_test.csv")

        # Déclaration de l'encodeur ;
        LR: float = 0.01
        encoder: AutoEncoder = AutoEncoder(input_dim=x_train.shape[1], encoded_dim=64, learning_rate=LR)

        # Paramètres d'entraînement ;
        epochs: int = 10
        batch_size: int = 32
        losses: list[float] = encoder.train(x_train=x_train, epochs=epochs, batch_size=batch_size)

        displayImages: int = 5

        plt.figure(figsize=(10, displayImages))
        for i in range(displayImages):
            v = x_test[i]
            ax = plt.subplot(2, displayImages, i + 1)
            plt.imshow(np.reshape(v, (28, 28)), cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            decoded_v = encoder.forward(v)
            ax = plt.subplot(2, displayImages, i + 1 + displayImages)
            plt.imshow(np.reshape(decoded_v, (28, 28)), cmap="gray")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()


if __name__ == "__main__":

    MainExecutor.main()

