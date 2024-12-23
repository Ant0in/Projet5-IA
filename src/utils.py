
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from datetime import datetime



class Logger:

    def __init__(self, filepath: str | None = None, verbose: bool = False):
        
        self._verbose: bool = verbose
        self._filepath: str | None = filepath

    @property
    def verbose(self) -> bool:
        return self._verbose
    
    @property
    def filepath(self) -> str | None:
        return self._filepath
    
    def getTimestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def writeInFile(self, msg: str) -> None:  
        if not (self.filepath is None):
            # could keep the file descriptor somewhere but idk I'll pass
            with open(file=self.filepath, mode='a', encoding='utf-8') as f:
                f.write(f'{msg}\n')

    def printInSTDOUT(self, msg: str) -> None:
        if self.verbose: print(msg)

    def logMessage(self, msg: str) -> None:
        timestamped_msg: str = f"[{self.getTimestamp()}] {msg}"
        self.writeInFile(msg=timestamped_msg)
        self.printInSTDOUT(msg=timestamped_msg)

    def info(self, msg: str) -> None:
        self.logMessage(msg=f"[i] {msg}")

    def warning(self, msg: str) -> None:
        self.logMessage(msg=f"[w] {msg}")

    def error(self, msg: str) -> None:
        self.logMessage(msg=f"[e] {msg}")

    def critical(self, msg: str) -> None:
        self.logMessage(msg=f"[c] {msg}")



def plot_image(v: np.ndarray) -> None:
    
    """
    Displays a squared image, given a flat vector.
    """
    plt.imshow(np.reshape(v, (int(v.shape[0]**0.5), int(v.shape[0]**0.5))), cmap="gray")
    plt.show()

    
def get_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Reads and preprocess a training and testing dataset.
    Parameters:
        - filepath (str): the path to the csv file containing the dataset.
    Returns:
        - x_train, y_train, x_test, y_test (np.ndarray): the images and labels for training and testing sets.
    """

    d = np.loadtxt(filepath, delimiter=",", dtype=str)[1:].astype(np.int64)
    x, y = d[:, 1:], d[:, 0].T
    return x/255., y


def activation(x: int | float | np.ndarray) -> int | float | np.ndarray:
    
    """
    The sigmoid activation function.
    """

    return 1 / (1 + np.exp(-x))

    
def derivative(x: int | float | np.ndarray) -> int | float | np.ndarray:
    
    """
    The derivative of the sigmoid activation function.
    """

    return x * (1 - x)

