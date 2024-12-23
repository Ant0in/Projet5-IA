
import numpy as np
from src.AutoEncoder import AutoEncoder
from src.utils import *
import argparse



class MainExecutor:

    def __init__(self, logger: Logger) -> None:
        self.logger: Logger = logger  # Logger idk cool shit

    def readDataset(self, dataset_fp: str) -> np.ndarray:
        self.logger.info(msg=f'Reading dataset @ {dataset_fp}.')
        x_, _ = get_dataset(filepath=dataset_fp)
        self.logger.info(msg=f'Finished reading dataset.')
        return x_

    def initializeAutoEncoder(self, x_vector: np.ndarray, learning_rate: float, encoded_dim: int) -> AutoEncoder:
        # Generates an AutoEncoder based on the dataset written in x_vector and the parameters.
        self.logger.info(msg=f'Initializing AutoEncoder with parameters : (learning_rate={learning_rate}, encoded_dim={encoded_dim}).')
        ae: AutoEncoder = AutoEncoder(input_dim=x_vector.shape[1], encoded_dim=encoded_dim, learning_rate=learning_rate)
        self.logger.info(msg=f'AutoEncoder initialized.')
        return ae
    
    def train(self, encoder: AutoEncoder, x_vector: np.ndarray, batch_size: int, epochs: int) -> list[float]:
        # Trains 'encoder' AutoEncoder with 'x_vector', and returns the losses vector.
        self.logger.info(msg=f'Starting training for encoder with parameters : (epochs={epochs}, batch_size={batch_size})')
        losses: list[float] = encoder.train(x_train=x_vector, epochs=epochs, batch_size=batch_size)
        self.logger.info(msg=f'Training complete for encoder.')
        self.logger.info(msg=f'Losses over epochs ({epochs}) : {[float(round(l, 4)) for l in losses]}')
        return losses



class Argparser:

    @staticmethod
    def parseArgs() -> dict:
        parser = argparse.ArgumentParser(description="AutoEncoder Training Script")
        parser.add_argument("--dataset", type=str, required=True, help="Dataset path (ex: './dataset/mnist_train.csv').")
        parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for autoencoder (default : 0.01).")
        parser.add_argument("--encoded_dim", type=int, default=64, help="Encoded vector dimension (default : 64).")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default : 32).")
        parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training (default : 10).")
        parser.add_argument("--verbose", action="store_true", default=True, help="Set logger verbose flag (default : true)")
        parser.add_argument("--log_path", type=str, default=None, help="Log file path (ex: './logs/logs.log').")
        return vars(parser.parse_args())
    


def main() -> list[float]:
    parameters: dict = Argparser.parseArgs()
    executor: MainExecutor = MainExecutor(logger=Logger(filepath=parameters['log_path'], verbose=parameters['verbose']))
    x_train: np.ndarray = executor.readDataset(dataset_fp=parameters['dataset'])
    encoder: AutoEncoder = executor.initializeAutoEncoder(x_vector=x_train,
        learning_rate=parameters['learning_rate'], encoded_dim=parameters['encoded_dim'])
    losses: list[float] = executor.train(encoder=encoder, x_vector=x_train, batch_size=parameters['batch_size'], epochs=parameters['epochs'])
    return losses


if __name__ == "__main__":

    # run examples : 
    # >> python .\main.py --dataset .\dataset\mnist_train.csv --learning_rate 0.01 --batch_size 32 --epochs 10

    _ = main()

