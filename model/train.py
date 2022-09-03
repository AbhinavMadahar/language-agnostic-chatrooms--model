import argparse

from typing import List, Tuple


def train(learning_rate: float) -> Tuple[List[float], List[float]]:
    """
    Trains the model.

    :param learning_rate: The learning rate.
    :return: A list of the train loss values and a list of the validation loss values.
    """

    raise NotImplementedError


def main() -> None:
    """
    Given a hyperparameter configuration, train the model using that configuration.
    The configuration is given by the command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        required=False,
        default=0.001,
    )

    args = parser.parse_args()

    train(**args)


if __name__ == "__main__":
    main()