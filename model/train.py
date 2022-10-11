import argparse
import itertools
import pickle
import random
import torch

from rouge import Rouge
from typing import Callable, Dict, List, Generator, Iterable, Tuple

from data import read
from model import Encoder, Decoder, EncoderDecoderModel
from vocab import Vocabulary, SOS


def rouge_n(n: int) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """
    Finds the ROUGE-n score for a given pair of sequences.

    Arguments:
        n: The length of subsequences to use (i.e. the n in ROUGE-n).

    Returns:
        A function where:
            Arguments:
                sequence_1: The first sequence. This should be a one-dimensional sequence of
                            integers, e.g. torch.Tensor([1, 2, 3])
                sequence_2: The second sequence. This should also be a one-dimensional sequence of
                            integers.
            Returns:
                The ROUGE-n score, which is between 0 and 1.

    Raises:
        ValueError if n is not a positive integer.
    """

    if n <= 0:
        raise ValueError("n must be a positive integer.")

    def closure(sequence_1: torch.Tensor, sequence_2: torch.Tensor) -> float:
        sequence_1_as_string = ' '.join(str(u) for u in sequence_1)
        sequence_2_as_string = ' '.join(str(u) for u in sequence_2)

        rouge = Rouge(metrics=[f'rouge-{n}'])

        scores = rouge.get_scores(sequence_1_as_string, sequence_2_as_string)
        score = scores[0][f'rouge-{n}']['r']
        return score

    return closure


def evaluate(encoder: Encoder,
             decoder: Decoder,
             data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
             criterion: Callable[[torch.Tensor, torch.Tensor], float],
             num_sentences: int,
             max_length: int,
             device: torch.device,
    ) -> float:
    """
    Evaluate the machine translation model using a validation set.

    Args:
        encoder: The model encoder.
        decoder: The model decoder.
        data: The data stream. The sentence pairs can be from any language pair.
        criterion: A criterion which compares the generated sequence with the ground truth. Common
                   criteria are ROUGE and BLEU.
        num_sentences: How many sentences to evalaute on.
        max_length: The maximum length of a sentence; sentences are cropped to this.
        device: The device to run on.

    Raises:
        NotImplementedError: Until I implement this.

    Returns:
        The value returned by the criterion.
    """

    raise NotImplementedError


def train_many_to_many(model: EncoderDecoderModel,
                       data: Iterable[Tuple[
                           Tuple[torch.Tensor, torch.Tensor],
                           Tuple[int, int]]],
                       optimizer: torch.optim.Optimizer,
                       criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                       validation_split: float,
                       batch_size: int,
                       vocab: Vocabulary,
    ) -> Tuple[List[List[float]], List[float]]:
    """
    Trains the model for the first phase.
    In this phase, we train the model to translate from every language to every other language.

    Arguments:
        model: The model to train.
        data: A stream of input and target sequences and their lengths.
        optim: The optimizer to use.
        criterion: A loss function.
        learning_rate: The learning rate to use during optimization.
        validation_split: How much of the data to use for validation as a proportion from 0 to 1.
                          This cannot be equal to 1.
        batch_size: The batch size to use during training.
        num_batches: How many batches to go through before validation.
        vocab: The vocabulary to use during training.

    Returns:
        A tuple of two lists. The first list contains the training losses grouped by epoch and the
        second list contains the validation losses.

    Raises:
        ValueError: If validation_split is not in [0, 1).
    """

    if not (0 <= validation_split < 1):
        raise ValueError('The validation split must be between 0 (inclusive) and 1 (exclusive), ' +
                         f'but a validation split of {validation_split} was given.')

    num_epochs = 10
    num_batches_per_epoch = 3
    train_losseses: List[List[float]] = []
    validation_losses: List[float] = []
    for _ in range(num_epochs):
        train_losses, validation_loss = train_many_to_many_single_epoch(model,
                                                                        data,
                                                                        optimizer,
                                                                        criterion,
                                                                        validation_split,
                                                                        batch_size,
                                                                        num_batches_per_epoch,
                                                                        vocab)
        train_losseses.append(train_losses)
        validation_losses.append(validation_loss)

    return train_losseses, validation_losses


def train_many_to_many_single_epoch(model: EncoderDecoderModel,
                                    data: Iterable[Tuple[
                                        Tuple[torch.Tensor, torch.Tensor],
                                        Tuple[int, int]]],
                                    optimizer: torch.optim.Optimizer,
                                    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                    validation_split: float,
                                    batch_size: int,
                                    num_batches: int,
                                    vocab: Vocabulary,
    ) -> Tuple[List[float], float]:
    """
    Trains the model for a single epoch during the first phase.
    It trains on the data and then evaluates on a validation set.

    Args:
        model: The model to train.
        data: A stream of input and target sequences and their lengths.
        optim: The optimizer to use.
        criterion: A loss function.
        learning_rate: The learning rate to use during optimization.
        validation_split: How much of the data to use for validation as a proportion from 0 to 1.
                          This cannot be equal to 1.
        batch_size: The batch size to use during training.
        num_batches: How many batches to go through before validation.
        vocab: The vocabulary to use during training.

    Returns:
        A two-tuple where the first element is a list of the training losses and the second element
        is the validation loss.

    Raises:
        ValueError: If validation_split is not in [0, 1).
    """

    def model_inputs(data: Iterable[Tuple[ Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]],
                     batch_size: int = batch_size // 2) \
        -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        pairs_with_valid_lengths = list(itertools.islice(data, batch_size))
        pairs = [pair for pair, _ in pairs_with_valid_lengths]
        valid_lengths = [valid_lengths for _, valid_lengths in pairs_with_valid_lengths]

        batch   = torch.stack([pair[0] for pair in pairs] + [pair[1] for pair in pairs])
        targets = torch.stack([pair[1] for pair in pairs] + [pair[0] for pair in pairs])

        valid_lengths = [valid_length[0] for valid_length in valid_lengths] + \
                        [valid_length[1] for valid_length in valid_lengths]

        decoder_input_sequences = torch.zeros_like(targets)
        decoder_input_sequences[:, 0] = vocab.token_to_index[SOS]
        decoder_input_sequences_valid_lengths = torch.ones(batch_size)

        return (batch,
                valid_lengths,
                decoder_input_sequences,
                decoder_input_sequences_valid_lengths,
                targets)


    if not (0 <= validation_split < 1):
        raise ValueError('validation split must be between 0 (inclusive) and 1 (exclusive), ' + \
                         f'but {validation_split} was passed.')

    model.train()

    losses: List[float] = []

    for _ in range(num_batches):
        # we divide the batch size by two because we use the pairs in both directions in phase 1,
        # e.g. we use an (EN, FR) pair to train in the EN -> FR direction and the FR -> EN direction
        (batch,
         valid_lengths,
         decoder_input_sequences,
         decoder_input_sequences_valid_lengths,
         targets) = model_inputs(data)

        output = model(batch,
                       valid_lengths,
                       decoder_input_sequences,
                       decoder_input_sequences_valid_lengths)
        predictions = output.argmax(axis=2)
        predictions, targets = predictions.to(torch.float), targets.to(torch.float)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        losses.append(float(loss))

    validation_size = int(validation_split * (num_batches * batch_size))
    (batch,
     valid_lengths,
     decoder_input_sequences,
     decoder_input_sequences_valid_lengths,
     targets) = model_inputs(data, batch_size=validation_size)

    output = model(batch,
                   valid_lengths,
                   decoder_input_sequences,
                   decoder_input_sequences_valid_lengths)
    predictions = output.argmax(axis=2)
    predictions, targets = predictions.to(torch.float), targets.to(torch.float)
    validation_loss = float(criterion(predictions, targets))

    return losses, validation_loss


def train_many_to_one(encoder: Encoder,
                      decoder: Decoder,
                      data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                      optim: torch.optim.Optimizer,
                      criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                      validation_split: float,
                      batch_size: int,
                      max_length: int,
                      device: torch.device,
    ) -> Tuple[List[List[float]], List[float], List[float]]:
    """
    Trains the model for the second phase.
    In this phase, we fine-tune the general, many-to-many model to translate to a specific language.

    Arguments:
        encoder: The encoder.
        decoder: The decoder.
        data: The data stream. The sentence pairs can be from any language pair.
        optim: The optimizer.
        criterion: The loss function to use.
        validation_split: The proportion of the data to use for validation. It must be between zero
                          (inclusive) and one (exclusive).
        batch_size: The batch size to use.
        max_length: The maximum length a sequence can be.
        device: The device to train on. Note that training on multiple device is unsupported.

    Returns:
        A tuple of three lists. The first list contains the training losses grouped by epoch, the
        second list contains the validation losses, and the third list contains the values returned
        by the criterion.

    Raises:
        ValueError: If validation_split is not in [0, 1).
    """

    if not (0 <= validation_split < 1):
        raise ValueError('validation split must be between 0 (inclusive) and 1 (exclusive), ' + \
                         f'but {validation_split} was passed.')

    raise NotImplementedError


def train_many_to_one_single_epoch(encoder: Encoder,
                                   decoder: Decoder,
                                   data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                                   optim: torch.optim.Optimizer,
                                   criterion: Callable[[torch.Tensor, torch.Tensor], float],
                                   learning_rate: float,
                                   validation_split: float,
                                   batch_size: int,
                                   num_batches: int,
                                   max_length: int,
                                   device: torch.device,
    ) -> Tuple[List[float], float, float]:
    """
    Trains the model for a single epoch during the second phase.
    It trains on the data and then evaluates on a validation set.

    Args:
        encoder: The encoder to train.
        decoder: The decoder to train.
        data: A stream of input and target sequences. The input sentences can be in any language,
              but the target sentences must all be in the same language.
        optim: The optimizer to use.
        criterion: A function which finds how similar two sentences are.
        learning_rate: The learning rate to use during optimization.
        validation_split: How much of the data to use for validation as a proportion from 0 to 1.
                          This cannot be equal to 1.
        batch_size: The batch size to use during training.
        num_batches: How many batches to go through before validation.
        max_length: The maximum length a sequence can be.
        device: The device to use for training. Note that we can only use a single device for
                training.

    Returns:
        A three-tuple where the first element is a list of the training losses, the second element
        is the validation loss, and the third element is the value recieved by the criterion.

    Raises:
        ValueError: If validation_split is not in [0, 1).
    """

    if not (0 <= validation_split < 1):
        raise ValueError('validation split must be between 0 (inclusive) and 1 (exclusive), ' + \
                         f'but {validation_split} was passed.')

    raise NotImplementedError


def clone(encoder: Encoder, decoder: Decoder) -> Tuple[Encoder, Decoder]:
    """
    Clone a model.
    """

    raise NotImplementedError


def train(encoder: Encoder,
          decoder: Decoder,
          data: Dict[Tuple[str, str], Iterable[Tuple[torch.Tensor, torch.Tensor]]],
          first_phase_learning_rate: int,
          second_phase_learning_rate: int,
          validation_split: float,
          batch_size: int,
          max_length: int,
          device: torch.device,
    ) -> Dict[str, Tuple[Encoder, Decoder]]:
    """
    Trains the model over both phases.
    It returns the many-to-one machine translation models.

    :param encoder: The encoder.
    :param decoder: The decoder.
    :param data: A dictionary which maps language pairs (e.g. ('en', 'fr')) to an iterator of
                 sentence pairs tensors.
    :param first_phase_learning_rate: The learning rate for the first phase.
    :param second_phase_learning_rate: The learning rate for the second phase.
    :return: A dictionary mapping languages to their many-to-one machine translation models.
    """

    def randomly_sampled_sentence_pairs(
        data: Dict[Tuple[str, str], Iterable[Tuple[torch.Tensor, torch.Tensor]]],
        ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Use the existing iterables to make a new iterable which randomly samples
        with uniformity over the language pairs.
        """

        language_pairs_which_still_have_data = list(data.keys())
        while len(language_pairs_which_still_have_data) != 0:
            language_pair = random.choice(language_pairs_which_still_have_data)
            try:
                tensor_1, tensor_2 = next(data[language_pair])
                yield tensor_1, tensor_2
                yield tensor_2, tensor_1
            except StopIteration:
                language_pairs_which_still_have_data.remove(language_pair)

    def randomly_sampled_sentence_pairs_for_single_language_pair(
        data: Dict[Tuple[str, str], Iterable[Tuple[torch.Tensor, torch.Tensor]]],
        language: str,
        ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Tensors which translate from any language to the given language.
        This should be used in the second phase.
        """

        pairs_involving_language_which_still_have_data = [pair for pair in data if language in pair]
        while len(pairs_involving_language_which_still_have_data) != 0:
            language_pair = random.choice(pairs_involving_language_which_still_have_data)
            try:
                tensor_1, tensor_2 = next(data[language_pair])
                yield tensor_1, tensor_2 if language_pair[1] == language else tensor_2, tensor_1
            except StopIteration:
                pairs_involving_language_which_still_have_data.remove(language_pair)

    # phase 1
    train_many_to_many(encoder,
                       decoder,
                       randomly_sampled_sentence_pairs(data),
                       learning_rate=first_phase_learning_rate,
                       validation_split=validation_split)

    # phase 2
    languages = set()
    for pair in data.keys():
        languages.add(pair[0])
        languages.add(pair[1])

    many_to_one_models: Dict[str, Tuple[Encoder, Decoder]] = dict()
    base_encoder, base_decoder = encoder, decoder
    for language in languages:
        encoder, decoder = clone(base_encoder), clone(base_decoder)
        train_many_to_one(encoder,
                          decoder,
                          randomly_sampled_sentence_pairs_for_single_language_pair(data, language),
                          second_phase_learning_rate,
                          validation_split=validation_split)
        many_to_one_models[language] = (encoder, decoder)

    return many_to_one_models


def main() -> None:
    """
    Given a hyperparameter configuration, train the model using that configuration.
    The configuration is given by the command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--languages',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--first_phase_learning_rate',
        type=float,
        required=False,
        default=0.001,
    )
    parser.add_argument(
        '--second_phase_learning_rate',
        type=float,
        required=False,
        default=0.001,
    )
    parser.add_argument(
        '--validation_split',
        type=float,
        required=False,
        default=0.1,
    )

    encoder_args = ['encoder_num_hiddens',
                    'encoder_ffn_num_hiddens',
                    'encoder_num_heads',
                    'encoder_num_blocks',
                    'max_length']
    decoder_args = ['decoder_num_hiddens',
                    'decoder_ffn_num_hiddens',
                    'decoder_num_heads',
                    'decoder_num_blocks']
    for argname in encoder_args + decoder_args:
        parser.add_argument(
            f'--{argname}',
            type=int,
            required=True,
        )
    parser.add_argument(
        '--encoder_use_bias',
        type=bool,
        required=True
    )
    parser.add_argument(
        '--encoder_dropout',
        type=float,
        required=True
    )
    parser.add_argument(
        '--decoder_dropout',
        type=float,
        required=True
    )

    args = parser.parse_args()

    # we load in the dataset
    with open(f'data/vocabulary.pickle', 'rb') as file:
        vocab: Vocabulary = pickle.load(file)

    data = read(args.languages.split(' '), vocab)

    encoder = Encoder(len(vocab),
                      args.encoder_num_hiddens,
                      args.encoder_ffn_num_hiddens,
                      args.encoder_num_heads,
                      args.encoder_num_blocks,
                      args.encoder_dropout,
                      args.max_length,
                      args.encoder_use_bias)
    decoder = Decoder(len(vocab),
                      args.decoder_num_hiddens,
                      args.decoder_ffn_num_hiddens,
                      args.decoder_num_heads,
                      args.decoder_num_blocks,
                      args.decoder_dropout,
                      args.max_length)

    train(encoder,
          decoder,
          data,
          args.first_phase_learning_rate,
          args.second_phase_learning_rate,
          args.validation_split)


if __name__ == '__main__':
    main()
