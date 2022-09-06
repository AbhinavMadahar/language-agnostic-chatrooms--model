import torch


class EncoderBlock:
    """
    A encoder block consisting of a multi-head attention and a positionwise feedforward layer, both wrapped with residual connections and layer norm.
    """

    def __init__(self) -> None:
        raise NotImplementedError
    
    def init_weights(self) -> None:
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Encoder:
    """
    Transformer-based encoder for the machine translation model.
    """

    def __init__(self):
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EncoderBlock:
    """
    A decoder block consisting of a masked multi-head attention, a multi-body attention using the encoder outputs, and a positionwise feedforward layer,
    all wrapped with residual connections and layer norm.
    """

    def __init__(self) -> None:
        raise NotImplementedError
    
    def init_weights(self) -> None:
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Decoder:
    """
    Transformer-based decoder for the machine translation model.
    """

    def __init__(self):
        raise NotImplementedError

    def init_state(self) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError