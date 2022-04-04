import torch
import torch.nn as nn
from torch.autograd import Variable


class GRU(nn.Module):
    """
    We have defined our GRU because we have imlemented some funcionality which was not not available from nn.GRU
    at the time of writing, e.g., ability to handle variable length sequences, option to provide a label per frame or
    per sequence. It might worth chekcing in the future if this functionality is supported from the built-in version.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        every_frame: bool = True,
        device=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        print(f"GRU hidden size {hidden_size}")
        self.num_layers = num_layers
        self.every_frame = every_frame
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.device = device

    def forward(self, x, lengths):
        batch_size = x.size(0)
        lengths = torch.as_tensor(lengths, dtype=torch.int64)
        # Forward propagate RNN
        if batch_size > 1:
            # crop sequences from 0 to length
            x = [
                torch.index_select(
                    x[index],
                    0,
                    Variable(torch.LongTensor(range(i)).squeeze()).cuda(self.device),
                )
                for index, i in enumerate(lengths)
            ]
            # pad with 0s
            x = [
                torch.cat(
                    (
                        s,
                        Variable(torch.zeros(lengths[0] - s.size(0), s.size(1))).cuda(self.device),
                    ),
                    0,
                )
                if s.size(0) != lengths[0]
                else s
                for s in x
            ]
            x = torch.stack(x, 0)
            # from list of torch tensors to packed sequence object
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
            # GRU fwd pass
            x, _ = self.gru(x)
            x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x = x.contiguous()
        elif batch_size == 1:
            x, _ = self.gru(x)

        if self.every_frame:
            x = self.fc(x)  # predictions based on every time step
        else:
            x = self.fc(x[:, -1, :])  # predictions based on last time-step
        return x
