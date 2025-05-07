import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mmlatch.attention import Attention
from mmlatch.util import pad_mask


class PadPackedSequence(nn.Module):
    """Some Information about PadPackedSequence"""
    """Converts a packed sequence back to a padded tensor"""
    def __init__(self, batch_first=True): #Initialization
        super(PadPackedSequence, self).__init__()
        self.batch_first = batch_first # determines if we have BxT or TxB (B:batch, T:Tensor)

    def forward(self, x, lengths):
        max_length = lengths.max().item()
        x, _ = pad_packed_sequence( # converts a batch of sequences x to a padded tensor of max_length
            x, batch_first=self.batch_first, total_length=max_length
        )
        return x


class PackSequence(nn.Module):
    """Packs a padded  sequence into a format suitable for RNNs"""
    def __init__(self, batch_first=True): #initialzation
        super(PackSequence, self).__init__()
        self.batch_first = batch_first # determines if we have BxT or TxB (B:batch, T:Tensor)

    def forward(self, x, lengths):
        x = pack_padded_sequence(#Packs a Tensor containing sequences of variable length for effecient RNN processing
            x, lengths, batch_first=self.batch_first, enforce_sorted=False #no sorting by lenght
        )
        lengths = lengths[x.sorted_indices.cpu()]#indices sort the sequence stored in cpu
        return x, lengths 

class RNN(nn.Module): 
    """Customizable RNN that supports LSTM and GRU"""
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        layers=1,
        bidirectional=False,
        merge_bi="cat", # how to merge bidirectional outputs cat is concatenation and sum is addition
        dropout=0,
        rnn_type="lstm", # type of RNN
        packed_sequence=True,
        device="cpu",
    ):

        super(RNN, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.merge_bi = merge_bi
        self.rnn_type = rnn_type.lower()

        self.out_size = hidden_size

        if bidirectional and merge_bi == "cat":
            self.out_size = 2 * hidden_size

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size,
            hidden_size,
            batch_first=batch_first,
            num_layers=layers,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)#
        self.packed_sequence = packed_sequence

        if packed_sequence:
            self.pack = PackSequence(batch_first=batch_first)
            self.unpack = PadPackedSequence(batch_first=batch_first)

    def _merge_bi(self, forward, backward): 
        """merges bidirectional outputs"""
        if self.merge_bi == "sum":
            return forward + backward

        return torch.cat((forward, backward), dim=-1)

    def _select_last_unpadded(self, out, lengths):
        """selects the last not padded token of each sequence"""
        gather_dim = 1 if self.batch_first else 0 # determine from which direction to gather based on batch_first
        gather_idx = (
            (lengths - 1)  # -1 to convert to indices (0-indexed)
            .unsqueeze(1)  # (B) -> (B, 1)
            .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
            # (B, 1, H) if batch_first else (1, B, H)
            .unsqueeze(gather_dim) # unsqueeze and expand to match the shape of out
        )
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)

        return last_out

    def _final_output(self, out, lengths):
        """computes the final output of the RNN"""
        # Collect last hidden state
        # Code adapted from https://stackoverflow.com/a/50950188

        #if not bidirectional, return the last hidden state
        if not self.bidirectional:
            return self._select_last_unpadded(out, lengths)
        
        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])
        # Last backward corresponds to first token
        last_backward_out = backward[:, 0, :] if self.batch_first else backward[0, ...]
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)

        return self._merge_bi(last_forward_out, last_backward_out)

    def merge_hidden_bi(self, out):
        """Merges hidden states of bidirectional RNN"""
        if not self.bidirectional:
            return out

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])

        return self._merge_bi(forward, backward)

    def forward(self, x, lengths, initial_hidden=None):
        """Forward pass of the RNN"""
        self.rnn.flatten_parameters()

        if self.packed_sequence: # if packed sequence, pack the input
            lengths = lengths.to("cpu")
            x, lengths = self.pack(x, lengths)
            lengths = lengths.to(self.device)

        if initial_hidden is not None: # if initial hidden state is provided
            out, hidden = self.rnn(x, initial_hidden)
        else:
            out, hidden = self.rnn(x)

        if self.packed_sequence: # if packed sequence, unpack the output
            out = self.unpack(out, lengths)

        out = self.drop(out) # apply dropout to the output
        last_timestep = self._final_output(out, lengths)
        out = self.merge_hidden_bi(out)

        return out, last_timestep, hidden
        #return the RNN output, the output of the last valud timestep and the final hidden state

class AttentiveRNN(nn.Module):
    """RNN with attention mechanism"""
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        layers=1,
        bidirectional=False,
        merge_bi="cat", # how to merge bidirectional outputs cat is concatenation and sum is addition
        dropout=0.1,
        rnn_type="lstm", # type of RNN
        packed_sequence=True,
        attention=False, # whether to use attention
        return_hidden=False,
        device="cpu",
    ):
        super(AttentiveRNN, self).__init__()
        self.device = device
        self.rnn = RNN(
            input_size,
            hidden_size,
            batch_first=batch_first,
            layers=layers,
            merge_bi=merge_bi,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=packed_sequence,
            device=device,
        )
        self.out_size = self.rnn.out_size
        self.attention = None
        self.return_hidden = return_hidden

        if attention:# Initialize attention mechanism with the output size of the RNN and  dropout
            self.attention = Attention(attention_size=self.out_size, dropout=dropout)

    def forward(self, x, lengths, initial_hidden=None):
        """Forward pass of the RNN"""
        #pass the input to the RNN
        out, last_hidden, _ = self.rnn(x, lengths, initial_hidden=initial_hidden)
  
        if self.attention is not None:
            out, _ = self.attention( #apply attention to the output of the RNN
                out, attention_mask=pad_mask(lengths, device=self.device)#mask to ignore padding
            )

            if not self.return_hidden:
                out = out.sum(1)
        else:
            out = last_hidden

        return out
