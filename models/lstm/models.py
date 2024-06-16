import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lstm.layers import ConvLSTMlayer


'''
Implemented from: https://www.nature.com/articles/s41598-021-96674-0

Notes:
- They are using Keras, we are using Pytorch. There's a little friction.
- They used "hard sigmoid" instead of "tanh" for the recurrent activation function; torch's LSTM
    does not give you this option. We could have reimplemented an LSTMCell that had it, but it 
    would have run slower. We'd rather just train for longer.
- They specifically call it a "stateless" LSTM. In Keras, the "stateful" is generally used when
    models are trained auto-regressively. Also, their models output a different variable. 
    Thus, I assume they train the LSTM in the simplest way:
    always feed it a sequence of real data at training time, and make a single future prediction.
- I used an existing ConvLSTM implementation from another paper. A brief look through the internals
    and it looks like it's doing the same thing.
- They instantiate a separate ConvLSTM layer for each input. 
    This effectively groups the output feature maps by the input feature.
- I worked backwards to figure out their layer parameters for ConvLSTM. 
    - It's identical to their MaxPool2D from their Conv model
    - The total size is 480/5 = 96 features per input. 
    - 24*2*2 == 96 => Their maxpool reduces down to a [1(Time) x] 2(height) x 2(width) image
    - Assuming keras defaults, the maxpool input / conv output need to be an image sized 4 or 5.
    - I will assume they used padding in conv so the 5x5 input stays a 5x5 output
'''

class CommonHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 1)
        )
    def forward(self, x):
        x1 = self.seq(x)
        # Output a "time sequence" with just one time step
        return x1[:, None]


class LSTM(nn.Module):
    def __init__(self, in_channels, input_size=(5,5)):
        super().__init__()
        
        # To explain the "kernel" and "r.kernel" from the keras diagram:
        # As an optimisation, all of the weights of the gates are stored in just two Variables. 
        # The equations for LSTMs have 8 matrix multiplications with weights. 
        # Four for the input, and four for the hidden state.
        # The kernel, then, shows all the weights which multiply the input.
        #  In the diagram (125x196), which is just (125x[4x48]) concatenated variables.
        # And the "r.kernel" is the recurrent kernel, which multiply the hidden state.
        #  In the diagram (48x196), which is just (48x[4x48]) concatenated variables.

        # tl;dr - They didn't do anything weird, it's just that LSTMs weight sizes are confusing.
        h, w = input_size
        in_dim = in_channels * h * w
        self.lstm = nn.LSTM(in_dim, 48, batch_first=True)

        self.head = CommonHead(48)

    def forward(self, x, batch_positions=None, lead=None):
        assert lead is None, 'LSTM does not support variable lead time'
        # Flatten spatial dimensions
        x_flat = einops.rearrange(x, 'b t c h w -> b t (c h w)')
        lstm_out, lstm_hidden = self.lstm(x_flat)
        # Use the output at the last timestep
        # "output" is just the hidden state of the last layer, and is shaped:
        #      [Batch, Seq Len, Hidden size (projected)])
        out = lstm_out[:, -1]

        return self.head(out)


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels=24, input_size=(5, 5)):
        super().__init__()
        k_s = 3
        self.convlstms = nn.ModuleList([
            ConvLSTMlayer(input_size, 1, hidden_channels, kernel_size=(k_s, k_s))
            for i in range(in_channels)
        ])
        # For each input channel, an "image" shaped [2 2 hidden_channels] is created
        # after maxpooling. This image is then flattened and concatenated before the head.
        img_size = (input_size[0]//2) * (input_size[1]//2) * hidden_channels
        self.head = CommonHead(in_channels * img_size)

    def forward(self, x, batch_positions=None, lead=None):
        assert lead is None, 'LSTM does not support variable lead time'
        # Separate the inputs and push them through the convLSTMs independently
        convlstm_out = []
        channel_dim = 2
        for c_idx in range(x.shape[channel_dim]):
            # ConvLSTM
            one_channel = x.select(channel_dim, c_idx).unsqueeze(channel_dim)
            lstm_out, lstm_hidden = self.convlstms[c_idx](one_channel)
            # (lstm_out[-1][:, -1] != lstm_hidden[-1][0]).sum() # gives 0
            out = lstm_out[-1][:, -1]

            # Pool/flatten from [B 24 5 5] to [B 96]
            # (Note time dimension is removed, so we just use a max_pool2d; it's the same)
            pooled = F.max_pool2d(out, 2, padding=0)
            flat   = einops.rearrange(pooled, 'b c h w -> b (c h w)')
            convlstm_out.append(flat)
        
        convlstm_out = torch.concat(convlstm_out, dim=1)

        # The rest of the network is a common architecture.
        return self.head(convlstm_out)
