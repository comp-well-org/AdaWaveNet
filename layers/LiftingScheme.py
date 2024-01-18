import torch
import torch.nn as nn
import torch.nn.functional as F

class Splitting(nn.Module):
    def __init__(self, channel_first):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        self.channel_first = channel_first
        if(channel_first):
            self.conv_even = lambda x: x[:, :, ::2]
            self.conv_odd = lambda x: x[:, :, 1::2]
        else:
            self.conv_even = lambda x: x[:, ::2, :]
            self.conv_odd = lambda x: x[:, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))

class LiftingScheme(nn.Module):
    def __init__(self, in_channels, input_size, modified=True, splitting=True, k_size=4, simple_lifting=True):
        super(LiftingScheme, self).__init__()
        self.modified = modified
        kernel_size = k_size
        pad = (k_size // 2, k_size - 1 - k_size // 2)

        self.splitting = splitting
        self.split = Splitting(channel_first=True)

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:            
            modules_P += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=1, groups=in_channels),
                nn.GELU(),
                nn.LayerNorm([in_channels, input_size // 2])
            ]
            modules_U += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=1, groups=in_channels),
                nn.GELU(),
                nn.LayerNorm([in_channels, input_size // 2])
            ]
        else:
            size_hidden = 2
            
            modules_P += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels*prev_size, in_channels*size_hidden, kernel_size=kernel_size, stride=1, groups=in_channels),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels*prev_size, in_channels*size_hidden, kernel_size=kernel_size, stride=1, groups=in_channels),
                nn.Tanh()
            ]
            prev_size = size_hidden

            # Final dense
            modules_P += [
                nn.Conv1d(in_channels*prev_size, in_channels, kernel_size=1, stride=1, groups=in_channels),
                nn.Tanh()
            ]
            modules_U += [
                nn.Conv1d(in_channels*prev_size, in_channels, kernel_size=1, stride=1, groups=in_channels),
                nn.Tanh()
            ]

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c)
            return (c, d)
        else:
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            return (c, d)
        
        
class InverseLiftingScheme(nn.Module):
    def __init__(self, in_channels, input_size, kernel_size=4, simple_lifting=False):
        super(InverseLiftingScheme, self).__init__()
        self.wavelet = LiftingScheme(in_channels, k_size=kernel_size, simple_lifting=simple_lifting, input_size=input_size * 2)

    def forward(self, c, d):
        if self.wavelet.modified:
            x_even = c - self.wavelet.U(d)
            x_odd = d + self.wavelet.P(x_even)
        else:
            x_even = c - self.wavelet.U(d)
            x_odd = d + self.wavelet.P(x_even)

        # Merge the even and odd components to reconstruct the original signal
        B, C, L = c.size()  # or c.shape
        x = torch.zeros((B, C, 2 * L), dtype=c.dtype, device=c.device)
        x[..., ::2] = x_even
        x[..., 1::2] = x_odd

        return x