import torch
import torch.nn as nn
import math

from layers.LiftingScheme import LiftingScheme, InverseLiftingScheme

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # x - B, C, L
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        # print(front.shape, x.shape, end.shape)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x

import torch
import torch.nn as nn
import math

class moving_avg_imputation(nn.Module):
    """
    Moving average block modified to ignore zeros in the moving window.
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg_imputation, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Padding on the both ends of time series
        # x - B, C, L
        num_channels = x.shape[1]
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x_padded = torch.cat([front, x, end], dim=-1)

        # Create a mask for non-zero elements
        non_zero_mask = x_padded != 0

        # Calculate sum of non-zero elements in each window
        window_sum = torch.nn.functional.conv1d(x_padded, 
                                                weight=torch.ones((1, num_channels, self.kernel_size)).cuda(),
                                                stride=self.stride)

        # Count non-zero elements in each window
        window_count = torch.nn.functional.conv1d(non_zero_mask.float(), 
                                                  weight=torch.ones((1, num_channels, self.kernel_size)).cuda(),
                                                  stride=self.stride)

        # Avoid division by zero; set count to 1 where there are no non-zero elements
        window_count = torch.clamp(window_count, min=1)

        # Compute the moving average
        moving_avg = window_sum / window_count
        return moving_avg



class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=24, stride=1, imputation=False):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.moving_avg = moving_avg(kernel_size, stride=stride) if not imputation else moving_avg_imputation(self.kernel_size, self.stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x) 
        res = x - moving_mean
        return res, moving_mean
    
def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BottleneckBlock, self).__init__()
        # self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU()
        # This disable the conv if compression rate is equal to 1
        self.disable_conv = in_planes == out_planes
        if not self.disable_conv:
            self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        if self.disable_conv:
            return self.relu(x)
        else:
            return self.conv1(self.relu(x))

class AdpWaveletBlock(nn.Module):
    # def __init__(self, in_channels, kernel_size, share_weights, simple_lifting, regu_details, regu_approx):
    def __init__(self, configs, input_size):
        super(AdpWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)

    def forward(self, x):
        (c, d) = self.wavelet(x)
        x = c

        r = None
        if(self.regu_approx + self.regu_details != 0.0):
            if self.regu_details:
                rd = self.regu_details * d.abs().mean()
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(c.mean(), x.mean(), p=2)
            if self.regu_approx == 0.0:
                r = rd
            elif self.regu_details == 0.0:
                r = rc
            else:
                r = rd + rc

        x = self.norm_x(x)
        d = self.norm_d(d)
        
        return x, r, d
    
class InverseAdpWaveletBlock(nn.Module):
    # def __init__(self, in_channels, kernel_size, share_weights, simple_lifting):
    def __init__(self, configs, input_size):
        super(InverseAdpWaveletBlock, self).__init__()
        self.inverse_wavelet = InverseLiftingScheme(configs.enc_in, input_size=input_size, kernel_size=configs.lifting_kernel_size)

    def forward(self, c, d):
        reconstructed = self.inverse_wavelet(c, d)
        return reconstructed
    
class Model(nn.Module):
    # def __init__(self, input_size=1000, input_channels=1, seasonal_embed_channels=16,
    #              number_levels=4, kernel_size=4,
    #              share_weights=False, simple_lifting=False,
    #              regu_details=0.01, regu_approx=0.01, K=512):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.number_levels = configs.lifting_levels
        self.nb_channels_in = configs.enc_in

        self.series_decomp = series_decomp(imputation = self.task_name=='imputation')
        self.seasonal_linear = nn.Linear(self.seq_len, self.pred_len)
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
        
        self.seasonal_linear.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.trend_linear.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        
        # Construct the levels recursively (encoder)
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        in_planes = self.nb_channels_in
        input_size = configs.seq_len
        for i in range(configs.lifting_levels):
            self.encoder_levels.add_module(
                'encoder_level_'+str(i),
                AdpWaveletBlock(configs, input_size)
            )
            in_planes *= 1
            input_size = input_size // 2
            self.linear_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size),
                    # nn.GELU()
                )
            )
            self.coef_linear_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size),
                    # nn.GELU()
                )
            )
                    

        self.input_size = input_size
        
        # Construct the levels recursively (decoder)
        self.decoder_levels = nn.ModuleList()

        for i in range(configs.lifting_levels-1, -1, -1):
            self.decoder_levels.add_module(
                'decoder_level_'+str(i),
                InverseAdpWaveletBlock(configs, input_size=input_size)
            )
            in_planes //= 1
            input_size *= 2
        
    def forecast(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = x.permute(0,2,1)
        x, moving_mean = self.series_decomp(x)
        _, N, L = x.shape
        
        means = x.mean(2, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
                
        means_moving_avg = moving_mean.mean(2, keepdim=True).detach()
        moving_mean = moving_mean - means_moving_avg
        stdev_moving_avg = torch.sqrt(
            torch.var(moving_mean, dim=2, keepdim=True, unbiased=False) + 1e-5)
        moving_mean /= stdev_moving_avg
        
        encoded_coefficients = []
        rs = []
        # Encoding
        level = 0
        for l, l_linear in zip(self.encoder_levels, self.linear_levels):
            # print("Level", level, "x_shape:", x.shape)
            x, r, details = l(x)
            # print("The size of x is", x.size(), "and the size of details is", details.size(), l_linear)
            # x = l_linear(x)
            encoded_coefficients.append(details)
            rs += [r]
            level += 1
        
        # Decoding
        for dec, l_linear, c_linear in zip(self.decoder_levels, self.linear_levels[::-1], self.coef_linear_levels[::-1]):
            details = encoded_coefficients.pop()
            x = l_linear(x)
            details = c_linear(details)
            x = dec(x, details)
            
        moving_mean_out = self.trend_linear(moving_mean)
        x = self.seasonal_linear(x)
        x = x * (stdev[:, :, 0].unsqueeze(2).repeat(1, 1, L))
        x = x + (means[:, :, 0].unsqueeze(2).repeat(1, 1, L))
        moving_mean_out = moving_mean_out * (stdev_moving_avg[:, :, 0].unsqueeze(2).repeat(1, 1, L))
        moving_mean_out = moving_mean_out + (means_moving_avg[:, :, 0].unsqueeze(2).repeat(1, 1, L))
        
        x = x + moving_mean_out
        
        return x
    
    def imputation(self, x, x_mark_enc, x_dec, x_mark_dec, mask):
        x = x.permute(0,2,1)
        mask = mask.permute(0,2,1)
        x, moving_mean = self.series_decomp(x)
        # moving_mean = moving_avg_imputation(24, 1)(x)
        # x = x - moving_mean
        _, N, L = x.shape
        
        means = torch.sum(x, dim=2) / torch.sum(mask == 1, dim=2)
        means = means.unsqueeze(2).detach()
        x = x - means
        stdev = torch.sqrt(torch.sum(x * x, dim=2) /
                           torch.sum(mask == 1, dim=2) + 1e-5).unsqueeze(2).detach()
        x /= stdev
                
        means_moving_avg = moving_mean.mean(2, keepdim=True).detach()
        moving_mean = moving_mean - means_moving_avg
        stdev_moving_avg = torch.sqrt(
            torch.var(moving_mean, dim=2, keepdim=True, unbiased=False) + 1e-5)
        moving_mean /= stdev_moving_avg
        
        encoded_coefficients = []
        rs = []
        # Encoding
        level = 0
        for l, l_linear in zip(self.encoder_levels, self.linear_levels):
            # print("Level", level, "x_shape:", x.shape)
            x, r, details = l(x)
            # print("The size of x is", x.size(), "and the size of details is", details.size(), l_linear)
            # x = l_linear(x)
            encoded_coefficients.append(details)
            rs += [r]
            level += 1
        
        # Decoding
        for dec, l_linear, c_linear in zip(self.decoder_levels, self.linear_levels[::-1], self.coef_linear_levels[::-1]):
            details = encoded_coefficients.pop()
            x = l_linear(x)
            details = c_linear(details)
            x = dec(x, details)
            
        moving_mean_out = self.trend_linear(moving_mean)
        x = self.seasonal_linear(x)
        x = x * (stdev[:, :, 0].unsqueeze(2).repeat(1, 1, L))
        x = x + (means[:, :, 0].unsqueeze(2).repeat(1, 1, L))
        moving_mean_out = moving_mean_out * (stdev_moving_avg[:, :, 0].unsqueeze(2).repeat(1, 1, L))
        moving_mean_out = moving_mean_out + (means_moving_avg[:, :, 0].unsqueeze(2).repeat(1, 1, L))
        
        x = x + moving_mean_out
        return x.permute(0, 2, 1)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out.permute(0, 2, 1)  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None