import torch
from torch import nn
from torch.nn import functional as F
import math

class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        # set them as True during inference
        self.bias_initialized = False 
        self.scale_initialized = False
        
        # automaticlly registered as parameters, reference: https://github.com/pytorch/pytorch/blob/d1a44676828ef65067414c938b15412f85d1a39e/torch/nn/modules/module.py#L225-L283
        self.bias = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        # self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.logscale = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        
        
    def initialize_bias(self, x):
        """
        initialize bias
        :param: x: input, type: torch.Tensor, shape: (N,C,H,W)
        """

        with torch.no_grad():
            # comppute initial value, x_mean shape: (1,C,1,1)
            x_mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
             # Copy to parameters
            self.bias.data.copy_(-x_mean.data)
            self.bias_initialized = True

    def initialize_scale(self, x):
        """
        Initialize scale
        :param x: input, type: torch.Tensor, shape (N,C,H,W)
        """
        with torch.no_grad():
            # comppute initial value, x_std shape: (1,C,1,1)
            x_std = torch.std(x, dim=[0, 2, 3], keepdim=True)
            scale = 1/(x_std + 1e-9)
            # self.scale.data.copy_(scale.data)
            # self.logscale.data.copy_(torch.log(scale).data)

            logscale_tmp = torch.log(scale)/3.
            self.logscale.data.copy_(logscale_tmp.data)

            # Copy to parameters
            self.scale_initialized = True

    def actnorm_center(self, x, reverse=False):
        """
        center operation of activation normalization
        :param: x: input; type : torch.Tensor, shape (N,C,H,W)
        :param: reverse: whether to reverse bias; type reverse: bool
        :return: centered input; rtype: torch.Tensor, shape (N,C,H,W)
        """

        if reverse:
            return x - self.bias 
        else:
            return x + self.bias 

    def actnorm_scale(self, x, reverse=False):
        """
        scale operation of activation normalization
        :param x: input, type : torch.Tensor, shape (N,C,H,W)
        :param reverse: whether to reverse bias, type: bool
        :return: centered input and logdet, rtype: tuple(torch.Tensor, torch.Tensor)
        """
        logscale_tmp = self.logscale * 3  # wzh
        if reverse:
            # x /= self.scale
            # x /= torch.exp(self.logscale)
            # x /= (torch.exp(self.logscale)  + 1e-9 )
            x /= (torch.exp(logscale_tmp)  + 1e-9 )

        else:
            # x *= self.scale
            # x *= (torch.exp(self.logscale)  + 1e-9 )
            x *= (torch.exp(logscale_tmp)  + 1e-9 )

            
        _,_,h,w = x.shape

        logdet_factor = h * w  
        # dlogdet = torch.sum(torch.log(torch.abs(self.scale))) * logdet_factor
        dlogdet = torch.sum(logscale_tmp)* logdet_factor

        if reverse:
            dlogdet *= -1
        logdet = dlogdet

        return x, logdet
         
    def forward(self, x, logdet=None):
        """
        forward activation normalization layer
        :param x: input, type: torch.Tensor, shape: (N,C,H,W)
        :return: normalized input and logdet, rtype: tuple(torch.Tensor, torch.Tensor)
        """
        if not self.bias_initialized:
            self.initialize_bias(x)
        if not self.scale_initialized:
            self.initialize_scale(x)
        # center and scale
        x = self.actnorm_center(x, reverse=False)
        x, dlogdet = self.actnorm_scale(x, reverse=False)

        if logdet is not None:
            logdet = logdet + dlogdet
        
        return x, logdet
    
    def reverse(self, x):
        """
        reverse activation normalization layer
        :param x: input, type: torch.Tensor, shape: (N,C,H,W)
        :return: normalized input and logdet, rtype: tuple(torch.Tensor, torch.Tensor)
        """
        # scale and center
        x, logdet = self.actnorm_scale(x, reverse=True)
        x = self.actnorm_center(x, reverse=True)
        
        return x, logdet

class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        pass
    def forward(self, input):
        pass
        return y, logdet

class InvConv2dLU(nn.Module):
    """
    invertible 1x1 convolution layer by utilizing LU decomposition
    p is fixed while LUS is optimized
    """
    def __init__(self, in_channel, weight=None):
        # in_channel indicates the channel size of the input
        super().__init__()
        self.in_channel = in_channel

        # set an random orthogonal matrix as the initial weight
        if weight is None:
            weight = torch.randn(in_channel, in_channel)
        Weight,_ = torch.qr(weight)

        # PLU decomposition
        self.weight = Weight # only for testing

        Weight_LU, pivots = torch.lu(Weight)
        w_p, w_l, w_u = torch.lu_unpack(Weight_LU, pivots)
        w_s = torch.diag(w_u)
        w_logs = torch.log(torch.abs(w_s))
        s_sign = torch.sign(w_s)

        w_u = torch.triu(w_u, 1)

        u_mask = torch.triu(torch.ones_like(w_u), 1)
        l_mask = u_mask.T

        l_eye = torch.eye(l_mask.shape[0])


        # fix P
        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", s_sign)
        self.register_buffer("l_eye", l_eye)

        self.w_l = torch.nn.Parameter(w_l)
        self.w_u = torch.nn.Parameter(w_u)
        # self.w_s = torch.nn.Parameter(w_s)
        self.w_logs = torch.nn.Parameter(w_logs)

    def forward(self, x, logdet=None):
        """
        forward invertible 1x1 convolution layer
        :param x: input, type: torch.Tensor, shape: (N,C,H,W)
        :return: convolutional result (N,C,H,W) and accumulated logdet, rtype: tuple(torch.Tensor, torch.Tensor)
        """
        N, c, h, w = x.shape
        # convolutional kernel size (in_channel,in_channel,1,1)
        W = self.calc_W(inverse=False)
        out = nn.functional.conv2d(x, W)

        # dlogdet = h * w * torch.sum(torch.log(torch.abs(self.w_s)))
        dlogdet = h * w * torch.sum(self.w_logs)
        if logdet is None:
            logdet = dlogdet
        else:
            logdet += dlogdet

        return out, logdet

    def reverse(self, x):
        """
        reverse invertible 1x1 convolution layer
        :param x: input, type: torch.Tensor, shape: (N,C,H,W)
        :return: reverse convolutional result (N,C,H,W), rtype: torch.Tensor
        """
        W = self.calc_W(inverse=True)
        x = nn.functional.conv2d(x, W)
        return x

    # def calc_W_pre(self, inverse=False):
    #     if inverse is False:
    #         Wtmp = self.w_p @ self.w_l @ (self.w_u + torch.diag(self.s_sign * torch.exp(self.w_logs)))
    #         W = Wtmp.unsqueeze(2).unsqueeze(3)
    #     else:
    #         Wtmp = self.w_p @ self.w_l @ (self.w_u + torch.diag(self.s_sign * torch.exp(self.w_logs)))
    #         W = Wtmp.inverse().unsqueeze(2).unsqueeze(3)
    #     return W

    def calc_W(self, inverse=False):
        # Wtmp = self.w_p @ (self.w_l * self.l_mask + self.l_eye) @ \
        #        ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        Wtmp = self.w_p @ (self.w_l * self.l_mask + self.l_eye) @ \
               ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_logs)))
        if inverse is False:
            W = Wtmp.unsqueeze(2).unsqueeze(3)
        else:
            W = Wtmp.inverse().unsqueeze(2).unsqueeze(3)
        return W

class Permutation2d(nn.Module):
    def __init__(self, in_channel, weight=None):
        # in_channel indicates the channel size of the input
        super().__init__()
        raise("fix permutation is not implemented.")

class Split2d(nn.Module):
    pass

class Squeeze2d(nn.Module):
    pass

class ZeroConv2d(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.zeroConv2d = nn.Conv2d(input_size, output_size, kernel_size=3, padding=0)
        self.zeroConv2d.weight.data.zero_()
        self.zeroConv2d.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, output_size, 1, 1))

    def forward(self, x):
        out = F.pad(x, [1, 1, 1, 1], 'constant', value=1)
        out = self.zeroConv2d(out)
        out *= torch.exp(3 * self.scale)
        
        return out
    
class NN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size // 2, hidden_size, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.zeroConv2d = ZeroConv2d(hidden_size, input_size)
        # self.zeroConv2d = ZeroConv2d(hidden_size, 1)

        # self.zeroConv2d = nn.Conv2d(hidden_size, input_size, kernel_size=3, padding=0)
        # self.zeroConv2d.weight.data.zero_()
        # self.zeroConv2d.bias.data.zero_()
        # self.scale = nn.Parameter(torch.zeros(1, input_size, 1, 1))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.zeroConv2d(x)
        return x
      
class AffineCoupling(nn.Module):
    """
        flow coupling layer
        """
    def __init__(self, in_channel, hidden_size=512):
        super().__init__()
        self.net = NN(in_channel, hidden_size)

    def forward(self, x, logdet):
        num_channel = x.shape[1]
        xa, xb = x.chunk(2, 1)
        # logs, t = self.net(xb).chunk(2, 1)
        xm = self.net(xb)
        t = xm[:, 0::2, ...]
        s = xm[:, 1::2, ...]

        s = torch.sigmoid(s + 2.)  + 1e-9
        ya = (xa + t)*s
        yb = xb
        y = torch.cat([ya, yb], 1)
        logdet = torch.sum(torch.log(s).view(x.shape[0], -1), 1) + logdet
        # logdet += torch.sum(torch.log(s).view(x.shape[0], -1), 1)

        return y, logdet

    def reverse(self, y):
        ya, yb = y.chunk(2, 1)
        # logs, t = self.net(yb).chunk(2, 1)

        # logs, t = self.net(yb).chunk(2, 1)
        ym = self.net(yb)
        t = ym[:, 0::2, ...]
        s = ym[:, 1::2, ...]


        # s = torch.sigmoid(logs + 2)
        s = torch.sigmoid(s + 2.) + 1e-9
        xa = ya / s - t
        xb = yb
        x = torch.cat([xa, xb], 1)
        return x


class Flow(nn.Module):
    """
    One step flow
    """
    def __init__(self, in_channel):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2dLU(in_channel)
        self.affine_coupling = AffineCoupling(in_channel)
        pass
    
    def forward(self, x, logdet=None):
        y, logdet = self.actnorm(x,logdet)
        y, logdet = self.invconv(y, logdet)
        y, logdet = self.affine_coupling(y, logdet)
        return y, logdet

    def reverse(self, x):
        y = self.affine_coupling.reverse(x)
        y = self.invconv.reverse(y)
        y, _ = self.actnorm.reverse(y)
        return y 
    
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channel, n_flows, split=True):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(Flow(in_channel*4))
        if split:
            self.prior = ZeroConv2d(in_channel*2, in_channel*4)
        else:
            self.prior = ZeroConv2d(in_channel*4, in_channel*8)
        self.split = split
        
    @staticmethod
    def gaussian_log_p(x, mean, log_sd):
        return -0.5 * torch.log(2 * torch.Tensor([math.pi])).cuda() - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)
    
    @staticmethod
    def gaussian_sample(eps, mean, log_sd):
        return mean + torch.exp(log_sd) * eps
    
    def forward(self, input, logdet):
        N, C, H, W = input.shape
        squeezed = input.view(N, C, H // 2, 2, W // 2, 2) # (N,C,H//2,2,W//2,2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4) # (N,C,2,2,H//2,W//2)
        out = squeezed.contiguous().view(N, C * 4, H // 2, W // 2) #(N,c*4,H//2,W//2)
        
        # logdet = 0
        # logdet = None
        for flow in self.flows:
            # out, logdet_ = flow(out)
            # logdet = logdet + logdet_
            out, logdet = flow(out, logdet)

            
        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = self.gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(N, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = self.gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(N, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new
      
    def reverse(self, input, eps=None):
        if self.split:
            input = torch.cat([input, eps], 1)
        else:
            input = eps
        for flow in self.flows[::-1]:
            input = flow.reverse(input)
    
        N, C, H, W = input.shape
        unsqueezed = input.view(N, C // 4, 2, 2, H, W)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            N, C // 4, H * 2, W * 2
        )

        return unsqueezed


if __name__ == '__main__':
    # AC = AffineCoupling(12)
    x = torch.randn(8,3,16,16) # img rgb; size 16x16; batch size 8
    
    # y = AC(x, torch.randn(8))
    # y = AC.reverse(x)
    flow = Flow(3)
    y = flow(x)
    z = flow.reverse(y[0])
    MSB1 = MultiScaleBlock(3,4)
    MSB2 = MultiScaleBlock(6,4)
    out1, logdet, log_p, z_new1 = MSB1.forward(x)
    out2, logdet, log_p, z_new2 = MSB2.forward(out1)
    unsqueezed = MSB2.reverse(out2,z_new2)
    print("pause")