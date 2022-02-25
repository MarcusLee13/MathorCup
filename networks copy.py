import functools
import torch
from torch import nn


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='batch'):
    """Return a normalization layer.

    Parameters:
        norm_type (str) -- the name of the normalization layer: [batch | instance | none]

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    Usually, `batch` is much better than `instance` and `none` but keeps much memory.
    """
    # functools.partial uses to fix some attrs
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class UnetGenerator(nn.Module):
    """Create a Unet-based hiding network."""
    def __init__(self, input_nc, output_nc, num_downs, nhf=64, norm_type='none', use_dropout=True, output_function='sigmoid'):
        """Construct a Unet generator.

        Parameters:
            input_nc (int)        -- the number of channels in input images
            output_nc (int)       -- the number of channels in output images
            num_downs (int)       -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                     image of size 128x128 will become of size 1x1 # at the bottleneck
            nhf (int)             -- the number of filters in the last conv layer of hiding network
            norm_type (str)       -- normalization layer type
            use_dropout (bool)    -- if use dropout layers
            output_function (str) -- activation function for the outmost layer [sigmoid | tanh]

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        
        # construct unet structure (from inner to outer)
        unet_block = UnetSkipConnectionBlock(nhf*8, nhf*8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            # considering dropout
            unet_block = UnetSkipConnectionBlock(nhf*8, nhf*8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from nhf*8 to nhf
        unet_block = UnetSkipConnectionBlock(nhf*4, nhf*8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nhf*2, nhf*4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nhf, nhf*2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, nhf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_function=output_function)

        if output_function == 'tanh':
            self.factor = 1.0   # by referencing the engineering choice in universal adversarial perturbations
            # self.factor = 10 / 255  # by referencing the engineering choice in universal adversarial perturbations
        elif output_function == 'sigmoid':
            self.factor = 1.0
        else:
            raise NotImplementedError('activation funciton [%s] is not found' % output_function)

    def forward(self, X):
        """standard forward."""
        return self.factor * self.model(X)


class UnetSkipConnectionBlock(nn.Module):
    """Define the Unet submodule with skip connection."""
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=True, output_function='sigmoid'):
        """Construct a Unet submodule with skip connection.
        
        Parameters:
            outer_nc (int)                      -- the number of filters in the outer conv layer
            inner_nc (int)                      -- the number of filters in the inner conv layer
            input_nc (int)                      -- the number of channels in the input images / features
            submodule (UnetSkipConnectionBlack) -- previous defined submodules
            outermost (bool)                    -- if this module is the outermost module
            innermost (bool)                    -- if this module is the innermost module
            norm_layer                          -- normalization layer
            use_dropout (bool)                  -- if use dropout layers
            output_function (str)               -- activation function for the outmost layer [sigmoid | tanh]
        """
        super(UnetSkipConnectionBlock, self).__init__()
        # `submodulde` is None if and only if this block is an innermost block"
        # `input_nc` is None if and only if this block is not an outermost block"

        self.outmost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)  # after Conv2d 
        downnorm = norm_layer(inner_nc)

        uprelu = nn.ReLU(True)  # after ConvTranspose2d
        upnorm = norm_layer(outer_nc)

        if outermost:
            # no dropout
            # no relu in down
            # no normalization in down and up
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            if output_function == 'tanh':
                up = [uprelu, upconv, nn.Tanh()]
            elif output_function == 'sigmoid':
                up = [uprelu, upconv, nn.Sigmoid()]
            else:
                raise NotImplementedError('activation funciton [%s] is not found' % output_function)
            # outer=torch.argmax(up,dim=3).repeat(1,3,1,1)
            model = down + [submodule] + up
        elif innermost:
            # no dropout
            # no normalization in down
            upconv = nn.ConvTranspose2d(inner_nc,outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outmost:
            
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)  # cat by channel


