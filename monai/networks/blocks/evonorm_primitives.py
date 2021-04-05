import torch
import torch.nn as nn

class PrimitiveAdd(nn.Module):
    """ Primitive Add Operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 2

    def forward(self, x, y):
        return torch.add(x, y)

class PrimitiveMul(nn.Module):
    """Primitive Multiply Operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 2

    def forward(self, x, y):
        return torch.mul(x, y)

class PrimitiveDiv(nn.Module):
    """Primitive Division Operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 2

    def forward(self, x, y, epsilon=1e-6):
        return torch.mul(x, y+epsilon)


class PrimitiveMax(nn.Module):
    """Primitive Max Operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 2

    def forward(self, x, y):
        return torch.max(x, y)

class PrimitiveSigmoid(nn.Module):
    """Primitive sigmoid Operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.sigmoid(x)

class PrimitiveNeg(nn.Module):
    """Primitive negation operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return x * -1

class PrimitiveTanh(nn.Module):
    """Primitive tanh operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.tanh(x)

class PrimitiveExp(nn.Module):
    """Primitive exp operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.exp(x)

class PrimitiveLog(nn.Module):
    """Primitive log operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.sign(x) * torch.log(torch.abs(x))

class PrimitiveAbs(nn.Module):
    """Primitive abs operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.abs(x)

class PrimitiveSquare(nn.Module):
    """Primitive square operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.square(x)

class PrimitiveSqrt(nn.Module):
    """Primitive sqrt operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))    

################### Primitive mean

class PrimitiveMeanBWH(nn.Module):
    """Primitive mean operation along b, w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(x, dim=[0, 2, 3], keepdim=True).expand_as(x)

class PrimitiveMeanWHC(nn.Module):
    """Primitive mean operation along w, h, c dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(x, dim=[1, 2, 3], keepdim=True).expand_as(x)

class PrimitiveMeanWH(nn.Module):
    """Primitive mean operation along w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(x, dim=[2, 3], keepdim=True).expand_as(x)

class PrimitiveMeanWHCGrouped(nn.Module):
    """Primitive mean operation along w, h, cgrouped dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        num_channels = x.size(1)
        channel_grp = torch.empty_like(x)
        for channel in range(num_channels):
            grouped_mean = torch.mean(x[:,channel,:,:], dim=[1,2], keepdim=True)
            channel_grp[:,channel,:,:] = grouped_mean.expand_as(x[:,channel,:,:])
        
        return channel_grp


################### Primitive STD

class PrimitiveStdBWH(nn.Module):
    """Primitive std operation along b, w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(torch.square(x), dim=[0, 2, 3], keepdim=True).expand_as(x)

class PrimitiveStdWHC(nn.Module):
    """Primitive Std operation along w, h, c dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(torch.square(x), dim=[1, 2, 3], keepdim=True).expand_as(x)

class PrimitiveStdWH(nn.Module):
    """Primitive Std operation along w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(torch.square(x), dim=[2, 3], keepdim=True).expand_as(x)

class PrimitiveStdWHCGrouped(nn.Module):
    """Primitive Std operation along w, h, cgrouped dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        num_channels = x.size(1)
        channel_grp = torch.empty_like(x)
        for channel in range(num_channels):
            grouped_Std = torch.mean(torch.square(x[:,channel,:,:]), dim=[1,2], keepdim=True)
            channel_grp[:,channel,:,:] = grouped_Std.expand_as(x[:,channel,:,:])
        
        return channel_grp

################### Primitive STD Centered

class PrimitiveStdcenteredBWH(nn.Module):
    """Primitive std centered operation along b, w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.std(x, dim=[0, 2, 3], keepdim=True).expand_as(x)

class PrimitivestdcenteredWHC(nn.Module):
    """Primitive stdcentered operation along w, h, c dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.std(x, dim=[1, 2, 3], keepdim=True).expand_as(x)

class PrimitivestdcenteredWH(nn.Module):
    """Primitive stdcentered operation along w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.std(x, dim=[2, 3], keepdim=True).expand_as(x)

class PrimitivestdcenteredWHCGrouped(nn.Module):
    """Primitive stdcentered operation along w, h, cgrouped dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        num_channels = x.size(1)
        channel_grp = torch.empty_like(x)
        for channel in range(num_channels):
            grouped_stdcentered = torch.std(x[:,channel,:,:], dim=[1,2], keepdim=True)
            channel_grp[:,channel,:,:] = grouped_stdcentered.expand_as(x[:,channel,:,:])
        
        return channel_grp


PRIMITIVES = [
    PrimitiveAdd,
    PrimitiveMul,
    PrimitiveDiv,
    PrimitiveMax,
    PrimitiveNeg,
    PrimitiveSigmoid,
    PrimitiveTanh,
    PrimitiveExp,
    PrimitiveLog,
    PrimitiveAbs,
    PrimitiveSquare,
    PrimitiveSqrt,

    # Aggregate layers
    [PrimitiveMeanBWH,
    PrimitiveMeanWHC,
    PrimitiveMeanWH,
    PrimitiveMeanWHCGrouped,
    PrimitiveStdBWH,
    PrimitiveStdWHC,
    PrimitiveStdWH,
    PrimitiveStdWHCGrouped,
    PrimitiveStdcenteredBWH,
    PrimitivestdcenteredWHC,
    PrimitivestdcenteredWH,
    PrimitivestdcenteredWHCGrouped]
]