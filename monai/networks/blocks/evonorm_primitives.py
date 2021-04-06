import torch
import torch.nn as nn

class EvonormPrimitive(nn.Module):
    """Abstract base class for the evonorm primitive layers
    """
    def __init__(self):
        if type(self) is EvonormPrimitive:
            raise NotImplementedError
        super().__init__()

class PrimitiveAdd(EvonormPrimitive):
    """ Primitive Add Operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 2

    def forward(self, x, y):
        return torch.add(x, y)

class PrimitiveMul(EvonormPrimitive):
    """Primitive Multiply Operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 2

    def forward(self, x, y):
        return torch.mul(x, y)

class PrimitiveDiv(EvonormPrimitive):
    """Primitive Division Operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 2

    def forward(self, x, y, epsilon=1e-6):
        return torch.div(x, y+epsilon)


class PrimitiveMax(EvonormPrimitive):
    """Primitive Max Operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 2

    def forward(self, x, y):
        return torch.max(x, y)

class PrimitiveSigmoid(EvonormPrimitive):
    """Primitive sigmoid Operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.sigmoid(x)

class PrimitiveNeg(EvonormPrimitive):
    """Primitive negation operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return x * -1

class PrimitiveTanh(EvonormPrimitive):
    """Primitive tanh operation
    """

    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.tanh(x)

class PrimitiveExp(EvonormPrimitive):
    """Primitive exp operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.exp(x)

class PrimitiveLog(EvonormPrimitive):
    """Primitive log operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.sign(x) * torch.log(torch.abs(x))

class PrimitiveAbs(EvonormPrimitive):
    """Primitive abs operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.abs(x)

class PrimitiveSquare(EvonormPrimitive):
    """Primitive square operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.square(x)

class PrimitiveSqrt(EvonormPrimitive):
    """Primitive sqrt operation
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

################### Primitive mean

class PrimitiveMeanBWH(EvonormPrimitive):
    """Primitive mean operation along b, w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(x, dim=[0, 2, 3], keepdim=True).expand_as(x)

class PrimitiveMeanWHC(EvonormPrimitive):
    """Primitive mean operation along w, h, c dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(x, dim=[1, 2, 3], keepdim=True).expand_as(x)

class PrimitiveMeanWH(EvonormPrimitive):
    """Primitive mean operation along w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(x, dim=[2, 3], keepdim=True).expand_as(x)

class PrimitiveMeanWHCGrouped(EvonormPrimitive):
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

class PrimitiveStdBWH(EvonormPrimitive):
    """Primitive std operation along b, w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(torch.square(x), dim=[0, 2, 3], keepdim=True).expand_as(x)

class PrimitiveStdWHC(EvonormPrimitive):
    """Primitive Std operation along w, h, c dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(torch.square(x), dim=[1, 2, 3], keepdim=True).expand_as(x)

class PrimitiveStdWH(EvonormPrimitive):
    """Primitive Std operation along w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.mean(torch.square(x), dim=[2, 3], keepdim=True).expand_as(x)

class PrimitiveStdWHCGrouped(EvonormPrimitive):
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

class PrimitiveStdcenteredBWH(EvonormPrimitive):
    """Primitive std centered operation along b, w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.std(x, dim=[0, 2, 3], keepdim=True).expand_as(x)

class PrimitivestdcenteredWHC(EvonormPrimitive):
    """Primitive stdcentered operation along w, h, c dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.std(x, dim=[1, 2, 3], keepdim=True).expand_as(x)

class PrimitivestdcenteredWH(EvonormPrimitive):
    """Primitive stdcentered operation along w, h dimensions
    """
    def __init__(self):
        super().__init__()
        self.arity = 1

    def forward(self, x):
        # expect input in B, C, H, W format
        return torch.std(x, dim=[2, 3], keepdim=True).expand_as(x)

class PrimitivestdcenteredWHCGrouped(EvonormPrimitive):
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
