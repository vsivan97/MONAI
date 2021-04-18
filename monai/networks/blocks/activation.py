# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F

class AdaptiveActivation(nn.Module):
    """Adaptive activation layer
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.activations = [
            nn.ELU(),
            nn.Hardshrink(),
            nn.Hardtanh(),
            nn.LeakyReLU(),
            nn.LogSigmoid(),
            nn.ReLU(),
            nn.PReLU(),
            nn.SELU(),
            nn.CELU(),
            nn.Sigmoid(),
            nn.Softplus(),
            nn.Softshrink(),
            nn.Softsign(),
            nn.Tanh(),
            nn.Tanhshrink()
        ]

        self.P = [torch.nn.Parameter(torch.randn(1, requires_grad=True))
                  for _ in self.activations]

        for activation, param in zip(self.activations, self.P):
            activation_name = str(activation).split("(")[0]
            self.add_module(name=activation_name, module=activation)
            self.register_parameter(name=activation_name + "p", param=param)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(input)
        for activation, param in zip(self.activations, self.P):
            out += param.clamp(-1, 1) * activation(input)
        return out


class Swish(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Swish}(x) = x * \text{Sigmoid}(\alpha * x) for constant value alpha.

    Citation: Searching for Activation Functions, Ramachandran et al., 2017, https://arxiv.org/abs/1710.05941.


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input


    Examples::

        >>> m = Act['swish']()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(self.alpha * input)


class Mish(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Mish}(x) = x * tanh(\text{softplus}(x)).

    Citation: Mish: A Self Regularized Non-Monotonic Activation Function, Diganta Misra, 2019, https://arxiv.org/abs/1908.08681.


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input


    Examples::

        >>> m = Act['mish']()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.tanh(torch.nn.functional.softplus(input))
