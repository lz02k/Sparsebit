"""
shape.py 文件存放所有关于tensor shape操作的算子, 这些算子在Quant过程不参与量化,
故在该文件的类实现只是把function转为module.
"""


import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[torch.flatten, torch.Tensor.flatten, nn.Flatten])
class Flatten(nn.Module):
    """量化Flatten层。认为flatten不改变运算前后值域范围,所以不做量化。

    是QuantOpr的子类。

    Attributes:
        start_dim (any): 同 ``torch.nn.Flatten`` 。
        end_dim (any): 同 ``torch.nn.Flatten`` 。
    """

    def __init__(self, org_module=None, config=None):
        super(Flatten, self).__init__()
        if isinstance(org_module, torch.fx.Node):
            start_dim = org_module.args[1]
            end_dim = org_module.args[2] if len(org_module.args) == 3 else -1
            self.start_dim = start_dim
            self.end_dim = end_dim
        else:
            self.start_dim = org_module.start_dim
            self.end_dim = org_module.end_dim

    def forward(self, x_in, *args):
        """Flatten层的前向传播,不做量化。"""
        out = torch.flatten(x_in, self.start_dim, self.end_dim)
        return out


@register_qmodule(sources=[torch.Tensor.size])
class Size(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(Size, self).__init__()
        if isinstance(org_module, torch.fx.Node):
            if "dim" in org_module.kwargs:
                self.dim = org_module.kwargs["dim"]
            elif len(org_module.args) == 2:
                self.dim = org_module.args[1]
            else:
                self.dim = None
        else:
            self.dim = None

    def forward(self, x, *args, **kwargs):
        return x.size(dim=self.dim)


@register_qmodule(sources=[torch.reshape, torch.Tensor.reshape, torch.Tensor.view])
class Reshape(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(Reshape, self).__init__()

    def forward(self, x_in, *args):
        return torch.reshape(x_in, args)


@register_qmodule(sources=[torch.cat])
class Concat(nn.Module):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        self.dim = org_module.args[1]
        self._repr_info = "Concat"

    def forward(self, x_in, *args, **kwargs):
        out = torch.cat(x_in, dim=self.dim)
        return out
