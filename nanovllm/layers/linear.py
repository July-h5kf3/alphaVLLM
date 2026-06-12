import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.bias_loader
        else:
            self.register_parameter("bias", None)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

#每个rank都包含一份完整的权重参数，forward时每个rank都进行完整的线性变换
class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

#每个rank只包含权重参数的一部分，每个rank都是输出通道的一部分
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim) 
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)#沿tp_dim维度切分权重
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

#把多个ColumnParallelLinear合并为一个大Linear，但加载权重时还能按原来的几个子矩阵分别塞进去。
"""
这里会有一个疑问就是为什么每个rank上都要有子权重的一部分,可以拿gate和up这两个矩阵合并来看一看:

在SWiGLU中,我们要进行的计算是:
    silu(gate[i]) * up[i]
其中gate[i] = linear_{gate}(input)[i], up[i] = linear_up(input)[i]
我们可以仿照QKV合并的做法,把gate和up合并为一个大的权重
如果我们每个rank上要么有gate的子权重,要么有up的子权重,那每个rank上就只能计算silu(gate[i]) * up[i]中的一个部分,无法完成整个计算
而如果每个rank上既有gate的子权重又有up的子权重,那每个rank上就可以计算silu(gate[i]) * up[i]中的一个部分,最后再把所有rank上的结果加起来就可以得到完整的结果了
"""
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        #loaded_shard_id表示当前传进来的loaded_weight是原来子矩阵中的第几个
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        #该子权重应该放到当前rank本地大参数的哪个位置
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

"""
假设hidden_size=1024, head_size=128, total_num_heads=32 total_num_kv_heads=32 tp_size=4
那么每个rank可以拿到:
num_heads = 32 // 4 = 8
num_kv_heads = 32 // 4 = 8
每个rank本地的q/k/v输出大小是:
q: 8 * 128 = 1024
k: 8 * 128 = 1024
v: 8 * 128 = 1024
所以每个rank上合并后的权重大小是:
output_size = (8 + 8 + 8) * 128 = 3072
内部的布局是:
[
    q_head_0, q_head_1, ..., q_head_7,#1024行
    k_head_0, k_head_1, ..., k_head_7,
    v_head_0, v_head_1, ..., v_head_7,
]
"""
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

#和ColumnParallelLinear反过来，但是因为每个rank上都要有子权重的一部分，所以每个rank上都要进行完整的线性变换，最后再把所有rank上的结果加起来
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y