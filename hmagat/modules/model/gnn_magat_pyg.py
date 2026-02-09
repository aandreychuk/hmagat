import typing
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import math

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    scatter,
)
from torch_geometric.utils.sparse import set_sparse_value

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload

import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.utils.num_nodes import maybe_num_nodes

def masked_softmax(
    src: Tensor,
    mask: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> Tensor:
    if ptr is not None and (
        ptr.dim() == 1
        or (ptr.dim() > 1 and index is None)
        or (torch_geometric.typing.WITH_TORCH_SCATTER and not is_compiling())
    ):
        raise NotImplementedError("Yet to implement this.")
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce="max")
        out = src - src_max.index_select(dim, index)
        out = out.exp()
        out = out * mask
        out_sum = scatter(out, index, dim, dim_size=N, reduce="sum") + 1e-16
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError("'softmax' requires 'index' to be specified")

    return out / out_sum


class MAGATMultiplicativeConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
        residual: bool = False,
        use_edge_attr_for_messages: Optional[str] = None,
        reset_parameters_init: bool = True,
        same_lins_for_src_and_dst: bool = False,
        scaled_product: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.same_lins_for_src_and_dst = same_lins_for_src_and_dst

        self.scaled_product = scaled_product
        if scaled_product:
            num_dims = in_channels
            if not isinstance(in_channels, int):
                num_dims = in_channels[0]
            self.scale_coef = math.sqrt(num_dims)

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = Linear(
                in_channels,
                heads * out_channels,
                bias=False,
                weight_initializer="glorot",
            )
            self.lin_att = Linear(
                in_channels,
                heads * in_channels,
                bias=False,
                weight_initializer="glorot",
            )
        else:
            if in_channels[0] != in_channels[1]:
                assert not same_lins_for_src_and_dst

            self.in_channels = in_channels[1]

            self.lin_src = Linear(
                in_channels[0], heads * out_channels, False, weight_initializer="glorot"
            )
            if same_lins_for_src_and_dst:
                self.lin_dst = self.lin_src
            else:
                self.lin_dst = Linear(
                    in_channels[1],
                    heads * out_channels,
                    False,
                    weight_initializer="glorot",
                )
            self.lin_att = Linear(
                in_channels[0],
                heads * in_channels[1],
                bias=False,
                weight_initializer="glorot",
            )

        self.lin_edge, self.lin_att_edge = None, None
        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim,
                heads * self.out_channels,
                bias=False,
                weight_initializer="glorot",
            )
            self.lin_att_edge = Linear(
                edge_dim,
                heads * self.in_channels,
                bias=False,
                weight_initializer="glorot",
            )

        self.use_edge_attr_for_messages = use_edge_attr_for_messages

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer="glorot",
            )
        else:
            self.register_parameter("res", None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter("bias", None)

        if reset_parameters_init:
            self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_src is not None:
            self.lin_src.reset_parameters()
        if self.lin_dst is not None and not self.same_lins_for_src_and_dst:
            self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        if self.lin_att is not None:
            self.lin_att.reset_parameters()
        if self.lin_att_edge is not None:
            self.lin_att_edge.reset_parameters()
        zeros(self.bias)

    @overload
    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        edge_weight: OptTensor = None,
        size: Size = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        edge_weight: OptTensor = None,
        size: Size = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        edge_weight: OptTensor = None,
        size: Size = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        edge_weight: OptTensor = None,
        size: Size = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            size ((int, int), optional): The shape of the adjacency matrix.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.res is not None:
                res = self.res(x)

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                # If the module is initialized as bipartite, transform source
                # and destination node features separately:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)

        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"

            if x_dst is not None and self.res is not None:
                res = self.res(x_dst)

            if self.lin is not None:
                # If the module is initialized as non-bipartite, we expect that
                # source and destination node features have the same shape and
                # that they their transformations are shared:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        # alpha_src = (self.lin_att(x).view(-1, H, C) * self.att_src).sum(dim=-1)
        # alpha_dst = (
        #     None
        #     if x_dst is None
        #     else (self.lin_att(x).view(-1, H, C) * self.att_dst).sum(-1)
        # )
        if isinstance(x, Tensor):
            alpha = (x, x)
        else:
            alpha = x

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor, edge_weight: OptTensor)
        alpha = self.edge_updater(
            edge_index,
            alpha=alpha,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            size=size,
        )

        x = (x_src, x_dst)
        if self.use_edge_attr_for_messages is not None:
            if self.lin_edge is not None and edge_attr is not None:
                edge_attr = self.lin_edge(edge_attr).view(-1, H, C)
        else:
            edge_attr = None

        # propagate_type: (x: OptPairTensor, alpha: Tensor, edge_attr: OptTensor)
        out = self.propagate(
            edge_index, x=x, alpha=alpha, edge_attr=edge_attr, size=size
        )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def edge_update(
        self,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        edge_attr: OptTensor,
        edge_weight: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        dim_size: Optional[int],
    ) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        E = alpha_i.shape[0]
        alpha_i = alpha_i.reshape((E, self.in_channels, 1))
        alpha_j = self.lin_att(alpha_j).reshape((E, self.heads, self.in_channels))
        if edge_attr is not None and self.lin_att_edge is not None:
            edge_attr = self.lin_att_edge(edge_attr)
            edge_attr = edge_attr.reshape((E, self.heads, self.in_channels))
            alpha_j += edge_attr
        alpha = torch.bmm(alpha_j, alpha_i)
        if self.scaled_product:
            alpha = alpha / self.scale_coef
        alpha = torch.squeeze(alpha, dim=-1)
        if index.numel() == 0:
            return alpha

        alpha = F.leaky_relu(alpha, self.negative_slope)
        if edge_weight is not None:
            # To apply mask, we need to use rolled out softmax
            alpha = masked_softmax(alpha, edge_weight, index, ptr, dim_size)
        else:
            alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor, edge_attr: OptTensor) -> Tensor:
        if edge_attr is not None:
            x_j = x_j + edge_attr
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class DirectionalHMAGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim=None,
        heads=1,
        hyperedge_feature_generator="magat",
        final_feature_generator="magat",
        residual=True,
        use_edge_attr_for_messages=None,
        scaled_product=False,
    ):
        super().__init__()

        if hyperedge_feature_generator == "magat":
            self.node_to_hyperedge = MAGATMultiplicativeConv(
                in_channels=[in_channels, in_channels],
                out_channels=out_channels,
                heads=heads,
                add_self_loops=False,
                edge_dim=edge_dim,
                residual=False,
                bias=False,
                use_edge_attr_for_messages=use_edge_attr_for_messages,
                same_lins_for_src_and_dst=True,
                scaled_product=scaled_product,
            )
        else:
            raise ValueError(
                f"{hyperedge_feature_generator} Hyperedge Feature Generator not supported."
            )

        if final_feature_generator == "magat":
            self.hyperedge_to_node = MAGATMultiplicativeConv(
                in_channels=[out_channels * heads, in_channels],
                out_channels=out_channels,
                heads=heads,
                add_self_loops=False,
                edge_dim=edge_dim,
                residual=residual,
                use_edge_attr_for_messages=use_edge_attr_for_messages,
                scaled_product=scaled_product,
            )
        else:
            raise ValueError(
                f"{final_feature_generator} Final Feature Generator not supported."
            )

    def reset_parameters(self):
        self.node_to_hyperedge.reset_parameters()
        self.hyperedge_to_node.reset_parameters()

    def forward(
        self,
        x,
        edge_index_src,
        edge_index_dst,
        hton_edge_index_src,
        hton_edge_index_dst,
        edge_attr=None,
        edge_weight=None,
        hton_edge_attr=None,
        hton_edge_weight=None,
    ):
        # First, we compute the hyperedge features (based on the head)
        hyperedge_attr = x[hton_edge_index_dst]
        hyperedge_attr = scatter(
            hyperedge_attr, hton_edge_index_src, dim=0, reduce="mean"
        )

        # Node -> Hyperedge
        edge_index = torch.stack([edge_index_src, edge_index_dst], dim=0)
        hyperedge_attr = self.node_to_hyperedge(
            (x, hyperedge_attr),
            edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
        )

        # Hyperedge -> Node
        edge_index = torch.stack([hton_edge_index_src, hton_edge_index_dst], dim=0)
        x = self.hyperedge_to_node(
            (hyperedge_attr, x),
            edge_index,
            edge_attr=hton_edge_attr,
            edge_weight=hton_edge_weight,
        )
        return x
