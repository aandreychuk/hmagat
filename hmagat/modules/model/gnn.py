import torch
import torch.nn.functional as F

from hmagat.modules.model.gnn_magat_pyg import MAGATMultiplicativeConv, DirectionalHMAGAT


class GNNWrapper(torch.nn.Module):
    def __init__(
        self,
        gnn,
        use_edge_weights=False,
        use_edge_attr=False,
        access_graph_index=False,
        two_edge_index=False,
    ):
        super().__init__()
        self.use_edge_weights = use_edge_weights
        self.use_edge_attr = use_edge_attr
        self.access_graph_index = access_graph_index
        self.gnn = gnn
        self.two_edge_index = two_edge_index

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, data, **kwargs):
        args = [x]
        forward_kwargs = dict()
        if "edge_index" in kwargs:
            if self.two_edge_index:
                args.append(kwargs["edge_index"][0])
                args.append(kwargs["edge_index"][1])
            else:
                args.append(kwargs["edge_index"])
        elif self.access_graph_index:
            args.append(data.graph_edge_index)
        elif self.two_edge_index:
            args = args + [data.edge_index_src, data.edge_index_dst]
            args = args + [data.hton_edge_index_src, data.hton_edge_index_dst]
            if "hton_edge_attr" in data:
                forward_kwargs["hton_edge_attr"] = data.hton_edge_attr
        else:
            args.append(data.edge_index)

        if self.use_edge_attr:
            args.append(
                kwargs["edge_attr"] if "edge_attr" in kwargs else data.edge_attr
            )
        if self.use_edge_weights:
            if "edge_weight" in kwargs:
                edge_weights = kwargs["edge_weight"]
            elif self.access_graph_index:
                edge_weights = data.graph_edge_weight
            else:
                edge_weights = data.edge_weight
            args.append(edge_weights)
        return self.gnn(*args, **forward_kwargs)


def GNNFactory(
    in_channels,
    out_channels,
    num_attention_heads,
    model_type="MAGAT",
    use_edge_weights=False,
    use_edge_attr=False,
    edge_dim=None,
    residual=None,
    access_graph_index=False,
    **model_kwargs,
):
    if use_edge_attr:
        assert (
            edge_dim is not None
        ), "Expecting edge_dim to be given if using edge attributes"
    elif use_edge_weights:
        edge_dim = 1
    else:
        assert (
            edge_dim is None
        ), f"Not using edge attr or weights, so expect node_dim to be None, but got {edge_dim}"
    kwargs = dict()
    if edge_dim is not None:
        kwargs = kwargs | {"edge_dim": edge_dim}
    if residual is not None:
        kwargs = kwargs | {"residual": residual}
    if "bias" in model_kwargs:
        kwargs = kwargs | {"bias": model_kwargs["bias"]}

    two_edge_index = False
    if model_type == "DirectionalHMAGAT":
        two_edge_index = True

    def _factory():
        if model_type == "MAGAT":
            use_edge_attr_for_messages = None
            if "use_edge_attr_for_messages" in model_kwargs:
                use_edge_attr_for_messages = model_kwargs[
                    "use_edge_attr_for_messages"
                ]
            return MAGATMultiplicativeConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_attention_heads,
                use_edge_attr_for_messages=use_edge_attr_for_messages,
                **kwargs,
            )
        elif model_type == "DirectionalHMAGAT":
            use_edge_attr_for_messages = None
            if "use_edge_attr_for_messages" in model_kwargs:
                use_edge_attr_for_messages = model_kwargs["use_edge_attr_for_messages"]
            return DirectionalHMAGAT(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_attention_heads,
                hyperedge_feature_generator=model_kwargs["hyperedge_feature_generator"],
                use_edge_attr_for_messages=use_edge_attr_for_messages,
                final_feature_generator=model_kwargs["final_feature_generator"],
                **kwargs,
            )
        else:
            raise ValueError(f"Currently, we don't support model: {model_type}")

    return GNNWrapper(
        _factory(),
        use_edge_weights=use_edge_weights,
        use_edge_attr=use_edge_attr,
        access_graph_index=access_graph_index,
        two_edge_index=two_edge_index,
    )


class GNNModule(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        embedding_sizes,
        num_attention_heads,
        num_gnn_layers=None,
        model_type="MAGAT",
        use_edge_weights=False,
        use_edge_attr=False,
        edge_dim=None,
        model_residuals=None,
        **model_kwargs,
    ):
        super().__init__()
        first_residual = None
        rest_residuals = None
        if model_residuals == "first":
            first_residual = True
        elif model_residuals == "only-first":
            first_residual = True
            rest_residuals = False
        elif model_residuals == "all":
            first_residual = True
            rest_residuals = True
        elif model_residuals == "none":
            first_residual = False
            rest_residuals = False
        elif model_residuals is not None:
            raise ValueError(f"Unsupported model residuals option: {model_residuals}")

        if num_gnn_layers is None:
            num_gnn_layers = len(embedding_sizes)
        else:
            assert num_gnn_layers == len(embedding_sizes)

        graph_convs = []
        graph_convs.append(
            GNNFactory(
                in_channels=in_channels,
                out_channels=embedding_sizes[0],
                model_type=model_type,
                num_attention_heads=num_attention_heads,
                use_edge_weights=use_edge_weights,
                use_edge_attr=use_edge_attr,
                edge_dim=edge_dim,
                residual=first_residual,
                **model_kwargs,
            )
        )

        for i in range(num_gnn_layers - 1):
            graph_convs.append(
                GNNFactory(
                    in_channels=num_attention_heads * embedding_sizes[i],
                    out_channels=embedding_sizes[i + 1],
                    model_type=model_type,
                    num_attention_heads=num_attention_heads,
                    use_edge_weights=use_edge_weights,
                    use_edge_attr=use_edge_attr,
                    edge_dim=edge_dim,
                    residual=rest_residuals,
                    **model_kwargs,
                )
            )

        self.graph_convs = torch.nn.ModuleList(graph_convs)

    def in_simulation(self, value):
        self.simulation = value

    def reset_parameters(self):
        for conv in self.graph_convs:
            conv.reset_parameters()

    def forward(self, x, data, **kwargs):
        for conv in self.graph_convs:
            x = conv(x, data, **kwargs)
            x = F.relu(x)

        return x
