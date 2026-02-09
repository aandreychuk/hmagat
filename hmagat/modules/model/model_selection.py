from hmagat.modules.model.gnn import GNNModule


def get_gnn_module(
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
    return GNNModule(
        in_channels=in_channels,
        embedding_sizes=embedding_sizes,
        num_attention_heads=num_attention_heads,
        num_gnn_layers=num_gnn_layers,
        model_type=model_type,
        use_edge_weights=use_edge_weights,
        use_edge_attr=use_edge_attr,
        edge_dim=edge_dim,
        model_residuals=model_residuals,
        **model_kwargs,
    )
