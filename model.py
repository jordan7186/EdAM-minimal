import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, global_add_pool
from torch.nn import (
    Module,
    BatchNorm1d,
    LayerNorm,
    ModuleList,
    Linear,
)
from torch_geometric.nn.models import MLP
from torch_geometric.data import Data


class GraphTransformer(Module):
    def __init__(
        self,
        channels: int,
        pe_dim: int,  # Positional Encoding dimension INSIDE the model
        num_layers: int,
        num_heads: int,
        node_feature_dim: int,
        num_classes: int,
        pe_enc_dim: int,  # Positional Encoding dimension in the INPUT GRAPH
        norm_type: str = "batch",  # "batch" or "layer" for PE normalization.
        use_mlp_classifier: bool = False,  # Whether to use a multi-layer classifier (final MLP).
        mlp_layers: int = 3,  # Total number of layers in the classifier.
        mlp_hidden_dim: int = None,  # Hidden dimension for the classifier (default: channels//2).
        use_transformer_mlp: bool = False,  # Whether to include an MLP after each transformer block.
        transformer_mlp_layers: int = 2,  # Number of layers for each transformer block MLP.
        transformer_mlp_hidden_dim: int = None,  # Hidden dimension for transformer block MLP (default: channels//2).
        use_hybrid: bool = False,  # Whether to include a GATConv branch.
        gat_conv_kwargs: dict = None,  # Keyword arguments for the GATConv layer.
    ):
        """
        Flexible Graph Transformer for graph classification with configurable MLPs.
        This version uses torch_geometric.nn.MLP for the final classifier and, optionally,
        an MLP after each transformer block.
        """
        super().__init__()

        self.channels = channels
        self.pe_dim = pe_dim
        self.use_hybrid = use_hybrid
        self.use_mlp_classifier = use_mlp_classifier
        self.use_transformer_mlp = use_transformer_mlp
        self.num_layers = num_layers

        # === Node Embedding ===
        # (Always a Linear layer in this version.)
        self.node_emb = Linear(node_feature_dim, channels - pe_dim, bias=False)

        # === Positional Encoding ===
        if norm_type == "batch":
            self.pe_norm = BatchNorm1d(pe_enc_dim)
        elif norm_type == "layer":
            self.pe_norm = LayerNorm(pe_enc_dim)
        else:
            raise ValueError("Unsupported norm_type. Use 'batch' or 'layer'.")
        self.pe_lin = Linear(pe_enc_dim, pe_dim)

        # === Transformer Branch: Multi-Head Attention Layers ===
        self.attentions = ModuleList()
        # If requested, create an MLP for each transformer block.
        if self.use_transformer_mlp:
            self.transformer_mlps = ModuleList()
        for _ in range(num_layers):
            self.attentions.append(
                torch.nn.MultiheadAttention(
                    embed_dim=channels,
                    num_heads=num_heads,
                    bias=False,
                    batch_first=True,
                )
            )
            if self.use_transformer_mlp:
                # Build an MLP that goes from channels -> hidden_dim -> channels.
                self.transformer_mlps.append(
                    MLP(
                        [channels, transformer_mlp_hidden_dim, channels],
                        num_layers=transformer_mlp_layers,
                    )
                )

        # === Parallel GATConv Branch per Layer ===
        if self.use_hybrid:
            self.gat_convs = ModuleList()
            # Ensure gat_conv_kwargs is a dict
            if gat_conv_kwargs is None:
                gat_conv_kwargs = {}
            for _ in range(num_layers):
                self.gat_convs.append(
                    GATConv(
                        in_channels=channels, out_channels=channels, **gat_conv_kwargs
                    )
                )
        # === Final Classifier ===
        if use_mlp_classifier:
            print("Using MLP classifier.")
            self.classifier = MLP(
                in_channels=channels,
                out_channels=num_classes,
                hidden_channels=mlp_hidden_dim,
                num_layers=mlp_layers,
            )
        else:
            self.classifier = Linear(in_features=channels, out_features=num_classes)

    def forward(
        self,
        data: Data,
        batch: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for graph classification.
        For each transformer layer, a parallel GATConv branch is executed (if enabled)
        and the outputs are combined.
        """
        # Unpack from data object
        x: Tensor = data.x.float()
        pe: Tensor = data.pe.float()
        edge_index: Tensor = data.edge_index.long()
        edge_attr: Tensor | None = data.edge_attr.float()

        # --- Node Feature & Positional Encoding Processing ---
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
        node_repr = self.node_emb(x)  # (N, channels - pe_dim)
        if self.pe_norm is not None:
            pe_normed = self.pe_norm(pe)
        else:
            pe_normed = pe
        pe_repr = self.pe_lin(pe_normed)  # (N, pe_dim)
        # Initial node representation: (N, channels)
        x_in = torch.cat((node_repr, pe_repr), dim=1)

        # --- Per-Layer Processing ---
        for i in range(self.num_layers):
            # Transformer branch:
            # MultiheadAttention expects input shape (batch, seq_len, embed_dim).
            x_attn, _ = self.attentions[i](
                x_in.unsqueeze(0),
                x_in.unsqueeze(0),
                x_in.unsqueeze(0),
                need_weights=False,
            )
            x_attn = F.relu(x_attn).squeeze(0)  # (N, channels)
            if self.use_transformer_mlp:
                x_attn = self.transformer_mlps[i](x_attn)
                x_attn = F.relu(x_attn)

            # Hybrid branch: parallel GATConv
            if self.use_hybrid:
                assert edge_index is not None, "edge_index must be provided."
                x_gat = self.gat_convs[i](x_in, edge_index, edge_attr)  # (N, channels)
                # Combine transformer and GAT outputs.
                x_out = x_attn + x_gat
            else:
                x_out = x_attn

            # Prepare input for next layer.
            x_in = x_out

        # --- Global Pooling & Classification ---
        x_pool = global_add_pool(x_in, batch)
        out = self.classifier(x_pool)
        return out

    @torch.no_grad()
    def get_attention(
        self,
        x: Tensor,
        pe: Tensor,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ):
        """
        Returns a dictionary with attention weights from each transformer layer,
        and, if hybrid mode is enabled, from each parallel GATConv branch.
        The computation mirrors the forward pass.
        """
        # --- Node Feature & Positional Encoding Processing ---
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
        node_repr = self.node_emb(x)  # (N, channels - pe_dim)
        if self.pe_norm is not None:
            pe_normed = self.pe_norm(pe)
        else:
            pe_normed = pe
        pe_repr = self.pe_lin(pe_normed)  # (N, pe_dim)
        # Initial node representation: (N, channels)
        x_in = torch.cat((node_repr, pe_repr), dim=1)

        transformer_attn_weights, gat_attn_weights_list = [], []

        # --- Per-Layer Processing ---
        for i in range(self.num_layers):
            # Transformer branch:
            # MultiheadAttention expects input shape (batch, seq_len, embed_dim).
            x_attn, gt_att = self.attentions[i](
                x_in.unsqueeze(0),
                x_in.unsqueeze(0),
                x_in.unsqueeze(0),
                need_weights=True,
            )
            x_attn = F.relu(x_attn).squeeze(0)  # (N, channels)
            transformer_attn_weights.append(gt_att)
            if self.use_transformer_mlp:
                x_attn = self.transformer_mlps[i](x_attn)
                x_attn = F.relu(x_attn)

            # Hybrid branch: parallel GATConv
            if self.use_hybrid:
                assert edge_index is not None, "edge_index must be provided."
                x_gat, gat_att = self.gat_convs[i](
                    x_in, edge_index, edge_weight, return_attention_weights=True
                )  # (N, channels)
                # Combine transformer and GAT outputs.
                gat_attn_weights_list.append(gat_att)
                x_out = x_attn + x_gat
            else:
                x_out = x_attn

            # Prepare input for next layer.
            x_in = x_out

        return {
            "transformer": transformer_attn_weights,
            "gat": gat_attn_weights_list if self.use_hybrid else None,
        }
