#!/usr/bin/env python
import os
import sys
import argparse
import torch
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm
from scipy.stats import pearsonr, kendalltau
from copy import deepcopy

from model import GraphTransformer as GTBase
from utils import (
    get_attention_matrix_GT,
    generate_random_walk_attr,
    generate_laplacian_pe,
)


###############################
# Helper Functions
###############################
def edge_index_to_dense_adj(edge_index: Tensor, add_self_loops: bool = True) -> Tensor:
    num_nodes = edge_index.max().item() + 1
    A = torch.zeros(
        (num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device
    )
    A[edge_index[1], edge_index[0]] = 1.0
    if add_self_loops:
        A += torch.eye(num_nodes, device=edge_index.device)
    return A


def measure_model_output_difference(
    original_logits: Tensor, manipulated_logits: Tensor
) -> int:
    original_class = torch.argmax(original_logits, dim=-1)
    manipulated_class = torch.argmax(manipulated_logits, dim=-1)
    prediction_diff = int((original_class != manipulated_class).item())
    return prediction_diff


def EdAM_s_attribution_with_popularity(
    adj_matrix: Tensor,
    att_matrix: Tensor,
    Kmax: int,
    adjust_popularity: bool = True,
    epsilon: float = 1e-4,
    device: str = "cuda",
) -> Tensor:
    A = adj_matrix.clone().detach().requires_grad_(True)
    AK = torch.matrix_power(A, Kmax).to(device)
    with torch.no_grad():
        B = torch.zeros_like(AK, device=device)
        nonzero_mask = AK != 0
        B[nonzero_mask] = att_matrix[nonzero_mask] / AK[nonzero_mask]
    out = (AK * B).sum()
    out.backward()
    grad_abs = A.grad.abs()
    if adjust_popularity:
        nonzero_mask = grad_abs != 0
        grad_abs[nonzero_mask] /= (
            compute_edge_popularity(adj_matrix, Kmax)[nonzero_mask] + epsilon
        )
        return grad_abs
    else:
        return grad_abs


def EdAM_i_attribution_with_popularity(
    adj_matrix: Tensor,
    att_matrix: Tensor,
    Kmax: int,
    adjust_popularity: bool = True,
    epsilon: float = 1e-4,
    device: str = "cuda",
) -> Tensor:
    if adj_matrix.device != device:
        adj_matrix = adj_matrix.to(device)
    if att_matrix.device != device:
        att_matrix = att_matrix.to(device)
    grad_accum = torch.zeros_like(adj_matrix, device=device)
    explained_matrix = torch.zeros_like(att_matrix, device=device)
    A_cum = torch.zeros_like(adj_matrix, device=device)

    for k in range(1, Kmax + 1):
        A_var = adj_matrix.clone().detach().requires_grad_(True)
        AK = torch.matrix_power(A_var, k)
        AK_no_grad = torch.matrix_power(adj_matrix, k)
        new_mask = (AK_no_grad != 0) & (A_cum == 0)
        A_cum += AK_no_grad

        with torch.no_grad():
            B_k = torch.zeros_like(AK, device=device)
            residual_k = att_matrix - explained_matrix
            B_k[new_mask] = residual_k[new_mask] / AK_no_grad[new_mask]

        if A_var.grad is not None:
            A_var.grad.zero_()
        out_k = (AK * B_k).sum()
        out_k.backward()
        curr_attribution = A_var.grad.detach().abs()
        if adjust_popularity and k > 1:
            nonzero_mask = curr_attribution != 0
            curr_attribution[nonzero_mask] /= (
                compute_edge_popularity(adj_matrix, k)[nonzero_mask] + epsilon
            )
        grad_accum += curr_attribution
        with torch.no_grad():
            explained_matrix += AK_no_grad * B_k

    return grad_accum


def compute_edge_popularity(adj_matrix: Tensor, k: int) -> Tensor:
    A_var = adj_matrix.clone().detach().requires_grad_(True)
    A_k = torch.matrix_power(A_var, k)
    total_paths = A_k.sum()
    total_paths.backward()
    popularity = A_var.grad.abs() * A_var.detach()
    return popularity


def direct_implementation(adj_matrix, att_matrix, Kmax, device="cuda", **kwargs):
    return att_matrix


@torch.no_grad()
def observe_model_reaction_single_edge(
    data: Data,
    model: GTBase,
    target_edge: tuple | Tensor,
    alpha_list: list[float],
    k: int,
    device: str = "cuda",
    pe_type: str = "RWPE",
) -> float:
    model.eval()
    model.to(device)
    data = data.to(device)
    edge_index = data.edge_index.clone().to(device)
    original_logits = model(data).cpu()

    data_temp = deepcopy(data)
    data_temp = data_temp.to(device)
    data_temp.edge_weight = torch.ones(
        data_temp.edge_index.shape[1], dtype=torch.float32, device=device
    )
    edge_index_mask = (
        (data_temp.edge_index[0] == target_edge[0])
        & (data_temp.edge_index[1] == target_edge[1])
    ) | (
        (data_temp.edge_index[0] == target_edge[1])
        & (data_temp.edge_index[1] == target_edge[0])
    )

    pred_diff_list = []
    for alpha in alpha_list:
        data_temp.edge_weight[edge_index_mask] = alpha
        if pe_type == "RWPE":
            data_temp = generate_random_walk_attr(
                data=data_temp,
                walk_length=k,
            )
        elif pe_type == "LapPE":
            data_temp = generate_laplacian_pe(
                data=data_temp,
                k=k,
            )
        else:
            raise ValueError(f"Unknown positional encoding type: {pe_type}")
        data_temp = data_temp.to(device)
        manipulated_logits = model(data_temp).cpu()
        pred_diff = measure_model_output_difference(original_logits, manipulated_logits)
        pred_diff_list.append(pred_diff)

    return sum(pred_diff_list) / len(pred_diff_list)


def calculate_correlation(
    edge_attr_from_method: Tensor,
    pred_diff_list: Tensor,
) -> torch.Tensor:
    pearson_corr_pred = pearsonr(
        edge_attr_from_method.cpu().numpy(), pred_diff_list.cpu().numpy()
    )[0]
    kendall_corr_pred = kendalltau(
        edge_attr_from_method.cpu().numpy(), pred_diff_list.cpu().numpy()
    )[0]
    corr_tensor = torch.tensor(
        [
            [pearson_corr_pred, kendall_corr_pred],
        ]
    )
    return corr_tensor


def perform_faithfulness_test_single_data(
    data: Data,
    model: GTBase,
    edge_attr_from_method: Tensor,
    alpha_list: list[float],
    k: int,
    target_edge_mask: torch.BoolTensor | None = None,
    device: str = "cuda",
    pe_type: str = "RWPE",
) -> torch.Tensor:
    model.eval()
    model.to(device)
    edge_list = data.edge_index.clone()

    if target_edge_mask is None:
        target_edge_mask = torch.ones(
            data.edge_index.shape[1], dtype=torch.bool, device=edge_list.device
        )
    edge_attr_from_method = edge_attr_from_method[target_edge_mask]
    edge_list = edge_list[:, target_edge_mask]

    pred_diff_list = []
    for i in range(edge_list.shape[1]):
        target_edge = tuple(edge_list[:, i].tolist())
        p_diff = observe_model_reaction_single_edge(
            data=data,
            model=model,
            target_edge=target_edge,
            alpha_list=alpha_list,
            k=k,
            device=device,
            pe_type=pe_type,
        )
        pred_diff_list.append(p_diff)

    pred_diff_tensor = torch.tensor(pred_diff_list)

    return calculate_correlation(
        edge_attr_from_method=edge_attr_from_method,
        pred_diff_list=pred_diff_tensor,
    )


def perform_faithfulness_test_whole_dataset(
    data_list: list[Data],
    model: GTBase,
    method,
    Kmax: int,
    alpha_list: list[float],
    k: int,
    layer_idx: int = 0,
    device: str = "cuda",
    class_select: int = None,
    sample: int = None,
    seed: int = 42,
    pe_type: str = "RWPE",
) -> torch.Tensor:
    if class_select is not None:
        data_list = [
            d for d in data_list if hasattr(d, "y") and int(d.y) == class_select
        ]
    if sample is not None and sample < len(data_list):
        import random

        random.seed(seed)
        data_list = random.sample(data_list, sample)

    corr_tensor_stack = []
    iterator = tqdm(data_list, desc=f"Faithfulness: {method.__name__}")
    for data in iterator:
        att_matrix = get_attention_matrix_GT(
            model=model,
            data=data,
            layer_idx=layer_idx,
            make_symmetric=False,
            device=device,
        ).to(device)
        adj_matrix = edge_index_to_dense_adj(data.edge_index).to(device)
        edge_attr_from_method = method(
            adj_matrix=adj_matrix, att_matrix=att_matrix, Kmax=Kmax
        )
        edge_attr_from_method = edge_attr_from_method[
            data.edge_index[1], data.edge_index[0]
        ]

        corr_tensor = perform_faithfulness_test_single_data(
            data=data,
            model=model,
            edge_attr_from_method=edge_attr_from_method,
            alpha_list=alpha_list,
            k=k,
            target_edge_mask=None,
            device=device,
            pe_type=pe_type,
        )
        corr_tensor_stack.append(corr_tensor.unsqueeze(-1))

    if len(corr_tensor_stack) == 0:
        raise ValueError(
            "No graphs selected for evaluation (check class_select/sample arguments)."
        )

    corr_tensor_stack = torch.cat(corr_tensor_stack, dim=-1)
    avg_corr = torch.nanmean(corr_tensor_stack, dim=-1)
    return avg_corr


###############################
# Main Execution
###############################
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Edge Attribution Faithfulness"
    )
    parser.add_argument(
        "--Kmax",
        type=int,
        default=2,
        help="Maximum power (path length) for attribution method",
    )
    parser.add_argument(
        "--alpha_list",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated list of alpha values for interpolation",
    )
    parser.add_argument(
        "--explanation_method",
        type=str,
        choices=["EdAM_s", "EdAM_i", "direct_interpretation"],
        default="direct_interpretation",
        help="Edge attribution method to use",
    )
    parser.add_argument(
        "--adjust_popularity",
        action="store_true",
        help="Enable edge popularity adjustment in attribution",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="Epsilon for normalization in attribution",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Display progress bar on console"
    )
    parser.add_argument(
        "--positional_encoding",
        type=str,
        default="RWPE",
        choices=["RWPE", "LapPE"],
        help="Type of positional encoding to use",
    )
    parser.add_argument(
        "--pe_dim", type=int, default=16, help="Positional encoding output dimension"
    )
    parser.add_argument(
        "--pe_enc_dim",
        type=int,
        default=16,
        help="Input PE dimension (RWPE walk length or LapPE k)",
    )
    parser.add_argument("--channels", type=int, default=32, help="Total channels")
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=1, help="Number of attention heads"
    )
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument(
        "--norm_type", type=str, default="batch", help="Normalization type"
    )
    parser.add_argument(
        "--use_mlp_classifier", type=bool, default=False, help="Use MLP classifier"
    )
    parser.add_argument(
        "--mlp_layers", type=int, default=3, help="MLP classifier layers"
    )
    parser.add_argument(
        "--mlp_hidden_dim", type=int, default=None, help="MLP classifier hidden dim"
    )
    parser.add_argument(
        "--use_transformer_mlp", type=bool, default=False, help="Use transformer MLP"
    )
    parser.add_argument(
        "--transformer_mlp_layers", type=int, default=2, help="Transformer MLP layers"
    )
    parser.add_argument(
        "--transformer_mlp_hidden_dim",
        type=int,
        default=16,
        help="Transformer MLP hidden dim",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        choices=["mutagenicity"],
        default="mutagenicity",
        required=True,
        help="Dataset filename (e.g., mutagenicity_data.pt)",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        required=True,
        help="Directory where the trained model is saved",
    )
    parser.add_argument(
        "--class_select",
        type=int,
        default=None,
        help="If set, only evaluate graphs with this class label (y).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, randomly sample this many graphs (after class filtering if class_select is set).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--edge_dim", type=int, default=16, help="Edge embedding dimension"
    )
    parser.add_argument(
        "--input_edge_dim", type=int, default=3, help="Input edge feature dimension"
    )
    parser.add_argument(
        "--max_in_degree",
        type=int,
        default=5,
        help="Max in-degree for centrality encoding",
    )
    parser.add_argument(
        "--max_out_degree",
        type=int,
        default=5,
        help="Max out-degree for centrality encoding",
    )

    args = parser.parse_args()

    try:
        alpha_list = [float(a.strip()) for a in args.alpha_list.split(",")]
    except Exception as e:
        print("Error parsing alpha_list:", e)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = "/workspace/Data_graphxai"
    dataset_path = os.path.join(data_path, args.data_name + "_data.pt")
    data_list = torch.load(dataset_path, map_location=device)

    for data in data_list:
        if data.edge_attr is None:
            data.edge_attr = torch.ones(data.edge_index.size(1), 1, device=device)
        else:
            data.edge_attr = data.edge_attr.to(device)

    num_data = len(data_list)
    train_ratio = 0.8
    val_ratio = 0.1
    num_train = int(num_data * train_ratio)
    num_val = int(num_data * val_ratio)
    num_test = num_data - num_train - num_val
    split_seed = 42
    train_data, val_data, test_data = torch.utils.data.random_split(
        data_list,
        [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(split_seed),
    )

    data_list = [data for data in train_data]

    # Apply positional encoding to each graph.
    if args.positional_encoding == "RWPE":
        data_list = [
            generate_random_walk_attr(data, walk_length=args.pe_enc_dim)
            for data in data_list
        ]
    elif args.positional_encoding == "LapPE":
        data_list = [
            generate_laplacian_pe(data, k=args.pe_enc_dim) for data in data_list
        ]

    if hasattr(data_list[0], "x"):
        node_feature_dim = data_list[0].x.shape[1]
    else:
        print("No node features found in the data. Exiting.")
        sys.exit(1)

    model_path = os.path.join(args.model_save_dir, f"{args.data_name}_GTbase_best.pt")
    if not os.path.exists(model_path):
        print(f"Model save path '{model_path}' not found. Exiting.")
        sys.exit(1)

    model = GTBase(
        channels=args.channels,
        pe_dim=args.pe_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        node_feature_dim=node_feature_dim,
        num_classes=args.num_classes,
        pe_enc_dim=args.pe_enc_dim,
        norm_type=args.norm_type,
        use_mlp_classifier=args.use_mlp_classifier,
        mlp_layers=args.mlp_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        use_transformer_mlp=args.use_transformer_mlp,
        transformer_mlp_layers=args.transformer_mlp_layers,
        transformer_mlp_hidden_dim=args.transformer_mlp_hidden_dim,
        use_hybrid=False,
        gat_conv_kwargs=None,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    if args.explanation_method == "direct_implementation":
        method_func = direct_implementation
    elif args.explanation_method == "EdAM_s":

        def method_func(adj_matrix, att_matrix, Kmax, **kwargs):
            return EdAM_s_attribution_with_popularity(
                adj_matrix,
                att_matrix,
                Kmax,
                adjust_popularity=args.adjust_popularity,
                epsilon=args.epsilon,
                device=device,
            )

    elif args.explanation_method == "EdAM_i":

        def method_func(adj_matrix, att_matrix, Kmax, **kwargs):
            return EdAM_i_attribution_with_popularity(
                adj_matrix,
                att_matrix,
                Kmax,
                adjust_popularity=args.adjust_popularity,
                epsilon=args.epsilon,
                device=device,
            )

    else:
        raise ValueError(f"Unknown explanation method: {args.explanation_method}")

    corr_tensor = perform_faithfulness_test_whole_dataset(
        data_list=data_list,
        model=model,
        method=method_func,
        Kmax=args.Kmax,
        alpha_list=alpha_list,
        k=args.pe_enc_dim,
        device=device,
        class_select=args.class_select,
        sample=args.sample,
        seed=args.seed,
        pe_type=args.positional_encoding,
    )

    print("Final Correlation Tensor")
    print("           \t Pearson \t Kendall")
    print(f"Prediction \t {corr_tensor[0][0]:.4f} \t {corr_tensor[0][1]:.4f}")


if __name__ == "__main__":
    main()
