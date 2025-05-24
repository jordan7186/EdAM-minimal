import torch
from torch_geometric.data import Data
from model import GraphTransformer as GTBase
from torch_geometric.utils import (
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_torch_csr_tensor,
    to_scipy_sparse_matrix,
    get_laplacian,
)
import numpy as np


def log_message(message: str, log_file, verbose: bool = False) -> None:
    """Write a message to the log file and optionally print it."""
    with open(log_file, "a") as f:
        f.write(message + "\n")
    if verbose:
        print(message)


# Directly from AddRandomWalkPE in torch_geometric.transforms
def generate_random_walk_attr(
    data: Data, walk_length: int = 5, explicit_output: bool = False
) -> Data:
    """Generate random walk based positional encoding attributes."""
    assert data.edge_index is not None
    row, col = data.edge_index
    N = data.x.shape[0]

    if data.edge_weight is None:
        value = torch.ones(data.num_edges, device=row.device)
    else:
        value = data.edge_weight
    value = scatter(value, row, dim_size=N, reduce="sum").clamp(min=1)[row]
    value = 1.0 / value

    if N <= 2_000:  # Dense code path for faster computation:
        adj = torch.zeros((N, N), device=row.device)
        adj[row, col] = value
        loop_index = torch.arange(N, device=row.device)
    else:
        adj = to_torch_csr_tensor(data.edge_index, value, size=data.size())

    def get_pe(out: torch.Tensor) -> torch.Tensor:
        if is_torch_sparse_tensor(out):
            return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
        return out[loop_index, loop_index]

    out = adj
    pe_list = [get_pe(out)]
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(get_pe(out))

    pe = torch.stack(pe_list, dim=-1)
    if explicit_output:
        return pe
    else:
        data.pe = pe
        return data


# Directly from AddLaplacianPE in torch_geometric.transforms
def generate_laplacian_pe(
    data: Data, k: int = 5, is_undirected: bool = True, explicit_output: bool = False
) -> Data:
    """Generate Laplacian positional encoding attributes, padded to k dimensions if needed."""
    assert data.edge_index is not None
    num_nodes = data.num_nodes
    assert num_nodes is not None
    SPARSE_THRESHOLD = 100

    edge_index, edge_weight = get_laplacian(
        data.edge_index,
        data.edge_weight,
        normalization="sym",
        num_nodes=num_nodes,
    )

    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    if num_nodes < SPARSE_THRESHOLD:
        from numpy.linalg import eig, eigh

        eig_fn = eig if not is_undirected else eigh

        eig_vals, eig_vecs = eig_fn(L.todense())
    else:
        from scipy.sparse.linalg import eigs, eigsh

        eig_fn = eigs if not is_undirected else eigsh

        eig_vals, eig_vecs = eig_fn(  # type: ignore
            L,
            k=min(k + 1, num_nodes),
            which="SR" if not is_undirected else "SA",
            return_eigenvectors=True,
        )

    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    actual_k = min(k, eig_vecs.shape[1] - 1)
    pe = torch.from_numpy(eig_vecs[:, 1 : 1 + actual_k])

    # Pad with zeros if actual_k < k
    if actual_k < k:
        pad = torch.zeros((pe.size(0), k - actual_k), dtype=pe.dtype, device=pe.device)
        pe = torch.cat([pe, pad], dim=1)

    if explicit_output:
        return pe
    else:
        data.pe = pe
        return data


@torch.no_grad()
def get_attention_matrix_GT(
    model: GTBase,
    data: Data,
    layer_idx: int,
    make_symmetric: bool = False,
    is_hybrid: bool = False,
    apply_softmax: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Returns the attention matrix of a specific layer of a SimpleGraphTransformerGraphClassification model.

    Args:
        model (GTBase): The model to get the attention matrix from.
        data (Data): The data object to pass through the model.
        layer_idx (int): The index of the layer to get the attention matrix from.
        make_symmetric (bool, optional): Whether to make the attention matrix symmetric. Defaults to True.
        device (str, optional): The device to put the attention matrix on. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The attention matrix of the specified layer.
    """
    # Get the attention weights
    model.eval()
    model.to(device)
    if data.edge_weight is None:
        edge_weight = torch.ones(data.edge_index.size(1), device=device)
    else:
        edge_weight = data.edge_weight.to(device)

    # Determine the correct edge feature dimension for Graphormer
    if hasattr(model, "input_edge_dim"):
        input_edge_dim = model.input_edge_dim
    else:
        input_edge_dim = 1  # fallback

    if data.edge_attr is not None:
        edge_weight = data.edge_attr.to(device)
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(1)
        if edge_weight.size(1) < input_edge_dim:
            # Pad with zeros
            pad = torch.zeros(
                edge_weight.size(0),
                input_edge_dim - edge_weight.size(1),
                device=edge_weight.device,
                dtype=edge_weight.dtype,
            )
            edge_weight = torch.cat([edge_weight, pad], dim=1)
        elif edge_weight.size(1) > input_edge_dim:
            edge_weight = edge_weight[:, :input_edge_dim]
    else:
        # If no edge_attr, create all-ones with correct feature dim
        edge_weight = torch.ones(data.edge_index.size(1), input_edge_dim, device=device)

    att_matrix_dict = model.get_attention(
        x=data.x.to(device),
        pe=data.pe.to(device),
        edge_index=data.edge_index.to(device),
        edge_weight=edge_weight,
        # data=data,
    )
    if is_hybrid:
        att_matrix_gat = att_matrix_dict["gat"][layer_idx]
        att_matrix_gt = att_matrix_dict["transformer"][layer_idx]

        return att_matrix_gat, att_matrix_gt

    else:
        att_matrix = att_matrix_dict["transformer"][layer_idx]
    if apply_softmax:
        att_matrix = torch.softmax(att_matrix, dim=-1)
    if make_symmetric:
        att_matrix = (att_matrix + att_matrix.t()) / 2
    return att_matrix.squeeze()


def get_fresh_adj(
    data: Data, self_loops: bool = True, norm_type: str = "sym", device: str = "cuda"
) -> torch.Tensor:
    """
    Returns the adjacency matrix of a graph data object in the form of a torch.Tensor.

    Args:
        data (Data): The graph data object.
        self_loops (bool, optional): Whether to add self-loops to the adjacency matrix. Defaults to True.
        norm_type (str, optional): The type of normalization to apply to the adjacency matrix. Choose from 'row', 'sym', or 'none'. Defaults to 'sym'.
        device (str, optional): The device to put the adjacency matrix on. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The adjacency matrix of the graph.
    """
    A = torch.zeros((data.x.size(0), data.x.size(0)), device=device)
    A[data.edge_index[1], data.edge_index[0]] = 1

    if self_loops:
        A[torch.arange(data.x.size(0)), torch.arange(data.x.size(0))] = 1

    if norm_type == "row":
        D = A.sum(dim=1)
        D[D == 0] = 1
        D = D.pow(-1)
        D = torch.diag(D)
        A = D @ A
    elif norm_type == "sym":
        D = A.sum(dim=1)
        D[D == 0] = 1
        D = D.pow(-0.5)
        D = torch.diag(D)
        A = D @ A @ D
    elif norm_type == "none":
        pass
    else:
        raise ValueError("Invalid norm type. Choose from 'row', 'sym', or 'none'.")

    return A


def get_fresh_adj_with_requires_grad(
    data: Data, self_loops: bool = True, norm_type: str = "sym", device: str = "cuda"
) -> torch.Tensor:
    """
    Returns the adjacency matrix of a graph data object in the form of a torch.Tensor.
    In this version, it sets requires_grad=True for the adjacency matrix.

    Args:
        data (Data): The graph data object.
        self_loops (bool, optional): Whether to add self-loops to the adjacency matrix. Defaults to True.
        norm_type (str, optional): The type of normalization to apply to the adjacency matrix. Choose from 'row', 'sym', or 'none'. Defaults to 'sym'.
        device (str, optional): The device to put the adjacency matrix on. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The adjacency matrix of the graph.
    """
    A = torch.zeros((data.x.size(0), data.x.size(0)), device=device)
    A[data.edge_index[1], data.edge_index[0]] = 1

    if self_loops:
        A[torch.arange(data.x.size(0)), torch.arange(data.x.size(0))] = 1

    if norm_type == "row":
        D = A.sum(dim=1)
        D[D == 0] = 1
        D = D.pow(-1)
        D = torch.diag(D)
    elif norm_type == "sym":
        D = A.sum(dim=1)
        D[D == 0] = 1
        D = D.pow(-0.5)
        D = torch.diag(D)
    elif norm_type == "none":
        pass
    else:
        raise ValueError("Invalid norm type. Choose from 'row', 'sym', or 'none'.")

    A.requires_grad = True

    if norm_type == "row":
        return D @ A
    elif norm_type == "sym":
        return D @ A @ D
    elif norm_type == "none":
        return A


# ---------------------------
# Evaluation Functions
# ---------------------------
@torch.no_grad()
def evaluate(dataloader, model: GTBase, device: str, loss_fn) -> tuple[float, float]:
    """
    Evaluate model accuracy and loss on the given dataloader.

    Return accuracy and average loss.
    """
    model.eval()
    num_correct = 0
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        batch = batch.to(device)
        outputs = model(batch, batch.batch if hasattr(batch, "batch") else None)
        loss = loss_fn(outputs, batch.y.long())
        total_loss += (
            loss.item() * batch.num_graphs
        )  # Because nn.CrossEntropyLoss returns the mean loss as default
        total_samples += batch.num_graphs

        preds = outputs.argmax(dim=1)
        labels = batch.y

        num_correct += (preds == labels).sum().item()

    accuracy = num_correct / total_samples
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss


# ---------------------------
# Training Functions
# ---------------------------
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch, batch.batch if hasattr(batch, "batch") else None)
        loss = loss_fn(outputs, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(dataloader.dataset)
