import numpy as np


def cartesian_product(*arrays) -> np.ndarray:
    """Constructs the cartesian product of numpy arrays.

    Returns:
        np.ndarray: Cartesian product of numpy arrays.
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def get_line_graph(edge_indices: np.ndarray, directions=[1, 0], remove_self_loops=True):
    """Constructs line graph edge indices from graph edge indices.

    Args:
        edge_indices (np.ndarray): Graph edge indices of shape (num_edges, 2).
        directions (list, optional): Specifies the line graph variant.
            [1,0] constructs line graph edge for between two graph edges of form (i, j) -> (j, k).
            [1,1] constructs line graph edge for between two graph edges of form (i, j) -> (k, j).
            Defaults to [1,0].
        remove_self_loops (bool, optional): Whether to remove self loop line graph edges.
            Defaults to True.

    Returns:
        np.ndarray: Line graph edge indices of shape (num_line_graph_edges, 2).
    """
    line_graph_edges = []
    for n in np.unique(edge_indices):
        edge_indices1 = np.argwhere(edge_indices[:, directions[0]] == n)[:, 0]
        if directions[0] == directions[1]:
            edge_indices2 = edge_indices1
        else:
            edge_indices2 = np.argwhere(edge_indices[:, directions[1]] == n)[:, 0]
        line_graph_edges_ = cartesian_product(edge_indices1, edge_indices2)
        line_graph_edges.append(line_graph_edges_)
    if not line_graph_edges:
        return np.zeros((0, 2))
    line_graph_edge_indices = np.concatenate(line_graph_edges, axis=0)
    if remove_self_loops:
        line_graph_edge_indices = line_graph_edge_indices[
            line_graph_edge_indices[:, 0] != line_graph_edge_indices[:, 1]
        ]
    return line_graph_edge_indices


def get_node_triples(edge_indices: np.ndarray, line_graph: np.ndarray):
    """Get node triples from the graph edge indices and line graph edge indices.

    Args:
        edge_indices (np.ndarray): Graph edge indices.
        line_graph (np.ndarray): Line graph edge indices.

    Returns:
        np.ndarray: Node triple indices of each line graph edge of shape (num_line_graph_edges, 3).
    """
    return edge_indices[line_graph.reshape(-1)].reshape(-1, 4)[:, [0, 1, 3]]
