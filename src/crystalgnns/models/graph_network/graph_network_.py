import tensorflow as tf
import tensorflow.keras as ks
from crystalgnns.kgcnn_layers.graph_network.graph_networks import NestedGraphNetwork, SequentialGraphNetwork, GraphNetwork,\
    GraphNetworkMultiplicityReadout, CrystalInputBlock
from crystalgnns.kgcnn_layers.embedding_layers.atom_embedding import AtomEmbedding
from crystalgnns.kgcnn_layers.embedding_layers.edge_embedding import EdgeEmbedding
from crystalgnns.periodic_table.periodic_table import PeriodicTable
from kgcnn.layers.mlp import MLP

def get_gn_block(edge_mlp = {'units': [64,64], 'activation': 'swish'},
                node_mlp = {'units': [64,64], 'activation': 'swish'},
                global_mlp = {'units': [64,32,1], 'activation': ['swish', 'swish', 'linear']},
                aggregate_edges_local="sum", aggregate_edges_global="sum", aggregate_nodes="sum",
                edge_attention_mlp_local = {'units': [1], 'activation': 'linear'},
                edge_attention_mlp_global = {'units': [1], 'activation': 'linear'},
                node_attention_mlp = {'units': [1], 'activation': 'linear'},
                edge_gate=None, node_gate=None, global_gate=None,
                residual_node_update=True, residual_edge_update=False, residual_global_update=False,
                update_edges_input=[True, True, True, False], # [edges, nodes_in, nodes_out, globals_]
                update_nodes_input=[True, False, False], # [aggregated_edges, nodes, globals_]
                update_global_input=[False, True, False], # [aggregated_edges, aggregated_nodes, globals_]
                multiplicity_readout=False):
    edge_mlp = MLP(**edge_mlp) if edge_mlp is not None else None
    node_mlp = MLP(**node_mlp) if node_mlp is not None else None
    global_mlp = MLP(**global_mlp) if global_mlp is not None else None
    edge_attention_mlp_local = MLP(**edge_attention_mlp_local) if edge_attention_mlp_local is not None else None
    edge_attention_mlp_global = MLP(**edge_attention_mlp_global) if edge_attention_mlp_global is not None else None
    node_attention_mlp = MLP(**node_attention_mlp) if node_attention_mlp is not None else None
    if multiplicity_readout:
        block = GraphNetworkMultiplicityReadout(edge_mlp, node_mlp, global_mlp,
                             aggregate_edges_local=aggregate_edges_local,
                             aggregate_edges_global=aggregate_edges_global,
                             aggregate_nodes=aggregate_nodes,
                             edge_attention_mlp_local=edge_attention_mlp_local,
                             edge_attention_mlp_global=edge_attention_mlp_global,
                             node_attention_mlp=node_attention_mlp,
                             edge_gate=edge_gate, node_gate=node_gate, global_gate=global_gate,
                             residual_edge_update=residual_edge_update,
                             residual_node_update=residual_node_update,
                             residual_global_update=residual_global_update,
                             update_edges_input=update_edges_input,
                             update_nodes_input=update_nodes_input,
                             update_global_input=update_global_input)
    else:
        block = GraphNetwork(edge_mlp, node_mlp, global_mlp,
                             aggregate_edges_local=aggregate_edges_local,
                             aggregate_edges_global=aggregate_edges_global,
                             aggregate_nodes=aggregate_nodes,
                             edge_attention_mlp_local=edge_attention_mlp_local,
                             edge_attention_mlp_global=edge_attention_mlp_global,
                             node_attention_mlp=node_attention_mlp,
                             edge_gate=edge_gate, node_gate=node_gate, global_gate=global_gate,
                             residual_edge_update=residual_edge_update,
                             residual_node_update=residual_node_update,
                             residual_global_update=residual_global_update,
                             update_edges_input=update_edges_input,
                             update_nodes_input=update_nodes_input,
                             update_global_input=update_global_input)
    return block

def get_nested_gn_block(edge_mlp = {'units': [64,64], 'activation': 'swish'},
                node_mlp = {'units': [64,64], 'activation': 'swish'},
                global_mlp = {'units': [64,32,1], 'activation': ['swish', 'swish', 'linear']},
                nested_blocks_params = {
                    'edge_mlp': {
                        'units': [64]*2,
                        'activation': ['swish']*(2-1)+['linear']},
                    'node_mlp': {
                        'units': [64]*2,
                        'activation': ['swish']*(2-1)+['linear']},
                    'global_mlp': None,
                    'aggregate_edges_local': 'mean',
                    'aggregate_edges_global': None,
                    'aggregate_nodes': None,
                    'edge_attention_mlp_local': {'units': [64, 1], 'activation': ['swish', 'linear']},
                    'edge_attention_mlp_global': None,
                    'node_attention_mlp': {'units': [64, 1], 'activation': ['swish', 'linear']},
                    'edge_gate': None,
                    'node_gate': None,
                    'global_gate': None,
                    'residual_node_update': True,
                    'residual_edge_update': False,
                    'residual_global_update': False,
                    'update_edges_input': [False, True, True, False],
                    'update_nodes_input': [True, False, False],
                    'update_global_input': [False, True, False]},
                nested_gn_depth=2,
                aggregate_edges_local="sum", aggregate_edges_global="sum", aggregate_nodes="sum",
                edge_attention_mlp_local = {'units': [1], 'activation': 'linear'},
                edge_attention_mlp_global = {'units': [1], 'activation': 'linear'},
                node_attention_mlp = {'units': [1], 'activation': 'linear'},
                edge_gate=None, node_gate=None, global_gate=None,
                residual_node_update=True, residual_edge_update=False, residual_global_update=False,
                update_edges_input=[True, True, True, False], # [edges, nodes_in, nodes_out, globals_]
                update_nodes_input=[True, False, False], # [aggregated_edges, nodes, globals_]
                update_global_input=[False, True, False], # [aggregated_edges, aggregated_nodes, globals_]
                multiplicity_readout=False):
    assert not multiplicity_readout, "Nested Graph Network blocks can not have multiplicity readout."
    edge_mlp = MLP(**edge_mlp) if edge_mlp is not None else None
    node_mlp = MLP(**node_mlp) if node_mlp is not None else None
    global_mlp = MLP(**global_mlp) if global_mlp is not None else None
    edge_attention_mlp_local = MLP(**edge_attention_mlp_local) if edge_attention_mlp_local is not None else None
    edge_attention_mlp_global = MLP(**edge_attention_mlp_global) if edge_attention_mlp_global is not None else None
    node_attention_mlp = MLP(**node_attention_mlp) if node_attention_mlp is not None else None
    nested_blocks = SequentialGraphNetwork(
            [get_gn_block(**nested_blocks_params) for _ in range(nested_gn_depth)],
            update_edges=False, update_global=False)
    block = NestedGraphNetwork(edge_mlp, node_mlp, global_mlp, nested_blocks,
                             aggregate_edges_local=aggregate_edges_local,
                             aggregate_edges_global=aggregate_edges_global,
                             aggregate_nodes=aggregate_nodes,
                             edge_attention_mlp_local=edge_attention_mlp_local,
                             edge_attention_mlp_global=edge_attention_mlp_global,
                             node_attention_mlp=node_attention_mlp,
                             edge_gate=edge_gate, node_gate=node_gate, global_gate=global_gate,
                             residual_edge_update=residual_edge_update,
                             residual_node_update=residual_node_update,
                             residual_global_update=residual_global_update,
                             update_edges_input=update_edges_input,
                             update_nodes_input=update_nodes_input,
                             update_global_input=update_global_input)
    return block

def get_input_block(node_size=64, edge_size=64,
        atomic_mass=True, atomic_radius=True, electronegativity=True, ionization_energy=True, oxidation_states=True,
        distance_embedding_args={'bins': 20, 'distance': 5., 'sigma': 0.4, 'offset': 0.0}, voronoi_embedding_args=None):
    periodic_table = PeriodicTable()

    atom_embedding_layer = AtomEmbedding(
        atomic_number_embedding_args={'input_dim': 119, 'output_dim': node_size},
        atomic_mass= periodic_table.get_atomic_mass() if atomic_mass else None,
        atomic_radius=periodic_table.get_atomic_radius() if atomic_radius else None,
        electronegativity=periodic_table.get_electronegativity() if electronegativity else None,
        ionization_energy=periodic_table.get_ionization_energy() if ionization_energy else None,
        oxidation_states=periodic_table.get_oxidation_states() if oxidation_states else None)
    edge_embedding_layer = EdgeEmbedding(
        distance_embedding_args=distance_embedding_args,
        voronoi_embedding_args=voronoi_embedding_args)
    
    crystal_input_block = CrystalInputBlock(atom_embedding_layer,
                                            edge_embedding_layer,
                                            atom_mlp=MLP([node_size]), edge_mlp=MLP([edge_size]))
    return crystal_input_block


def get_model(depth, input_block_params, intermediate_blocks_params, output_block_params, uses_voronoi_ridge_area=False):

    distance = ks.Input(shape=(None,), dtype=tf.float32, name='distance', ragged=True)
    if uses_voronoi_ridge_area:
        assert input_block_params['voronoi_embedding_args'] is not None
        voronoi_ridge_area = ks.Input(shape=(None,), dtype=tf.float32, name='voronoi_ridge_area', ragged=True)
    atomic_number = ks.Input(shape=(None,), dtype=tf.int32, name='atomic_number', ragged=True)
    edge_indices = ks.Input(shape=(None, 2), dtype=tf.int32, name='edge_indices', ragged=True)

    crystal_input_block = get_input_block(**input_block_params)
    sequential_gn = SequentialGraphNetwork(
        [get_gn_block(**intermediate_blocks_params) for _ in range(depth-1)],
        update_edges=False, update_global=False
    )
    output_block = get_gn_block(**output_block_params)
    
    if uses_voronoi_ridge_area:
        edge_features, node_features, _, _ = crystal_input_block([[distance, voronoi_ridge_area],
                                                              atomic_number,
                                                              None,
                                                              edge_indices])
        x = sequential_gn([edge_features, node_features, None, edge_indices])
        _, _, out, _ = output_block(x)
        return ks.Model(inputs=[distance, voronoi_ridge_area, atomic_number, edge_indices],
                    outputs=[out])
    else:
        edge_features, node_features, _, _ = crystal_input_block([distance,
                                                              atomic_number,
                                                              None,
                                                              edge_indices])
        x = sequential_gn([edge_features, node_features, None, edge_indices])
        _, _, out, _ = output_block(x)
        return ks.Model(inputs=[distance, atomic_number, edge_indices],
                    outputs=[out])

def get_nested_model(depth, input_block_params, intermediate_blocks_params, output_block_params, uses_voronoi_ridge_area=False):

    distance = ks.Input(shape=(None,), dtype=tf.float32, name='distance', ragged=True)
    if uses_voronoi_ridge_area:
        assert input_block_params['voronoi_embedding_args'] is not None
        voronoi_ridge_area = ks.Input(shape=(None,), dtype=tf.float32, name='voronoi_ridge_area', ragged=True)
    atomic_number = ks.Input(shape=(None,), dtype=tf.int32, name='atomic_number', ragged=True)
    edge_indices = ks.Input(shape=(None, 2), dtype=tf.int32, name='edge_indices', ragged=True)
    line_graph_edge_indices = ks.Input(shape=(None, 2), dtype=tf.int32, name='line_graph_edge_indices', ragged=True)

    crystal_input_block = get_input_block(**input_block_params)
    sequential_gn = SequentialGraphNetwork(
        [get_nested_gn_block(**intermediate_blocks_params) for _ in range(depth)],
        update_edges=False, update_global=False
    )
    output_block = get_gn_block(**output_block_params)
    
    if uses_voronoi_ridge_area:
        edge_features, node_features, _, _ = crystal_input_block([[distance, voronoi_ridge_area],
                                                              atomic_number,
                                                              {'line_graph_edge_indices': line_graph_edge_indices},
                                                              edge_indices])
        x = sequential_gn([edge_features, node_features, {'line_graph_edge_indices': line_graph_edge_indices}, edge_indices])
        _, _, out, _ = output_block(x)
        out = out['features']
        return ks.Model(inputs=[distance, voronoi_ridge_area, atomic_number, edge_indices, line_graph_edge_indices],
                    outputs=[out])
    else:
        edge_features, node_features, _, _ = crystal_input_block([distance,
                                                              atomic_number,
                                                              {'line_graph_edge_indices': line_graph_edge_indices},
                                                              edge_indices])
        x = sequential_gn([edge_features, node_features, {'line_graph_edge_indices': line_graph_edge_indices}, edge_indices])
        _, _, out, _ = output_block(x)
        out = out['features']
        return ks.Model(inputs=[distance, atomic_number, edge_indices, line_graph_edge_indices],
                    outputs=[out])
