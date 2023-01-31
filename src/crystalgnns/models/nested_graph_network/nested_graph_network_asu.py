import tensorflow as tf
import tensorflow.keras as ks
from crystalgnns.kgcnn_layers.graph_network.graph_networks import SequentialGraphNetwork, GraphNetwork,\
    GraphNetworkMultiplicityReadout, CrystalInputBlock, NestedGraphNetwork
from crystalgnns.kgcnn_layers.preprocessing_layers import LineGraphAngleDecoder
from crystalgnns.kgcnn_layers.embedding_layers.atom_embedding import AtomEmbedding
from crystalgnns.kgcnn_layers.embedding_layers.edge_embedding import EdgeEmbedding, SinCosExpansion
from crystalgnns.periodic_table.periodic_table import PeriodicTable
from kgcnn.layers.mlp import MLP

def get_model(depth=5, inner_depth=1, node_size=64, edge_size=64):
    
    periodic_table = PeriodicTable()
    
    distance = ks.Input(shape=(None,), dtype=tf.float32, name='distance', ragged=True)
    offset = ks.Input(shape=(None, 3), dtype=tf.float32, name='offset', ragged=True)
    # voronoi_ridge_area = ks.Input(shape=(None,), dtype=tf.float32, name='voronoi_ridge_area', ragged=True)
    atomic_number = ks.Input(shape=(None,), dtype=tf.int32, name='atomic_number', ragged=True)
    multiplicity_ = ks.Input(shape=(None,), dtype=tf.int32, name='multiplicity', ragged=True)
    multiplicity = tf.cast(tf.expand_dims(multiplicity_, axis=-1), float)
    line_graph_edge_indices = ks.Input(shape=(None, 2), dtype=tf.int32, name='line_graph_edge_indices', ragged=True)
    edge_indices = ks.Input(shape=(None, 2), dtype=tf.int32, name='edge_indices', ragged=True)

    atom_embedding_layer = AtomEmbedding(
        atomic_number_embedding_args={'input_dim': 119, 'output_dim': 64},
        atomic_mass=periodic_table.get_atomic_mass(),
        atomic_radius=periodic_table.get_atomic_radius(),
        electronegativity=periodic_table.get_electronegativity(),
        ionization_energy=periodic_table.get_ionization_energy(),
        oxidation_states=periodic_table.get_oxidation_states())
    edge_embedding_layer = EdgeEmbedding(
        distance_embedding_args={'bins': 20, 'distance': 6.5, 'sigma': 0.4, 'offset': 0.0})
        # voronoi_embedding_args={'bins': 20, 'distance': 20.0, 'sigma': 5.0, 'offset': 0.0})
    
    crystal_input_block = CrystalInputBlock(atom_embedding_layer,
                                            edge_embedding_layer,
                                            atom_mlp=MLP([node_size]),
                                            edge_mlp=MLP([edge_size]))

    angle_decoder = LineGraphAngleDecoder()
    angle_embedding = SinCosExpansion(dim=10, wave_length=3.141592653589793, base=2)
    
    nested_layers = get_nested_processing_layers(outer_depth=depth, inner_depth=inner_depth,
                                                 node_size=node_size, edge_size=edge_size)
    output_block = GraphNetworkMultiplicityReadout(
                       MLP([edge_size] * 3, activation='swish'),
                       MLP([node_size] * 3, activation='swish'),
                       MLP([64,64,32,1], activation=['swish', 'swish', 'swish', 'linear']),
                       aggregate_edges_local='attention', aggregate_edges_global=None, aggregate_nodes='attention',
                       edge_attention_mlp_local=MLP([32,1]), node_attention_mlp=MLP([32,1]))
    
    angles, _, _, _ = angle_decoder([None, offset, None, line_graph_edge_indices])
    angles_features = angle_embedding(angles)
    
    edge_features, node_features, _, _ = crystal_input_block([distance, atomic_number, None, edge_indices])
    node_features = {'features': node_features, 'multiplicity': multiplicity}
    global_features = {'line_graph_edge_indices': line_graph_edge_indices, 'line_graph_edge_features': angles_features}
    x = nested_layers([edge_features, node_features, global_features, edge_indices])
    _, _, out, _ = output_block(x)
    
    return ks.Model(inputs=[distance, offset, atomic_number, multiplicity_, line_graph_edge_indices, edge_indices], outputs=[out])

def get_nested_processing_layers(outer_depth=4, inner_depth=2, node_size=64, edge_size=64):
    outer_blocks = []
    for i in range(outer_depth):
        inner_blocks = []
        for j in range(inner_depth):
            block = GraphNetwork(MLP([edge_size] * 3, activation='swish'),
                             MLP([edge_size] * 3, activation='swish'),
                             None,
                             aggregate_edges_local='attention',
                             edge_attention_mlp_local=MLP([32,1],
                             activation='swish'))
            inner_blocks.append(block)
        inner_blocks_layer = SequentialGraphNetwork(inner_blocks, update_edges=False)
        outer_block = NestedGraphNetwork(MLP([edge_size] * 3, activation='swish'),
                             MLP([node_size] * 3, activation='swish'),
                             None,
                             inner_blocks_layer,
                             aggregate_edges_local='attention',
                             edge_attention_mlp_local=MLP([32,1],
                             activation='swish'))
        outer_blocks.append(outer_block)
    return SequentialGraphNetwork(outer_blocks, update_edges=False)


def get_shared_nested_processing_layers(outer_depth=4, inner_depth=2, node_size=64, edge_size=64):
    outer_blocks = []
    inner_blocks = []
    for j in range(inner_depth):
        block = GraphNetwork(MLP([edge_size] * 3, activation='swish'),
                         MLP([edge_size] * 3, activation='swish'),
                         None,
                         aggregate_edges_local='attention',
                         edge_attention_mlp_local=MLP([32,1],
                         activation='swish'))
        inner_blocks.append(block)
    inner_blocks_layer = SequentialGraphNetwork(inner_blocks, update_edges=False)
    for i in range(outer_depth):
        outer_block = NestedGraphNetwork(MLP([edge_size] * 3, activation='swish'),
                             MLP([node_size] * 3, activation='swish'),
                             None,
                             inner_blocks_layer,
                             aggregate_edges_local='attention',
                             edge_attention_mlp_local=MLP([32,1],
                             activation='swish'))
        outer_blocks.append(outer_block)
    return SequentialGraphNetwork(outer_blocks, update_edges=False)

