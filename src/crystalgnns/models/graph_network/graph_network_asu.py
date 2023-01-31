import tensorflow as tf
import tensorflow.keras as ks
from crystalgnns.kgcnn_layers.graph_network.graph_networks import SequentialGraphNetwork, GraphNetwork,\
    GraphNetworkMultiplicityReadout, CrystalInputBlock
from crystalgnns.kgcnn_layers.embedding_layers.atom_embedding import AtomEmbedding
from crystalgnns.kgcnn_layers.embedding_layers.edge_embedding import EdgeEmbedding
from crystalgnns.periodic_table.periodic_table import PeriodicTable
from kgcnn.layers.mlp import MLP

def get_model(depth=5, node_size=64, edge_size=64):
    
    periodic_table = PeriodicTable()
    
    distance = ks.Input(shape=(None,), dtype=tf.float32, name='distance', ragged=True)
    offset = ks.Input(shape=(None, 3), dtype=tf.float32, name='offset', ragged=True)
    # voronoi_ridge_area = ks.Input(shape=(None,), dtype=tf.float32, name='voronoi_ridge_area', ragged=True)
    atomic_number = ks.Input(shape=(None,), dtype=tf.int32, name='atomic_number', ragged=True)
    multiplicity_ = ks.Input(shape=(None,), dtype=tf.int32, name='multiplicity', ragged=True)
    multiplicity = tf.cast(tf.expand_dims(multiplicity_, axis=-1), float)
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
                                            atom_mlp=MLP([node_size]), edge_mlp=MLP([edge_size]))
    blocks = []
    for i in range(depth):
        block = GraphNetwork(MLP([edge_size] * 3, activation='swish'),
                             MLP([node_size] * 3, activation='swish'),
                             None,
                             aggregate_edges_local='attention',
                             edge_attention_mlp_local=MLP([32,1],
                             activation='swish'))
        blocks.append(block)
    sequential_gn = SequentialGraphNetwork(blocks, update_edges=False, update_global=False)
    output_block = GraphNetworkMultiplicityReadout(
                       MLP([edge_size] * 3, activation='swish'),
                       MLP([node_size] * 3, activation='swish'),
                       MLP([64,64,32,1], activation=['swish', 'swish', 'swish', 'linear']),
                       aggregate_edges_local='attention', aggregate_edges_global=None, aggregate_nodes='attention',
                       edge_attention_mlp_local=MLP([32,1]), node_attention_mlp=MLP([32,1]))
    
    edge_features, node_features, _, _ = crystal_input_block([distance,
                                                              atomic_number,
                                                              None,
                                                              edge_indices])
    node_features = {'features': node_features, 'multiplicity': multiplicity}
    x = sequential_gn([edge_features, node_features, None, edge_indices])
    _, _, out, _ = output_block(x)
    
    return ks.Model(inputs=[distance, offset,
                            atomic_number, multiplicity_, edge_indices],
                    outputs=[out])
