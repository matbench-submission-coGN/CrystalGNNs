import tensorflow.keras as ks
import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.layers.geom import EuclideanNorm
from crystalgnns.kgcnn_layers.graph_network.graph_networks import NestedGraphNetwork, SequentialGraphNetwork, GraphNetwork
from crystalgnns.kgcnn_layers.embedding_layers.atom_embedding import AtomEmbedding
from crystalgnns.kgcnn_layers.embedding_layers.edge_embedding import EdgeEmbedding, SinCosExpansion, GaussBasisExpansion
from crystalgnns.kgcnn_layers.preprocessing_layers import LineGraphAngleDecoder
from crystalgnns.models.graph_network.graph_network_configurator import GraphNetworkConfigurator

def get_model(input_block_cfg, processing_blocks_cfgs, output_block_cfg, multiplicity=False, line_graph=False, voronoi_ridge_area=False):
    
    offset = ks.Input(shape=(None,3), dtype=tf.float32, name='offset', ragged=True)
    atomic_number = ks.Input(shape=(None,), dtype=tf.int32, name='atomic_number', ragged=True)
    edge_indices = ks.Input(shape=(None, 2), dtype=tf.int32, name='edge_indices', ragged=True)
    
    if voronoi_ridge_area:
        inp_voronoi_ridge_area = ks.Input(shape=(None,), dtype=tf.float32, name='voronoi_ridge_area', ragged=True)
    if multiplicity:
        inp_multiplicity = ks.Input(shape=(None,), dtype=tf.int32, name='multiplicity', ragged=True)
        inp_multiplicity_ = tf.cast(inp_multiplicity, tf.float32)
    if line_graph:
        line_graph_edge_indices = ks.Input(shape=(None,2), dtype=tf.int32, name='line_graph_edge_indices', ragged=True)
        line_graph_angle_decoder = LineGraphAngleDecoder()
        angle_embedding_layer = GaussBasisExpansion.from_bounds(16,0,3.2)
        angles,_,_,_ = line_graph_angle_decoder([None, offset, None, line_graph_edge_indices])
        angle_embeddings = angle_embedding_layer(tf.expand_dims(angles,-1))
        
    euclidean_norm = EuclideanNorm()
    distance = euclidean_norm(offset)
    crystal_input_block = GraphNetworkConfigurator.get_input_block(**input_block_cfg)
    sequential_gn = SequentialGraphNetwork(
        [GraphNetworkConfigurator.get_gn_block(**cfg) for cfg in processing_blocks_cfgs]
    )
    output_block = GraphNetworkConfigurator.get_gn_block(**output_block_cfg)
    
    if multiplicity:
        node_input = {'features': atomic_number, 'multiplicity': inp_multiplicity_}
    else:
        node_input = atomic_number
    
    if voronoi_ridge_area:
        edge_input = (distance, inp_voronoi_ridge_area)
    else:
        edge_input = distance
    
    if line_graph:
        global_input = {'line_graph_edge_indices': line_graph_edge_indices, 'line_graph_edges': angle_embeddings}
    else:
        global_input = None
        
    
    edge_features, node_features, _, _ = crystal_input_block([edge_input,
                                                              node_input,
                                                              None,
                                                              edge_indices])
    x = sequential_gn([edge_features, node_features, global_input, edge_indices])
    _, _, out, _ = output_block(x)
    out = output_block.get_features(out)
    
    input_list = [offset, atomic_number, edge_indices]
    if multiplicity:
        input_list = input_list + [inp_multiplicity]
    if voronoi_ridge_area:
        input_list = input_list + [inp_voronoi_ridge_area]
    if line_graph:
        input_list = input_list + [line_graph_edge_indices]
    
    return ks.Model(inputs=input_list, outputs=out)

