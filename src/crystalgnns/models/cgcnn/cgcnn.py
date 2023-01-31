
import tensorflow as tf
from kgcnn.layers.conv.cgcnn_conv import CGCNNLayer
from kgcnn.layers.geom import DisplacementVectorsASU, DisplacementVectorsUnitCell, FracToRealCoordinates, \
    EuclideanNorm, GaussBasisLayer, NodePosition
from kgcnn.layers.pooling import PoolingNodes, PoolingWeightedNodes
from kgcnn.layers.modules import OptionalInputEmbedding, LazySubtract, DenseEmbedding
from kgcnn.layers.mlp import MLP
from kgcnn.utils.models import update_model_kwargs

ks = tf.keras

# Implementation of CGCNN in `tf.keras` from paper:
# Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties
# Tian Xie and Jeffrey C. Grossman
# https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301


model_crystal_default = {
    'name': 'CGCNN',
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "distances", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    'input_embedding': {'node': {'input_dim': 95, 'output_dim': 64}},
    'expand_distance': True,
    'gauss_args': {'bins': 40, 'distance': 5, 'offset': 0.0, 'sigma': 0.4},
    'depth': 3,
    "verbose": 10,
    'conv_layer_args': {
        'units': 64,
        'activation_s': 'softplus',
        'activation_out': 'softplus',
        'batch_normalization': True,
    },
    'node_pooling_args': {'pooling_method': 'mean'},
    "output_embedding": "graph",
    'output_mlp': {'use_bias': [True, False], 'units': [64, 1],
                   'activation': ['softplus', 'linear']},
}


@update_model_kwargs(model_crystal_default)
def make_crystal_model(inputs: list = None,
                       input_embedding: dict = None,
                       conv_layer_args: dict = None,
                       expand_distance: bool = None,
                       depth: int = None,
                       name: str = None,
                       verbose: int = None,
                       gauss_args: dict = None,
                       node_pooling_args: dict = None,
                       output_mlp: dict = None,
                       output_embedding: str = None,
                       ):
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    distance = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    edge_distances = tf.expand_dims(distance, -1)

    if expand_distance:
        edge_distances = GaussBasisLayer(**gauss_args)(edge_distances)

    # embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)

    n = DenseEmbedding(conv_layer_args["units"], activation='linear')(n)

    for _ in range(depth):
        n = CGCNNLayer(**conv_layer_args)([n, edge_distances, edge_index_input])

    out = PoolingNodes(**node_pooling_args)(n)

    out = MLP(**output_mlp)(out)

    # Only graph embedding for CGCNN.
    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `CGCNN`.")

    model = ks.models.Model(
            inputs=[node_input, distance, edge_index_input],
            outputs=out, name=name)
    return model
