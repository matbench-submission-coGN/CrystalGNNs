import tensorflow as tf
from kgcnn.layers.conv.dimenet_conv import DimNetInteractionPPBlock, DimNetOutputBlock, EmbeddingDimeBlock, SphericalBasisLayer
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.geom import EdgeAngle, BesselBasisLayer, EuclideanNorm
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate, LazyAdd
from kgcnn.utils.models import update_model_kwargs
from kgcnn.layers.mlp import MLP

from crystalgnns.kgcnn_layers.multiplicity_readout import MultiplcityReadout

ks = tf.keras

# Implementation of DimeNet++ in `tf.keras` from paper:
# Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules
# Johannes Klicpera, Shankari Giri, Johannes T. Margraf, Stephan Günnemann
# https://arxiv.org/abs/2011.14115
# Original code: https://github.com/gasteigerjo/dimenet

model_default = {
    "name": "DimeNetPP",
    "inputs": [{"shape": [None], "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": [None, 3], "name": "edge_offsets", "dtype": "float32", "ragged": True},
               {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
               {"shape": [None,], "name": "multiplicities", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128,
                                 "embeddings_initializer": {"class_name": "RandomUniform",
                                                            "config": {"minval": -1.7320508075688772,
                                                                       "maxval": 1.7320508075688772}}}},
    "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
    "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
    "cutoff": 5.0, "envelope_exponent": 5,
    "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
    "num_targets": 64, "extensive": True, "output_init": "zeros",
    "activation": "swish", "verbose": 10,
    "output_embedding": "graph",
    "use_output_mlp": True,
    "output_mlp": {"use_bias": [True, False],
                   "units": [64, 12], "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               emb_size: int = None,
               out_emb_size: int = None,
               int_emb_size: int = None,
               basis_emb_size: int = None,
               num_blocks: int = None,
               num_spherical: int = None,
               num_radial: int = None,
               cutoff: float = None,
               envelope_exponent: int = None,
               num_before_skip: int = None,
               num_after_skip: int = None,
               num_dense_output: int = None,
               num_targets: int = None,
               activation: str = None,
               extensive: bool = None,
               output_init: str = None,
               verbose: int = None,
               name: str = None,
               output_embedding: str = None,
               use_output_mlp: bool = None,
               output_mlp: dict = None
               ):
    """Make `DimeNetPP <https://arxiv.org/abs/2011.14115>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DimeNetPP.model_default`.
    .. note::
        DimeNetPP does require a large amount of memory for this implementation, which increase quickly with
        the number of connections in a batch.
    Inputs:
        list: `[node_attributes, node_coordinates, bond_indices, angle_indices]`
            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - node_coordinates (tf.RaggedTensor): Atomic coordinates of shape `(batch, None, 3)`.
            - bond_indices (tf.RaggedTensor): Index list for edges or bonds of shape `(batch, None, 2)`.
            - angle_indices (tf.RaggedTensor): Index list of angles referring to bonds of shape `(batch, None, 2)`.
    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.
    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        emb_size (int): Overall embedding size used for the messages.
        out_emb_size (int): Embedding size for output of :obj:`DimNetOutputBlock`.
        int_emb_size (int): Embedding size used for interaction triplets.
        basis_emb_size (int): Embedding size used inside the basis transformation.
        num_blocks (int): Number of graph embedding blocks or depth of the network.
        num_spherical (int): Number of spherical components in :obj:`SphericalBasisLayer`.
        num_radial (int): Number of radial components in basis layer.
        cutoff (float): Distance cutoff for basis layer.
        envelope_exponent (int): Exponent in envelope function for basis layer.
        num_before_skip (int): Number of residual layers in interaction block before skip connection
        num_after_skip (int): Number of residual layers in interaction block after skip connection
        num_dense_output (int): Number of dense units in output :obj:`DimNetOutputBlock`.
        num_targets (int): Number of targets or output embedding dimension of the model.
        activation (str, dict): Activation to use.
        extensive (bool): Graph output for extensive target to apply sum for pooling or mean otherwise.
        output_init (str, dict): Output initializer for kernel.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final :obj:`MLP`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation. Note that DimeNetPP originally defines the output dimension
            via `num_targets`. But this can be set to `out_emb_size` and the `output_mlp` be used for more
            specific control.
    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    offsets = ks.layers.Input(**inputs[1])
    bond_index_input = ks.layers.Input(**inputs[2])
    angle_index_input = ks.layers.Input(**inputs[3])
    multiplicities_ = ks.layers.Input(**inputs[4])
    multiplicities = tf.cast(tf.expand_dims(multiplicities_, -1), tf.float32)

    # Atom embedding
    # n = generate_node_embedding(node_input, input_node_shape, input_embedding["nodes"])
    if len(inputs[0]["shape"]) == 1:
        n = EmbeddingDimeBlock(**input_embedding["node"])(node_input)
    else:
        n = node_input

    edi = bond_index_input
    adi = angle_index_input

    # Calculate distances
    d = tf.expand_dims(EuclideanNorm()(offsets), -1)
    rbf = BesselBasisLayer(num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)(d)

    # Calculate angles
    a = EdgeAngle()([offsets, adi])
    sbf = SphericalBasisLayer(num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff,
                              envelope_exponent=envelope_exponent)([d, a, adi])

    # Embedding block
    rbf_emb = DenseEmbedding(emb_size, use_bias=True, activation=activation,
                             kernel_initializer="kgcnn>glorot_orthogonal")(rbf)
    n_pairs = GatherNodes()([n, edi])
    x = LazyConcatenate(axis=-1)([n_pairs, rbf_emb])
    x = DenseEmbedding(emb_size, use_bias=True, activation=activation, kernel_initializer="kgcnn>glorot_orthogonal")(x)
    ps = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                           output_kernel_initializer=output_init)([n, x, rbf, edi])

    # Interaction blocks
    add_xp = LazyAdd()
    for i in range(num_blocks):
        x = DimNetInteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip)(
            [x, rbf, sbf, adi])
        p_update = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                                     output_kernel_initializer=output_init)([n, x, rbf, edi])
        ps = add_xp([ps, p_update])

    if extensive:
        out = MultiplcityReadout(pooling_method="sum")([ps, multiplicities])
    else:
        out = MultiplcityReadout(pooling_method="mean")([ps, multiplicities])

    if use_output_mlp:
        out = MLP(**output_mlp)(out)

    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `DimeNetPP`.")

    model = ks.models.Model(inputs=[node_input, offsets, bond_index_input, angle_index_input, multiplicities_],
                            outputs=out)
    return model
