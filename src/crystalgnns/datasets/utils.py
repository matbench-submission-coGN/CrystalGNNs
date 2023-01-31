import h5py
import pandas as pd
from pathlib import Path
from crystalgnns.graph_dataset_tools.graph_tuple import HDFGraphTuple
from itertools import islice
from dataclasses import dataclass
from typing import TypeVar, Generic, Union, Tuple, Any, Optional, Iterable
from pymatgen.core.structure import Structure
from networkx import MultiDiGraph
from crystalgnns.crystal_preprocessing.crystal_preprocessors import CrystalPreprocessor
import json
from multiprocessing import Pool

T = TypeVar("T")


@dataclass
class MetaDataWrapper(Generic[T]):
    """Wrapper around arbitrary python object to add meta data to it.

    In the context of crystal graphs it is used to wrap pymatgen structures with extra
    information to store on the graph level."""

    x: T
    meta_data: Optional[dict] = None


Crystal = Union[
    MetaDataWrapper[MultiDiGraph], MetaDataWrapper[Structure], MultiDiGraph, Structure
]


class PreprocessorWrapper:
    """Callable that modifies the behaviour of CrystalProcessors to include extra global graph attributes.

    Returns:
        MultiDiGraph: Crystal graph with extra global graph attributes.
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def __call__(self, crystal: Crystal):
        if isinstance(crystal, MetaDataWrapper):
            meta_data = crystal.meta_data
            crystal = crystal.x
        else:
            crystal = crystal
            meta_data = None
        graph = self.preprocessor(crystal)
        if meta_data is not None:
            for k, v in meta_data.items():
                if isinstance(v, str):
                    v = v.encode()
                setattr(graph, k, v)
        return graph


def batcher(iterable, batch_size):
    """Creates batches for iterables"""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def create_graph_dataset(
    crystals: Union[Iterable[MetaDataWrapper], Iterable[Structure]],
    preprocessor: CrystalPreprocessor,
    out_file: Path,
    additional_graph_attributes=[],
    processes=None,
    batch_size=1000,
) -> Path:
    """Creates a HDF file containing crystal graphs.

    Args:
        crystals (Union[Iterable[MetaDataWrapper], Iterable[Structure]]): Iterable of pymatgen structures.
        preprocessor (_type_): _description_
        out_file (Path): _description_
        additional_graph_attributes (list, optional): Strings of additional graph attributes to include in the HDF file.
            Defaults to [].
        processes (_type_, optional): _description_. Defaults to None.
        batch_size (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    worker = PreprocessorWrapper(preprocessor)
    with h5py.File(str(out_file), "w") as f:
        with Pool(processes) as p:
            for batch in batcher(crystals, batch_size):
                graphs = p.imap(worker, batch)
                HDFGraphTuple.from_nx_graphs(
                    f,
                    graphs,
                    node_attribute_names=preprocessor.node_attributes,
                    edge_attribute_names=preprocessor.edge_attributes,
                    graph_attribute_names=preprocessor.graph_attributes
                    + additional_graph_attributes,
                )
            f.attrs["preprocessor_config"] = json.dumps(
                preprocessor.get_config(), indent=2
            )
    return out_file
