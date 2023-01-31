from pymatgen.core.structure import Structure
from typing import Callable, Optional
from hashlib import md5
from networkx import MultiDiGraph
import crystalgnns.crystal_preprocessing.graph_builder as graph_builder

class CrystalPreprocessor(Callable[[Structure], MultiDiGraph]):
    """Base class for crystal preprocessors.

    Concrete CrystalPreprocessors must be implemented as subclasses.
    """

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Should be implemented in a subclass.

        Args:
            structure (Structure): Crystal for which the graph representation should be calculated.

        Raises:
            NotImplementedError:Should be implemented in a subclass.

        Returns:
            MultiDiGraph: Graph representation of the crystal.
        """
        raise NotImplementedError()

    def get_config(self) -> dict:
        """Returns a dictionary uniquely identifying the CrystalPreprocessor and its configuration.

        Returns:
            dict: A dictionary uniquely identifying the CrystalPreprocessor and its configuration.
        """
        config = vars(self)
        config = {k:v for k,v in config.items() if not k.startswith("_")}
        config['preprocessor'] = self.__class__.__name__
        return config

    def hash(self) -> str:
        """Generates a unique hash for the CrystalPreprocessor and its configuration.

        Returns:
            str: A unique hash for the CrystalPreprocessor and its configuration.
        """
        return md5(str(self.get_config()).encode()).hexdigest()

    def __hash__(self):
        return int(self.hash(), 16)

    def __eq__(self, other):
        return hash(self) == hash(other)

class RadiusAsymmetricUnitCell(CrystalPreprocessor):
    """Preprocessor that builds asymmetric unit graphs with radius-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords', 'multiplicity']
    edge_attributes = ['cell_translation', 'distance', 'symmop', 'offset']
    graph_attributes = ['lattice_matrix', 'spacegroup']

    def __init__(self, radius: float=3.0):
        """Initializes the crystal preprocessor.

        Args:
            radius (float, optional): Cutoff radius for each atom in Angstrom units. Defaults to 3.0.
        """
        self.radius = radius

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=True)
        g = graph_builder.add_radius_bonds(g, radius=self.radius, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_asymmetric_unit_graph(g)
        return g

class KNNAsymmetricUnitCell(CrystalPreprocessor):
    """Preprocessor that builds asymmetric unit graphs with kNN-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords', 'multiplicity']
    edge_attributes = ['cell_translation', 'distance', 'symmop', 'offset']
    graph_attributes = ['lattice_matrix', 'spacegroup']

    def __init__(self, k: int=12, tolerance: Optional[float]=1e-9):
        """Initializes the crystal preprocessor.

        Args:
            k (int, optional): How many nearest neighbours to consider for edge selection. Defaults to 12.
            tolerance (Optional[float], optional): If tolerance is not None,
                edges with distances of the k-th nearest neighbor plus the tolerance value are included in the graph.
                Defaults to 1e-9.
        """
        self.k = k
        self.tolerance = tolerance

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=True)
        g = graph_builder.add_knn_bonds(g, k=self.k, tolerance=self.tolerance, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_asymmetric_unit_graph(g)
        return g

class VoronoiAsymmetricUnitCell(CrystalPreprocessor):
    """Preprocessor that builds asymmetric unit graphs with Voronoi-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords', 'multiplicity']
    edge_attributes = ['cell_translation', 'distance', 'symmop', 'offset', 'voronoi_ridge_area']
    graph_attributes = ['lattice_matrix', 'spacegroup']

    def __init__(self, min_ridge_area: Optional[float]=0.0):
        """Initializes the crystal preprocessor.

        Args:
            min_ridge_area (Optional[float], optional): Threshold value for ridge area between two Voronoi cells.
                If a ridge area between two voronoi cells is smaller than this value the corresponding edge between
                the atoms of the cells is excluded from the graph. Defaults to 0.0.
        """
        self.min_ridge_area = min_ridge_area

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=True)
        g = graph_builder.add_voronoi_bonds(g, min_ridge_area=self.min_ridge_area, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_asymmetric_unit_graph(g)
        return g


class RadiusUnitCell(CrystalPreprocessor):
    """Preprocessor that builds unit cell graphs with radius-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __init__(self, radius: float=3.0):
        """Initializes the crystal preprocessor.

        Args:
            radius (float, optional): Cutoff radius for each atom in Angstrom units. Defaults to 3.0.
        """
        self.radius = radius

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_radius_bonds(g, radius=self.radius, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        return g


class KNNUnitCell(CrystalPreprocessor):
    """Preprocessor that builds unit cell graphs with kNN-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __init__(self, k: int=12, tolerance: Optional[float]=1e-9):
        """Initializes the crystal preprocessor.

        Args:
            k (int, optional): How many nearest neighbours to consider for edge selection. Defaults to 12.
            tolerance (Optional[float], optional): If tolerance is not None,
                edges with distances of the k-th nearest neighbor plus the tolerance value are included in the graph.
                Defaults to 1e-9.
        """
        self.k = k
        self.tolerance = tolerance

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_knn_bonds(g, k=self.k, tolerance=self.tolerance, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        return g


class VoronoiUnitCell(CrystalPreprocessor):
    """Preprocessor that builds unit cell graphs with Voronoi-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset', 'voronoi_ridge_area']
    graph_attributes = ['lattice_matrix']

    def __init__(self, min_ridge_area: Optional[float]=0.0):
        """Initializes the crystal preprocessor.

        Args:
            min_ridge_area (Optional[float], optional): Threshold value for ridge area between two Voronoi cells.
                If a ridge area between two voronoi cells is smaller than this value the corresponding edge between
                the atoms of the cells is excluded from the graph. Defaults to 0.0.
        """
        self.min_ridge_area = min_ridge_area

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_voronoi_bonds(g, min_ridge_area=self.min_ridge_area, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        return g
        

class RadiusSuperCell(CrystalPreprocessor):
    """Preprocessor that builds super cell graphs with radius-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __init__(self, radius: float=3.0, size=[3,3,3]):
        """Initializes the crystal preprocessor.

        Args:
            radius (float, optional): Cutoff radius for each atom in Angstrom units. Defaults to 3.0.
            size (list, optional): How many cells the crystal will get expanded into each dimension.
                Defaults to [3,3,3].
        """
        self.radius = radius
        self.size = size

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_radius_bonds(g, radius=self.radius, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_supercell_graph(g, size=self.size)
        return g

class KNNSuperCell(CrystalPreprocessor):
    """Preprocessor that builds super cell graphs with kNN-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __init__(self, k:int=12, tolerance: Optional[float]=1e-9, size=[3,3,3]):
        """Initializes the crystal preprocessor.

        Args:
            k (int, optional): How many nearest neighbours to consider for edge selection. Defaults to 12.
            tolerance (Optional[float], optional): If tolerance is not None,
                edges with distances of the k-th nearest neighbor plus the tolerance value are included in the graph.
                Defaults to 1e-9.
            size (list, optional): How many cells the crystal will get expanded into each dimension.
                Defaults to [3,3,3].
        """
        self.k = k
        self.tolerance = tolerance
        self.size = size

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_knn_bonds(g, k=self.k, tolerance=self.tolerance, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_supercell_graph(g, size=self.size)
        return g
        
class VoronoiSuperCell(CrystalPreprocessor):
    """Preprocessor that builds super cell graphs with Voronoi-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset', 'voronoi_ridge_area']
    graph_attributes = ['lattice_matrix']

    def __init__(self, min_ridge_area: Optional[float]=0.0, size=[3,3,3]):
        """Initializes the crystal preprocessor.

        Args:
            min_ridge_area (Optional[float], optional): Threshold value for ridge area between two Voronoi cells.
                If a ridge area between two voronoi cells is smaller than this value the corresponding edge between
                the atoms of the cells is excluded from the graph. Defaults to 0.0.
            size (list, optional): How many cells the crystal will get expanded into each dimension.
                Defaults to [3,3,3].
        """
        self.size = size
        self.min_ridge_area = min_ridge_area

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_voronoi_bonds(g, min_ridge_area=self.min_ridge_area, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_supercell_graph(g, size=self.size)
        return g


class RadiusNonPeriodicUnitCell(CrystalPreprocessor):
    """Preprocessor that builds non-periodic graphs with radius-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __init__(self, radius: float=3.0):
        """Initializes the crystal preprocessor.

        Args:
            radius (float, optional): Cutoff radius for each atom in Angstrom units. Defaults to 3.0.
        """
        self.radius = radius

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_radius_bonds(g, radius=self.radius, inplace=True)
        g = graph_builder.to_non_periodic_unit_cell(g, add_reverse_edges=True, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        return g


class KNNNonPeriodicUnitCell(CrystalPreprocessor):
    """Preprocessor that builds non-periodic graphs with kNN-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __init__(self, k=12, tolerance=1e-9):
        """Initializes the crystal preprocessor.

        Args:
            k (int, optional): How many nearest neighbours to consider for edge selection. Defaults to 12.
            tolerance (float, optional): If tolerance is not None,
                edges with distances of the k-th nearest neighbor plus the tolerance value are included in the graph.
                Defaults to 1e-9.
        """
        self.k = k
        self.tolerance = tolerance

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_knn_bonds(g, k=self.k, tolerance=self.tolerance, inplace=True)
        g = graph_builder.to_non_periodic_unit_cell(g, add_reverse_edges=True, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        return g


class VoronoiNonPeriodicUnitCell(CrystalPreprocessor):
    """Preprocessor that builds non-periodic graphs with Voronoi-based edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset', 'voronoi_ridge_area']
    graph_attributes = ['lattice_matrix']

    def __init__(self, min_ridge_area: Optional[float]=0.0):
        """Initializes the crystal preprocessor.

        Args:
            min_ridge_area (Optional[float], optional): Threshold value for ridge area between two Voronoi cells.
                If a ridge area between two voronoi cells is smaller than this value the corresponding edge between
                the atoms of the cells is excluded from the graph. Defaults to 0.0.
        """
        self.min_ridge_area = min_ridge_area

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_voronoi_bonds(g, min_ridge_area=self.min_ridge_area, inplace=True)
        g = graph_builder.to_non_periodic_unit_cell(g, add_reverse_edges=True, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        return g


class UnitCell(CrystalPreprocessor):
    """Preprocessor that builds unit cell graphs without any edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = []
    graph_attributes = ['lattice_matrix']

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        return g

class AsymmetricUnitCell(CrystalPreprocessor):
    """Preprocessor that builds asymmetric unit graphs without any edges for crystals."""

    node_attributes = ['atomic_number', 'frac_coords', 'coords', 'asymmetric_mapping', 'symmop', 'multiplicity']
    edge_attributes = []
    graph_attributes = ['lattice_matrix', 'spacegroup']

    def __call__(self, structure: Structure) -> MultiDiGraph:
        """Builds the crystal graph (networkx.MultiDiGraph) for the pymatgen structure.

        Args:
            structure (Structure): Structure to convert to a crystal graph.

        Returns:
            MultiDiGraph: Crystal graph for the provided crystal.
        """
        if isinstance(structure, MultiDiGraph):
            g = structure
        else:
            g = graph_builder.structure_to_empty_graph(structure, symmetrize=True)
        return g

