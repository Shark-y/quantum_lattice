from dataclasses import asdict
from itertools import product
from math import pi
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rustworkx import PyGraph

from qiskit_nature.second_q.hamiltonians.lattices import *


class KagomeLattice(Lattice):


    def __init__(
        self,
        rows: int,
        cols: int,
        edge_parameter: Union[complex, Tuple[complex, complex, complex]] = 1.0,
        onsite_parameter: complex = 0.0,
        boundary_condition: BoundaryCondition = BoundaryCondition.OPEN,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.size = (rows, cols)
        self.dim = 2
        self.boundary_condition = boundary_condition

        if rows < 2 or cols < 2 or (rows, cols) == (2, 2):
            # If it's True, triangular lattice can't be well defined.
            raise ValueError(
                "Both of `rows` and `cols` must not be (2, 2)"
                "and must be greater than or equal to 2."
            )

        if isinstance(edge_parameter, (int, float, complex)):
            edge_parameter = (edge_parameter, edge_parameter, edge_parameter)
        elif isinstance(edge_parameter, tuple):
            if len(edge_parameter) != 3:
                raise ValueError(
                    f"The length of `edge_parameter` must be 3, not {len(edge_parameter)}."
                )

        self.edge_parameter = edge_parameter
        self.onsite_parameter = onsite_parameter

        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(np.prod(self.size)))

        # add edges excluding the boundary edges
        bulk_edges = self._bulk_edges()
        graph.add_edges_from(bulk_edges)

        # add self-loops
        self_loop_list = self._self_loops()
        graph.add_edges_from(self_loop_list)

        # add edges that cross the boundaries
        boundary_edge_list = self._boundary_edges()
        graph.add_edges_from(boundary_edge_list)

        # a list of edges that depend on the boundary condition
        self.boundary_edges = [(edge[0], edge[1]) for edge in boundary_edge_list]
        super().__init__(graph)
        # default position
        self.pos = self._default_position()

    def draw_without_boundary(
        self,
        *,
        self_loop: bool = False,
        style: Optional[LatticeDrawStyle] = None,
    ):
        graph = self.graph

        if style is None:
            style = LatticeDrawStyle()
        elif not isinstance(style, LatticeDrawStyle):
            style = LatticeDrawStyle(**style)

        if style.pos is None:
            if self.dim == 1:
                style.pos = {i: [i, 0] for i in range(self.size[0])}
            elif self.dim == 2:
                style.pos = {
                    i: [i % self.size[0], i // self.size[0]] for i in range(np.prod(self.size))
                }

        graph.remove_edges_from(self.boundary_edges)

        self._mpl(
            graph=graph,
            self_loop=self_loop,
            **asdict(style),
        )
