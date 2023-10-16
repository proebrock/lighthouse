import copy
import igraph
from trafolib.trafo3d import Trafo3d
import matplotlib.pyplot as plt
import numpy as np



class PoseGraph:

    def __init__(self) -> None:
        self._graph = igraph.Graph(directed=True)



    def is_connected(self):
        return self._graph.is_connected()



    def add_vertex(self, vertex_name: str):
        if self._graph.vcount() > 0:
            if vertex_name in self._graph.vs['name']:
                raise ValueError(f'Vertex with name {vertex_name} already exists.')
        self._graph.add_vertex(vertex_name)



    def add_vertices(self, vertex_names: list [str]):
        for vertex_name in vertex_names:
            self.add_vertex(vertex_name)



    def add_edge(self, from_vertex_name: str, to_vertex_name: str,
                 trafo=Trafo3d()):
        self._graph.add_edge(from_vertex_name, to_vertex_name,
            trafo=trafo, inverse=False)
        self._graph.add_edge(to_vertex_name, from_vertex_name,
            trafo=trafo.inverse(), inverse=True)



    def calculate_trafo_between_vertices(self, from_vertex_name: str, to_vertex_name: str) -> Trafo3d:
        # Find the shortest path between the source and target vertices
        if from_vertex_name == to_vertex_name:
            return Trafo3d()
        vpaths = self._graph.get_all_shortest_paths(from_vertex_name, to=to_vertex_name,
            mode=igraph.OUT)
        if len(vpaths) == 0:
            raise Exception(f'No path found between {from_vertex_name} and {to_vertex_name}.')
        # The method get_all_shortest_paths() does not provide edge-based graphs
        # with a switch like "output='vpath'""; if there are multiple edges between
        # two vertices along one path, we have to manually select and evaluate those;
        # vpaths just contains multiple instances of the same vertex lists
        vpaths_set = set(tuple(v) for v in vpaths) # Remove duplicates
        # Iterating over all paths: average trafos
        all_path_trafos = []
        for vpath in vpaths_set:
            # Iterating over all vertices of a path: concatenate trafos
            path_trafo = Trafo3d()
            for i in range(1, len(vpath)):
                # Iterating over all edges between two vertices: average trafos
                edges = self._graph.es.select(_source=vpath[i-1], _target=vpath[i])
                edge_trafos = [ edge['trafo'] for edge in edges ]
                path_trafo = path_trafo * Trafo3d.average(edge_trafos)
            all_path_trafos.append(path_trafo)
        return Trafo3d.average(all_path_trafos)



    def calculate_trafos_from_source_to_targets(self, source_vertex_name: str, target_vertex_names: list [str]) -> list [Trafo3d]:
        trafos = []
        for target_vertex_name in target_vertex_names:
            trafo = self.calculate_trafo_between_vertices(source_vertex_name, target_vertex_name)
            trafos.append(trafo)
        return trafos



    def plot(self, filter_inverse=True):
        # Copy graph for visualization
        plot_graph = copy.deepcopy(self._graph)
        if filter_inverse:
            # Remove all edges with "inverse" flags
            es = plot_graph.es.select(inverse=True)
            plot_graph.delete_edges(es)
        # Set visual styles
        visual_style = {
            #'vertex_label': [ v.index for v in plot_graph.vs ],
            'vertex_label': plot_graph.vs['name'],
            #'edge_label': [ e.index for e in plot_graph.es ],
            #'edge_label': [ str(t) for t in plot_graph.es['trafo'] ],
        }
        _, ax = plt.subplots()
        igraph.plot(plot_graph, target=ax, **visual_style)
        plt.show()
