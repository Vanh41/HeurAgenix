import numpy as np
import networkx as nx
import tsplib95
import gzip
from pathlib import Path
from src.problems.base.env import BaseEnv
from src.problems.tsp.components import Solution


class Env(BaseEnv):
    """TSP env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "tsp")
        self.construction_steps = self.instance_data["node_num"]
        self.key_item = "current_cost"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        return len(set(self.current_solution.tour)) == self.instance_data["node_num"]

    def load_data(self, data_path: str) -> dict:  # Changed return type annotation
        data_path = Path(data_path)
        
        try:
            # Check if file is gzipped
            if data_path.suffix == '.gz':
                with gzip.open(data_path, 'rt', encoding='utf-8') as f:
                    content = f.read()
                    problem = tsplib95.parse(content)
            else:
                problem = tsplib95.load(data_path)
                
            # Get the graph and convert to distance matrix
            graph = problem.get_graph()
            if graph.number_of_nodes() == 0:
                raise ValueError(f"No nodes found in problem file: {data_path}")
                
            distance_matrix = nx.to_numpy_array(graph)
            node_num = len(distance_matrix)
            
            if node_num == 0:
                raise ValueError(f"Distance matrix is empty for file: {data_path}")
                
            print(f"Loaded TSP problem with {node_num} nodes from {data_path}")
            return {"node_num": node_num, "distance_matrix": distance_matrix}
            
        except Exception as e:
            print(f"Error loading TSP file {data_path}: {e}")
            # Try to list a few files in the directory for debugging
            if data_path.is_dir():
                files = list(data_path.glob("*.tsp.gz"))[:3]  # First 3 TSP files
                if files:
                    print(f"Trying to load first available file: {files[0]}")
                    return self.load_data(files[0])
            raise e

    def init_solution(self) -> None:
        return Solution(tour=[])

    def get_key_value(self, solution: Solution=None) -> float:
        """Get the key value of the current solution based on the key item."""
        if solution is None:
            solution = self.current_solution
        current_cost = sum([self.instance_data["distance_matrix"][solution.tour[index]][solution.tour[index + 1]] for index in range(len(solution.tour) - 1)])
        if len(solution.tour) > 0:
            current_cost += self.instance_data["distance_matrix"][solution.tour[-1]][solution.tour[0]]
        return current_cost

    def validation_solution(self, solution: Solution=None) -> bool:
        """
        Check the validation of this solution in following items:
            1. Node Existence: Each node in the solution must exist within the problem instance's range of nodes.
            2. Uniqueness: No node is repeated within the solution path, ensuring that each node is visited at most once.
            3. Connectivity: Each edge (from one node to the next) must be connected, i.e., not marked as infinite distance in the distance matrix.
        """
        node_set = set()
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.tour, list):
            return False
        if solution is not None and solution.tour is not None:
            for index, node in enumerate(solution.tour):
                # Check node existence
                if not (0 <= node < self.instance_data["node_num"]):
                    return False

                # Check uniqueness
                if node in node_set:
                    return False
                node_set.add(node)

                # Check connectivity if not the last node
                if index < len(solution.tour) - 1:
                    next_node = solution.tour[index + 1]
                    if self.instance_data["distance_matrix"][node][next_node] == np.inf:
                        return False
        return True
