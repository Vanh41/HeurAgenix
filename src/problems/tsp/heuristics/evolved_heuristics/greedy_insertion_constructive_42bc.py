from src.problems.tsp.components import Solution, AppendOperator, InsertOperator
import numpy as np
from typing import Union, Dict, Any

# Define the type for the target operator. This heuristic returns AppendOperator or InsertOperator.
TargetOperatorType = Union[AppendOperator, InsertOperator]

def greedy_insertion_constructive_42bc(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[Union[TargetOperatorType, None], Dict[str, Any]]:
    """A constructive heuristic that builds a TSP tour by iteratively inserting the best unvisited node
    into the best possible position within the current partial tour, minimizing the increase in total cost.

    This algorithm can handle an empty initial solution, gradually building it up.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - node_num (int): The total number of nodes in the problem.
            - distance_matrix (numpy.ndarray): A 2D array representing the distances between nodes.
            - current_solution (Solution): The current solution instance, which can be an empty tour or a partial tour.
            - unvisited_nodes (list[int]): A list of node IDs that have not yet been included in the current_solution.

        algorithm_data (dict): The algorithm dictionary for current algorithm only. This dictionary is used to
            store state specific to this heuristic across multiple calls.
            - gi_start_node (int, optional): Stores the chosen starting node for the construction phase.
              This ensures consistency if the `start_node` is not explicitly provided in `kwargs`
              on subsequent calls when the solution is initially empty.

        kwargs: Additional hyper-parameters for the heuristic:
            - start_node (int, optional): The ID of the node to start the tour with if `current_solution` is empty.
              If not provided, the heuristic will default to the first node in `unvisited_nodes`.

    Returns:
        tuple[Union[AppendOperator, InsertOperator, None], dict]:
        - An instance of `AppendOperator` if the solution is empty and a starting node is chosen.
        - An instance of `InsertOperator` if the solution is partial and an unvisited node is inserted.
        - `None` if no further nodes can be inserted (e.g., all nodes are visited or no valid start node can be selected).
        - An updated `algorithm_data` dictionary, potentially containing the `gi_start_node` if it was set
          during this call.
    """
    current_solution_tour = problem_state['current_solution'].tour
    distance_matrix = problem_state['distance_matrix']
    unvisited_nodes = problem_state['unvisited_nodes']
    
    # Initialize algorithm_data for the start node if it doesn't exist.
    # This allows the heuristic to remember the initial start node if not explicitly given in kwargs.
    if 'gi_start_node' not in algorithm_data:
        algorithm_data['gi_start_node'] = None
    
    # --- Step 1: Handle an empty current solution ---
    # If the current solution is empty, the first step is to pick a starting node.
    if not current_solution_tour:
        # Determine the start node from kwargs, or algorithm_data, or default to the first unvisited node.
        start_node_candidate = kwargs.get('start_node', algorithm_data['gi_start_node'])

        if start_node_candidate is None: # If no start node specified in kwargs or previously stored
            if unvisited_nodes:
                # Default to the first available unvisited node
                chosen_start_node = unvisited_nodes[0]
            else:
                # No unvisited nodes to start a tour with. This should not happen if node_num > 0.
                return None, {} # No operator can be applied
        else:
            # Check if the candidate start node is actually unvisited
            if start_node_candidate in unvisited_nodes:
                chosen_start_node = start_node_candidate
            else:
                # If the specified start_node is already visited or invalid, fallback to default
                if unvisited_nodes:
                    chosen_start_node = unvisited_nodes[0]
                else:
                    return None, {} # No valid start node
        
        # Store the chosen start node in algorithm_data for consistency in future calls if needed
        # (e.g., if a hyper-heuristic iterates on this constructive method).
        updated_algorithm_data = algorithm_data.copy()
        updated_algorithm_data['gi_start_node'] = chosen_start_node
        
        # Return an AppendOperator to add the first node to the empty tour.
        return AppendOperator(node=chosen_start_node), updated_algorithm_data

    # --- Step 2: Handle non-empty current solution by finding best insertion ---
    # If the solution is not empty, we proceed to insert an unvisited node.
    # First, check if there are any unvisited nodes left to insert.
    if not unvisited_nodes:
        # All nodes have been visited, no more insertions are possible.
        return None, {} # No operator can be applied

    best_node_to_insert = -1
    best_position_to_insert = -1
    min_cost_increase = float('inf')
    
    n_current_tour = len(current_solution_tour)
    
    # Iterate through all unvisited nodes to find the best candidate for insertion.
    for node_to_insert in unvisited_nodes:
        # Iterate through all possible insertion positions in the current partial tour.
        # For a tour of length L, there are L+1 possible insertion points (indices 0 to L).
        for position in range(n_current_tour + 1):
            cost_increase = 0.0
            
            # Calculate the cost increase for inserting `node_to_insert` at `position`.
            if n_current_tour == 0:
                # This case is handled by the initial 'if not current_solution_tour:' block.
                # It should not be reached here, but as a safeguard.
                cost_increase = 0.0 # No cost if tour is empty
            elif n_current_tour == 1:
                # Current tour has one node, e.g., [A].
                # Inserting X at position 0: [X, A]. Cost increase is dist(X, A).
                # Inserting X at position 1: [A, X]. Cost increase is dist(A, X).
                # Both cases result in the same cost increase relative to an empty path.
                # Note: The problem description implies 'total path cost', not 'total tour cost' (which would close the loop).
                # For a path, the cost is sum of edge costs.
                cost_increase = distance_matrix[current_solution_tour[0], node_to_insert]
            else:
                # Current tour has two or more nodes, e.g., [A, B, C].
                
                # Inserting at the beginning of the tour (position 0): [X, A, B, C]
                # Old edge: None. New edge: (X, A).
                if position == 0:
                    node_after_insert = current_solution_tour[0]
                    cost_increase = distance_matrix[node_to_insert, node_after_insert]
                # Inserting at the end of the tour (position n_current_tour): [A, B, C, X]
                # Old edge: None. New edge: (C, X).
                elif position == n_current_tour:
                    node_before_insert = current_solution_tour[n_current_tour - 1]
                    cost_increase = distance_matrix[node_before_insert, node_to_insert]
                # Inserting in the middle of the tour: [..., A, B, ...] -> [..., A, X, B, ...]
                # Old edge: (A, B). New edges: (A, X) and (X, B).
                else:
                    node_before_insert = current_solution_tour[position - 1]
                    node_after_insert = current_solution_tour[position]
                    cost_increase = (distance_matrix[node_before_insert, node_to_insert] + 
                                     distance_matrix[node_to_insert, node_after_insert]) - \
                                    distance_matrix[node_before_insert, node_after_insert]
            
            # Update the best insertion if a lower cost increase is found.
            if cost_increase < min_cost_increase:
                min_cost_increase = cost_increase
                best_node_to_insert = node_to_insert
                best_position_to_insert = position
                
    # --- Step 3: Return the operator for the best insertion ---
    if best_node_to_insert != -1:
        # An optimal insertion was found. Return an InsertOperator.
        return InsertOperator(node=best_node_to_insert, position=best_position_to_insert), {}
    else:
        # This case should ideally not be reached if unvisited_nodes was not empty,
        # as a valid insertion should always be possible. It acts as a safeguard.
        return None, {}