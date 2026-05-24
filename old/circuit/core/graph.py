from .circuit import Node, Element, Circuit, Branch
import numpy as np
from numpy.typing import NDArray
from typing import Literal
from dataclasses import dataclass
from functools import cache


@dataclass
class CircuitGraph:
    nodes_dict: dict[str, Node]
    graph: dict[str, list[str]]
    branches: list[Branch]

    @property
    def branchnames(self):
        return [branch.name for branch in self.branches]

    def sort_tree(self, tree: list[Branch]) -> list[Branch]:
        """sort tree or cotree branches based on the graph branches order"""
        _order = self.branchnames
        return sorted(tree, key=lambda branch: _order.index(branch.name))
    
def circuit_2_graph(circuit:Circuit) -> CircuitGraph:
    """Convert a Circuit object to CircuitGraph object"""
    nodes_dict = { node.name:node for node in circuit.nodes }

    graph = {
        node.name: [
            neighbour.name
            for neighbour in node.neighbours
        ]
        for node in circuit.nodes
    }

    branches = []
    branch_keys = []
    for node in circuit.nodes:
        for branch in node.branches:
            if branch.key not in branch_keys:
                branches.append(branch)
                branch_keys.append(branch.key)

    return CircuitGraph(
        nodes_dict = nodes_dict,
        graph = graph,
        branches = sort_branches(branches)
    )

def sort_branches(branches: list[Branch]) -> list[Branch]:
    """sort branches by the following order:

    - current_source
    - rlc
    - voltage_source

    Args:
        branches (list[Branch]): graph branches, tree or cotree

    Returns:
        sorted branches
    """
    element_type_order = [
        "current_source",
        "rlc",
        "voltage_source"
    ]
    return sorted(branches, key=lambda branch: element_type_order.index(branch.element_type))

def get_cotree(branches: list[Branch], tree: list[Branch]) -> list[Branch]:
    """get co-tree of a tree from a graph branches"""
    tree_branches = [str(branch) for branch in tree]
    return [branch for branch in branches if str(branch) not in tree_branches]

def graph_2_tree(graph:CircuitGraph, root:str, method:Literal["dfs", "bfs"]="dfs") -> tuple[list[Branch], list[Branch]]:
    """find the tree of a circuit graph based on DFS or BFS algorithm

    - Then, find co-tree of the tree

    Args:
        graph (CircuitGraph): circuit graph
        root (str): root node name
        method (Literal["dfs", "bfs"], optional): DFS or BFS. Defaults to "dfs".

    Returns:
        tree, cotree
    """
    visited:set[str] = set() 
    tree:list[Branch] = []

    def dfs(node:Node):
        visited.add(node.name)
        for branch in node.branches:
            next_node = branch.get_the_other_node(node)
            if next_node.name not in visited:
                tree.append(branch)
                dfs(next_node)

    def bfs(node:Node):
        visited.add(node.name)
        queue = [node]
        while queue:
            node = queue.pop(0)
            for branch in node.branches:
                next_node = branch.get_the_other_node(node)
                if next_node.name not in visited:
                    tree.append(branch)
                    visited.add(next_node.name)
                    queue.append(next_node)

    match method:
        case "dfs":
            dfs(graph.nodes_dict[root])
        case "bfs":
            bfs(graph.nodes_dict[root])
        case _:
            ValueError(f"invalid search method")
    
    return graph.sort_tree(tree), graph.sort_tree(get_cotree(graph.branches, tree))

def graph_spanning_tree_iterator(graph:CircuitGraph, root:str):
    num_of_nodes = len(graph.nodes_dict)
    num_of_branches = len(graph.branches)
    found_trees: set[frozenset[int]] = set()

    @cache
    def get_branch_num(branchname: str):
        return graph.branchnames.index(branchname)
    
    def found_tree_2_output(found_tree: frozenset[int]) -> tuple[list[Branch], list[Branch]]:
        found_tree_nums = sorted(list(found_tree))
        cotree_nums = sorted(list(set(range(num_of_branches)) - found_tree))
        tree = np.array(graph.branches)[found_tree_nums].tolist()
        cotree = np.array(graph.branches)[cotree_nums].tolist()
        return tree, cotree

    def dfs(current_node: Node, previous_node: Node|None, visited_nodes: set[str], visited_branches: set[int]):
        _visited_nodes = {*visited_nodes, current_node.name}
        if len(_visited_nodes) == num_of_nodes:
            _found_tree = frozenset(visited_branches)
            if _found_tree not in found_trees:
                found_trees.add(_found_tree)
                yield found_tree_2_output(_found_tree)
        else:
            for branch in current_node.branches:
                if branch.name in visited_branches:
                    continue
                next_node = branch.get_the_other_node(current_node)
                if next_node.name not in visited_nodes:
                    yield from dfs(next_node, current_node, _visited_nodes, {*visited_branches, get_branch_num(branch.name)})
            if previous_node:
                yield from dfs(previous_node, None, _visited_nodes, visited_branches)

    yield from dfs(graph.nodes_dict[root], None, set(), set())


def tree_2_circuit(tree: list[Branch]) -> Circuit:
    """convert a circuit tree to a Circuit object

    - with the copy set of nodes and elements

    Args:
        tree (list[Branch]): a spanning tree of the circuit graph

    Returns:
        Circuit object
    """
    elements = [ ]
    links = []
    for branch in tree:
        #TODO: deep copy based on the json serialization will lose element class information
        elements.append(Element.parse_raw(branch.element.json()))
        links.extend(branch.link_strs)
    circuit = Circuit(
        elements = elements,
        links = links
    )
    circuit.connect()
    return circuit
    

def tree_2_graph(tree: list[Branch]) -> CircuitGraph:
    """convert a circuit tree to a New CircuitGraph object

    - with the copy set of nodes 

    Args:
        tree (list[Branch]): a spanning tree of the circuit graph

    Returns:
        CircuitGraph object
    """
    nodes_dict = {}
    graph = {}

    def update_nodes(node:Node):
        if node.name not in nodes_dict:
            nodes_dict[node.name] = Node.parse_raw(node.json())
            graph[node.name] = []
        return

    def update_graph(node:Node, neighbour:Node):
        if neighbour.name not in graph[node.name]:
            graph[node.name].append(neighbour.name)
        return

    for branch in tree:
        update_nodes(branch.from_node)
        update_nodes(branch.to_node)
        update_graph(branch.from_node, branch.to_node)
        update_graph(branch.to_node, branch.from_node)
    
    return CircuitGraph(
        nodes_dict = nodes_dict,
        graph = graph,
        branches = tree
    )

def get_incidence_matrix(graph:CircuitGraph) -> NDArray[np.integer]:
    """calculate the incidence matrix of a circuit graph

    Incidence Matrix:

    - rows: nodes
    - columns: branches
    - a_ij
        - 1 if branch j is leaving node i
        - -1 if branch j is entering node i

    Args:
        graph (CircuitGraph): circuit Graph object

    Returns:
        incidence matrix
    """
    a = np.zeros((len(graph.nodes_dict), len(graph.branches)), dtype=int)
    _nodes = list(graph.nodes_dict)
    for idx, branch in enumerate(graph.branches):
        _from = _nodes.index(branch.from_node.name)
        _to = _nodes.index(branch.to_node.name)
        a[_from, idx] = 1
        a[_to, idx] = -1
    return a

def get_fundamental_cut_nodes(branch:Branch, tree: list[Branch]) -> tuple[set[str], set[str]]:
    """cut the circuit graph into two groups of nodes,

    - cut only one tree branch at a time

    Args:
        branch (Branch): a tree branch
        tree (list[Branch]): a spanning tree of the circuit graph

    Returns:
        tuple of two nodes group: from_nodes, to_nodes
    """
    _tree = {branch.name:branch for branch in tree}
    from_nodes = set([branch.from_node.name])
    to_nodes = set([branch.to_node.name])
    _tree.pop(branch.name)
    while _tree:
        for _branch in _tree.copy().values():
            _branch_location: Literal["from", "to"]|None = None
            for node in (_branch.from_node, _branch.to_node):
                if node.name in from_nodes:
                    _branch_location = "from"
                elif node.name in to_nodes:
                    _branch_location = "to"
            match _branch_location:
                case "from":
                    from_nodes.add(_branch.from_node.name)
                    from_nodes.add(_branch.to_node.name)
                    _tree.pop(_branch.name)
                case "to":
                    to_nodes.add(_branch.from_node.name)
                    to_nodes.add(_branch.to_node.name)
                    _tree.pop(_branch.name)
                case None:
                    continue

                    
    return from_nodes, to_nodes

CutSet = list[tuple[Branch, Literal[1, -1]]]

def get_cut_matrix(graph:CircuitGraph, tree:list[Branch], cotree:list[Branch]) -> NDArray[np.integer]:
    """caculate Cut Matrix of a circuit graph given a spanning tree and correspond cotree

    Cut Matrix:

    - rows: fundamental cutset
        - Each fundamental cutset consists only one tree branch
    - columns: branches
    - q_ij
        - 1 if banch j in the cutset is in the same direction as the tree branch
        - -1 if banch j in the cutset is in the opposite direction as the tree branch

    Args:
        graph (CircuitGraph): circuit graph
        tree (list[Branch]): a spanning tree of the graph
        cotree (list[Branch]): cotree of the tree

    Returns:
        Cut Matrix
    """

    def get_cut(branch: Branch) -> CutSet:
        from_nodes, to_nodes = get_fundamental_cut_nodes(branch, tree)
        cut:CutSet = [(branch, 1)]
        for _branch in cotree:
            if (
                _branch.from_node.name in from_nodes 
                and _branch.to_node.name in to_nodes
            ):
                cut.append((_branch, 1))
            elif (
                _branch.from_node.name in to_nodes 
                and _branch.to_node.name in from_nodes
            ):
                cut.append((_branch, -1))
        return cut

    q = np.zeros((len(tree), len(graph.branches)), dtype=int)
    for i, branch in enumerate(tree):
        for _branch, sign in get_cut(branch):
            j = graph.branchnames.index(_branch.name)
            q[i, j] = sign
    return q

def get_fundamental_circuit_nodes(branch: Branch, tree:list[Branch]) -> list[str]:
    """find a fundamental circuit through a cotree branch

    - all the other branches in the circuit is from the tree

    Args:
        branch (Branch): a cotree branch
        tree (list[Branch]): a spinning tree of the circuit graph

    Returns:
        circuit nodes from to_node to from_node of the branch
    """
    g = tree_2_graph(tree)
    visited:set[str] = set()

    def dfs(node:Node, path:list[str], target:str):
        visited.add(node.name)
        for neighbour in g.graph[node.name]:
            if neighbour == target:
                path.append(neighbour)
                return path
            if neighbour not in visited:
                try:
                    return dfs(g.nodes_dict[neighbour], path+[neighbour], target)
                except ValueError:
                    pass
        raise ValueError("Can't reach target, dead end!")

    return dfs(branch.to_node, [branch.to_node.name], branch.from_node.name)

def get_circuit_matrix(graph:CircuitGraph, cotree:list[Branch], tree:list[Branch]) -> NDArray[np.integer]:
    """calculate the circuit matrix of a circuit graph given a cotree and a tree

    - rows: fundamental circuit
        - Each fundamental circuit consists only one cotree branch
    - columns: branches
    - b_ij
        - 1 if banch j in the circuit is in the same direction as the cotree branch
        - -1 if banch j in the circuit is in the opposite direction as the cotree branch

    Args:
        graph (CircuitGraph): circuit graph object
        cotree (list[Branch]): cotree of the graph
        tree (list[Branch]): correspond tree of the cotree

    Returns:
        circuit matrix
    """

    def get_circuit(branch: Branch) -> CutSet:
        nodes = get_fundamental_circuit_nodes(branch, tree)
        circuit: CutSet = [(branch, 1)]
        for _branch in tree:
            if _branch.from_node.name in nodes and _branch.to_node.name in nodes:
                orientation = nodes.index(_branch.to_node.name) - nodes.index(_branch.from_node.name)
                match orientation:
                    case 1:
                        circuit.append((_branch, 1))
                    case -1:
                        circuit.append((_branch, -1))
        return circuit

    b = np.zeros((len(cotree), len(graph.branches)), dtype=int)
    for i, branch in enumerate(cotree):
        for _branch, sign in get_circuit(branch):
            j = graph.branchnames.index(_branch.name)
            b[i, j] = sign
    return b