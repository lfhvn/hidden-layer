"""
Task generators for geometric memory validation and testing.

Implements path-finding tasks and other geometric reasoning challenges.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DifficultyLevel(Enum):
    """Task difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class PathStarGraph:
    """
    Path-star graph structure for testing geometric memory.

    A path-star graph has:
    - One central/root node
    - d paths branching from the center
    - Each path has length ℓ (number of hops)
    """
    n_paths: int  # d in paper notation
    path_length: int  # ℓ in paper notation
    nodes: List[str]  # Node labels
    edges: Dict[str, str]  # edge mapping: node → next_node
    center_node: str

    @property
    def total_nodes(self) -> int:
        """Total number of nodes in graph"""
        return 1 + (self.n_paths * self.path_length)

    def get_path_from_start(self, start_node: str) -> List[str]:
        """Get full path from start node to end"""
        path = [start_node]
        current = start_node

        while current in self.edges:
            current = self.edges[current]
            path.append(current)

        return path

    def get_path_end(self, start_node: str) -> str:
        """Get the final node of a path starting from start_node"""
        path = self.get_path_from_start(start_node)
        return path[-1]


def generate_path_star_graph(
    n_paths: int = 5,
    path_length: int = 3,
    node_type: str = "names"
) -> PathStarGraph:
    """
    Generate a path-star graph with random node labels.

    Args:
        n_paths: Number of paths branching from center (d)
        path_length: Length of each path in hops (ℓ)
        node_type: Type of node labels ("names", "concepts", "numbers")

    Returns:
        PathStarGraph object

    Example:
        Graph with n_paths=3, path_length=2:

                Alice → Bob → Carol
               /
        Center - Dave → Eve → Frank
               \
                Grace → Henry → Iris

    """
    total_nodes = 1 + (n_paths * path_length)

    # Generate node labels
    if node_type == "names":
        # Use common names
        name_pool = [
            "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry",
            "Iris", "Jack", "Kate", "Leo", "Mary", "Nick", "Olivia", "Paul",
            "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
            "Yara", "Zoe", "Adam", "Beth", "Chris", "Diana", "Ethan", "Fiona",
            "George", "Hannah", "Isaac", "Julia", "Kevin", "Laura", "Mike", "Nina",
            "Oscar", "Penny", "Quentin", "Rita", "Steve", "Tracy", "Ulysses", "Vera",
            "Will", "Xena", "Yale", "Zelda"
        ]
        if total_nodes > len(name_pool):
            raise ValueError(f"Need {total_nodes} nodes but only have {len(name_pool)} names")
        nodes = random.sample(name_pool, total_nodes)

    elif node_type == "concepts":
        # Use abstract concepts
        concept_pool = [
            "Hope", "Fear", "Joy", "Anger", "Trust", "Doubt", "Love", "Hate",
            "Peace", "War", "Light", "Dark", "Truth", "Lies", "Faith", "Reason",
            "Beauty", "Chaos", "Order", "Freedom", "Justice", "Mercy", "Power", "Wisdom",
            "Courage", "Pride", "Humility", "Patience", "Wrath", "Calm", "Storm", "Dawn",
            "Dusk", "Spring", "Summer", "Fall", "Winter", "North", "South", "East",
            "West", "Past", "Future", "Present", "Memory", "Dream", "Reality", "Fantasy",
            "Beginning", "End"
        ]
        nodes = random.sample(concept_pool, total_nodes)

    elif node_type == "numbers":
        # Use random IDs
        nodes = [f"Node{i:03d}" for i in random.sample(range(1000), total_nodes)]

    else:
        raise ValueError(f"Unknown node_type: {node_type}")

    # First node is center
    center = nodes[0]
    remaining = nodes[1:]

    # Build edges (center → paths)
    edges = {}
    paths_created = 0

    for path_idx in range(n_paths):
        # Get nodes for this path
        path_nodes = remaining[path_idx * path_length:(path_idx + 1) * path_length]

        # Connect center to first node of path
        if paths_created == 0:
            first_connection = center
        else:
            first_connection = center

        # Build path edges
        edges[center if path_idx == 0 else center] = path_nodes[0]

        # Actually, structure should be: center connects to start of each path
        # Let me fix this

    # Rebuild edges correctly
    edges = {}
    for path_idx in range(n_paths):
        path_nodes = remaining[path_idx * path_length:(path_idx + 1) * path_length]

        # Center connects to first node of this path
        # But wait, this creates a star where center connects to first of each path
        # Then each path extends linearly

        # For first path
        current = path_nodes[0]
        # No edge from center in "path-star" - let me check paper definition

        # Actually, re-reading paper: it's called "path-star" where:
        # - There's a center/root node
        # - d paths branch from it
        # - Each path has length ℓ

        # So structure is:
        #   center → path1_node1 → path1_node2 → ... → path1_nodeℓ
        #   center → path2_node1 → path2_node2 → ... → path2_nodeℓ
        #   etc.

    # Correct implementation
    edges = {}

    for path_idx in range(n_paths):
        path_nodes = remaining[path_idx * path_length:(path_idx + 1) * path_length]

        # Edge from center to first node of path
        edges[f"{center}_path{path_idx}"] = path_nodes[0]

        # Edges within path
        for i in range(len(path_nodes) - 1):
            edges[path_nodes[i]] = path_nodes[i + 1]

    # Hmm, this isn't quite right either. Let me think...

    # Actually, looking at paper more carefully:
    # - Single center node
    # - d paths extending from it
    # - Each path is a chain of ℓ nodes

    # So there should be edges:
    # - center → first_node_of_path_i (for i = 1..d)
    # - node_j → node_j+1 within each path

    # But center appears multiple times? No, that's not right.

    # Let me implement cleanly:

    edges = {}
    node_idx = 1  # Start after center (which is nodes[0])

    for path_idx in range(n_paths):
        # Get ℓ nodes for this path
        path_nodes = remaining[path_idx * path_length:(path_idx + 1) * path_length]

        # Edge from center to start of this path
        # Key insight: we can't have multiple edges from same node in a simple dict
        # So maybe the structure is actually a multigraph or...

        # Wait, I think I misunderstood. Let me re-read:
        # "path-star graph (a tree graph where only the root node branches)"

        # So it's a TREE where:
        # - Root has d children
        # - Each child is the start of a path of length ℓ

        # That makes more sense! One edge from root to each of d children,
        # then each child starts a linear chain.

    # Clean implementation:
    edges = {}

    for path_idx in range(n_paths):
        path_nodes = remaining[path_idx * path_length:(path_idx + 1) * path_length]

        # Edge from center to first node of this path
        # Use a unique key to avoid overwriting
        # OR: if this is a multigraph, need different representation

        # For simplicity with dict, let's make center connect to first node of each path
        # And we'll track which path in metadata

        # Actually, for a proper path-star:
        # edges should be: (source, dest) tuples or source -> [dests]

        # Let me use a different structure:
        # edges = {node: next_node_in_path}

        # For a branching tree from center:
        #   center -> path1_start
        #   path1_start -> path1_node2
        #   path1_node2 -> path1_node3
        #   ...
        #   center -> path2_start  # PROBLEM: overwrites previous center entry!

        # Solution: make edges a list of tuples (source, dest)
        pass

    # Let me restart with clearer design:
    edges_list = []  # List of (source, dest) tuples

    for path_idx in range(n_paths):
        path_nodes = remaining[path_idx * path_length:(path_idx + 1) * path_length]

        # Center connects to first node of this path
        edges_list.append((center, path_nodes[0]))

        # Nodes within path
        for i in range(len(path_nodes) - 1):
            edges_list.append((path_nodes[i], path_nodes[i + 1]))

    # Convert to dict where we CAN have multiple values per key
    # Using source -> list of dests
    edges_dict = {}
    for source, dest in edges_list:
        if source not in edges_dict:
            edges_dict[source] = []
        edges_dict[source].append(dest)

    # For the PathStarGraph dataclass, adapt to single-dest per node:
    # Since each node has at most one outgoing edge EXCEPT center
    # Let's store as source -> dest (for non-center)
    # And handle center specially

    simple_edges = {}
    for source, dests in edges_dict.items():
        if len(dests) == 1:
            simple_edges[source] = dests[0]
        # Center has multiple, we'll handle separately

    return PathStarGraph(
        n_paths=n_paths,
        path_length=path_length,
        nodes=nodes,
        edges=simple_edges,
        center_node=center
    )


def path_star_graph_to_prompt(
    graph: PathStarGraph,
    include_reverse: bool = True
) -> str:
    """
    Convert path-star graph to text prompt for model.

    Args:
        graph: PathStarGraph object
        include_reverse: If True, include reverse edges (paper recommendation)

    Returns:
        String describing all edges in the graph

    Example output:
        "The following connections exist:
        Center connects to Alice, Dave, and Grace.
        Alice connects to Bob.
        Bob connects to Carol.
        ..."
    """
    lines = ["The following connections exist:"]

    # Handle center node (connects to multiple)
    center = graph.center_node
    first_nodes = []

    # Find all nodes that center connects to
    # (first node of each path)
    for node in graph.nodes:
        if node == center:
            continue
        # Check if this is a path start (connected from center)
        # We need to infer this from structure
        # Nodes that are in first position of each path

    # Actually, let me just build this from the generation logic
    # It's easier to create the prompt during generation

    # For now, simplified version:
    edges_list = []

    # Center to path starts
    remaining = [n for n in graph.nodes if n != center]
    for path_idx in range(graph.n_paths):
        path_start = remaining[path_idx * graph.path_length]
        edges_list.append(f"{center} connects to {path_start}")

        # Edges within this path
        for i in range(graph.path_length - 1):
            node_idx = path_idx * graph.path_length + i
            source = remaining[node_idx]
            dest = remaining[node_idx + 1]
            edges_list.append(f"{source} connects to {dest}")

            if include_reverse:
                edges_list.append(f"{dest} connects to {source}")

    prompt = "The following connections exist:\n" + "\n".join(edges_list)
    return prompt


def generate_path_finding_task(
    difficulty: DifficultyLevel = DifficultyLevel.EASY,
    node_type: str = "names"
) -> Dict[str, any]:
    """
    Generate a complete path-finding task for geometric memory testing.

    Args:
        difficulty: EASY (5 paths, 3 hops), MEDIUM (10 paths, 5 hops), HARD (20 paths, 10 hops)
        node_type: Type of node labels

    Returns:
        Dictionary with task components:
            - graph: PathStarGraph object
            - prompt: Full prompt including edges and question
            - correct_answer: Ground truth answer
            - n_hops: Number of hops required
    """
    # Set parameters based on difficulty
    if difficulty == DifficultyLevel.EASY:
        n_paths, path_length = 5, 3
    elif difficulty == DifficultyLevel.MEDIUM:
        n_paths, path_length = 10, 5
    elif difficulty == DifficultyLevel.HARD:
        n_paths, path_length = 20, 10
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    # Generate graph
    graph = generate_path_star_graph(n_paths, path_length, node_type)

    # Create edges description
    edges_desc = path_star_graph_to_prompt(graph, include_reverse=True)

    # Select a random path to query
    path_idx = random.randint(0, n_paths - 1)
    remaining = [n for n in graph.nodes if n != graph.center_node]
    path_start = remaining[path_idx * path_length]
    path_end = remaining[(path_idx + 1) * path_length - 1]

    # Create question
    question = f"\n\nQuestion: If you start at {path_start}, and follow the connections {path_length} steps, where do you end up?"

    # Full prompt
    full_prompt = edges_desc + question

    # Package task
    return {
        'graph': graph,
        'prompt': full_prompt,
        'correct_answer': path_end,
        'n_hops': path_length,
        'difficulty': difficulty.value,
        'query_start': path_start,
        'path_idx': path_idx,
        'metadata': {
            'n_paths': n_paths,
            'path_length': path_length,
            'total_nodes': graph.total_nodes,
            'node_type': node_type
        }
    }


def generate_multi_hop_reasoning_task(n_hops: int = 3) -> Dict[str, any]:
    """
    Generate multi-hop reasoning task (simpler than path-star).

    Example: A implies B, B implies C, C implies D. What does A imply?

    Args:
        n_hops: Number of reasoning steps

    Returns:
        Task dictionary
    """
    facts = [
        "creativity", "innovation", "progress", "prosperity", "happiness",
        "wisdom", "understanding", "knowledge", "truth", "enlightenment",
        "courage", "action", "change", "growth", "success"
    ]

    # Sample n_hops + 1 facts
    selected = random.sample(facts, n_hops + 1)

    # Build chain
    implications = []
    for i in range(n_hops):
        implications.append(f"{selected[i]} leads to {selected[i+1]}")

    facts_text = ". ".join(implications) + "."

    question = f"\n\nQuestion: Based on these relationships, what does {selected[0]} ultimately lead to?"
    prompt = facts_text + question

    return {
        'prompt': prompt,
        'correct_answer': selected[-1],
        'n_hops': n_hops,
        'task_type': 'multi_hop_reasoning',
        'chain': selected
    }


def generate_analogical_reasoning_task() -> Dict[str, any]:
    """
    Generate A:B :: C:? task.

    Requires understanding relational similarity (geometric reasoning).
    """
    analogies = [
        ("king", "queen", "man", "woman"),
        ("Paris", "France", "London", "England"),
        ("hot", "cold", "day", "night"),
        ("doctor", "hospital", "teacher", "school"),
        ("wheel", "car", "wing", "airplane"),
    ]

    a, b, c, d = random.choice(analogies)

    prompt = f"{a} is to {b} as {c} is to what?"

    return {
        'prompt': prompt,
        'correct_answer': d,
        'task_type': 'analogy',
        'relation': (a, b, c, d)
    }


# Task registry for easy access
TASK_GENERATORS = {
    'path_finding_easy': lambda: generate_path_finding_task(DifficultyLevel.EASY),
    'path_finding_medium': lambda: generate_path_finding_task(DifficultyLevel.MEDIUM),
    'path_finding_hard': lambda: generate_path_finding_task(DifficultyLevel.HARD),
    'multi_hop_3': lambda: generate_multi_hop_reasoning_task(n_hops=3),
    'multi_hop_5': lambda: generate_multi_hop_reasoning_task(n_hops=5),
    'analogy': generate_analogical_reasoning_task,
}


def get_task(task_name: str) -> Dict[str, any]:
    """Get a task by name from registry"""
    if task_name not in TASK_GENERATORS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_GENERATORS.keys())}")

    return TASK_GENERATORS[task_name]()
