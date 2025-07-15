from omegaconf import DictConfig
from metta.mettagrid.curriculum.core import Task

from abc import ABC, abstractmethod
from typing import Optional
import random
import numpy as np


class MettaGridTask:
    def __init__(self, name: str, task_idx: int, env_config: DictConfig, task_set: "TaskSet"):
        self.name = name
        self.task_idx = task_idx
        self.env_config = env_config
        self.task_set = task_set

    def complete(self, score: float):
        self.task_set.complete_task(self.task_idx, score)


class Node(ABC):

    parent: Optional["Node"] = None
    children: list["Node"] = []
    child_index: Optional[int] = None
    curriculum_algorithm: Optional["CurriculumAlgorithm"] = None
    name: Optional[str] = None

    @abstractmethod
    def sample(self) -> MettaGridTask:
        pass

    def set_as_child(self, parent: "Node", index: int, name: Optional[str] = None):
        self.parent = parent
        self.child_index = index
        self.name = name

    def complete_task(self, task_idx: int, score: float):
        """Handle task completion and update weights if algorithm is set."""
        if self.curriculum_algorithm:
            self.curriculum_algorithm.update_weights(
                self, task_idx, score
            )
        if self.parent:
            assert self.child_index is not None, "Child index must be set when completing task"
            self.parent.complete_task(self.child_index, score)


class CurriculumAlgorithm(ABC):
    @abstractmethod
    def update_weights(self, node: Node, task_idx: int, score: float) -> None:
        """Update weights in-place based on task completion."""
        pass

class TaskSet(Node):
    parent: Optional[Node] = None

    def __init__(
        self,
        curriculum_algorithm: CurriculumAlgorithm,
        names: list[str],
        configs: list[DictConfig],
        weights: Optional[np.ndarray] = None,
    ):
        """Raw constructor with list-based inputs for efficiency.

        Args:
            curriculum_algorithm: Algorithm for updating weights based on task performance
            names: List of task names
            configs: List of DictConfig objects (must match names order)
            weights: Optional numpy array of weights (must match names length)
        """
        self.curriculum_algorithm = curriculum_algorithm
        self.names = names
        self.num_tasks = len(names)
        
        if self.num_tasks == 0:
            raise ValueError("TaskSet must have at least one task")
        if len(configs) != self.num_tasks:
            raise ValueError(f"Number of configs ({len(configs)}) must match number of names ({self.num_tasks})")
        
        # Check for duplicate names
        if len(set(names)) != len(names):
            raise ValueError("TaskSet must have unique task names")

        # Initialize weights as numpy array
        if weights is None:
            self.weights = np.ones(self.num_tasks, dtype=np.float32)
        else:
            if len(weights) != self.num_tasks:
                raise ValueError(f"Number of weights ({len(weights)}) must match number of tasks ({self.num_tasks})")
            self.weights = np.array(weights, dtype=np.float32)

        # Create task objects as a list indexed by task_idx
        self.tasks = [
            MettaGridTask(name, i, config, self)
            for i, (name, config) in enumerate(zip(names, configs))
        ]

    def get_weight_by_idx(self, idx: int) -> float:
        """Get weight for a specific task by index."""
        return float(self.weights[idx])

    def set_weight_by_idx(self, idx: int, weight: float):
        """Set weight for a specific task by index."""
        self.weights[idx] = weight

    def union(self, other: "TaskSet") -> "TaskSet":
        """Return a new TaskSet containing all tasks from both sets."""
        # Use dict to handle duplicates
        task_map = {}
        
        # Add all tasks from self
        for i, task in enumerate(self.tasks):
            task_map[task.name] = (task.env_config, float(self.weights[i]))
        
        # Add/update tasks from other
        for i, task in enumerate(other.tasks):
            if task.name in task_map:
                # Average weights for duplicates
                config, weight = task_map[task.name]
                task_map[task.name] = (config, (weight + float(other.weights[i])) / 2)
            else:
                task_map[task.name] = (task.env_config, float(other.weights[i]))
        
        # Convert back to lists
        names = list(task_map.keys())
        configs = [task_map[name][0] for name in names]
        weights = np.array([task_map[name][1] for name in names], dtype=np.float32)
        
        return TaskSet(
            self.curriculum_algorithm,
            names,
            configs,
            weights,
        )

    def intersection(self, other: "TaskSet") -> "TaskSet":
        """Return a new TaskSet containing only tasks present in both sets."""
        # Build lookup for other's tasks
        other_task_map = {task.name: i for i, task in enumerate(other.tasks)}
        
        names = []
        configs = []
        weights_list = []
        
        # Keep only common tasks
        for i, task in enumerate(self.tasks):
            if task.name in other_task_map:
                other_idx = other_task_map[task.name]
                names.append(task.name)
                configs.append(task.env_config)
                # Average weights
                avg_weight = (float(self.weights[i]) + float(other.weights[other_idx])) / 2
                weights_list.append(avg_weight)
        
        if not names:
            # Empty intersection
            return TaskSet(self.curriculum_algorithm, [], [], np.array([], dtype=np.float32))
        
        return TaskSet(
            self.curriculum_algorithm,
            names,
            configs,
            np.array(weights_list, dtype=np.float32),
        )

    def difference(self, other: "TaskSet") -> "TaskSet":
        """Return a new TaskSet containing tasks in self but not in other."""
        other_names = set(other.names)
        
        names = []
        configs = []
        weights_list = []
        
        for i, task in enumerate(self.tasks):
            if task.name not in other_names:
                names.append(task.name)
                configs.append(task.env_config)
                weights_list.append(float(self.weights[i]))
        
        if not names:
            # Empty difference
            return TaskSet(self.curriculum_algorithm, [], [], np.array([], dtype=np.float32))
        
        return TaskSet(
            self.curriculum_algorithm,
            names,
            configs,
            np.array(weights_list, dtype=np.float32),
        )

    def sample(self) -> MettaGridTask:
        """Sample a task based on current weights."""
        # Normalize weights to probabilities
        probabilities = self.weights / self.weights.sum()

        # Sample task index
        task_idx = np.random.choice(self.num_tasks, p=probabilities)

        return self.tasks[task_idx]

    def __len__(self) -> int:
        return self.num_tasks

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __iter__(self):
        return iter(self.names)


class TaskSetGraph(Node):

    def __init__(self, children: list[Node], names: Optional[list[str]] = None, weights: Optional[np.ndarray] = None):
        self.children = children
        self.num_children = len(children)
        self.names = names

        if self.num_children == 0:
            raise ValueError("TaskSetGraph must have at least one child")

        # Initialize child weights as array
        if weights is None:
            self.child_weights = np.ones(self.num_children, dtype=np.float32)
        else:
            if len(weights) != self.num_children:
                raise ValueError(f"Number of weights ({len(weights)}) must match number of children ({self.num_children})")
            self.child_weights = np.array(weights, dtype=np.float32)

        # Set parent references
        for i, child in enumerate(children):
            child.set_as_child(self, i, names[i] if names is not None else None)

    def get_child_weight(self, index: int) -> float:
        """Get weight of a specific child by index."""
        return float(self.child_weights[index])

    def set_child_weight(self, index: int, weight: float):
        """Set weight of a specific child by index."""
        self.child_weights[index] = weight

    def sample(self) -> MettaGridTask:
        """Sample a child based on their weights, then sample a task from it."""
        # Normalize weights to probabilities
        probabilities = self.child_weights / self.child_weights.sum()

        # Sample a child
        child_idx = np.random.choice(self.num_children, p=probabilities)
        selected_child = self.children[child_idx]

        # Sample from the selected child
        return selected_child.sample()


def create_task_set(
    curriculum_algorithm: CurriculumAlgorithm,
    task_configs: dict[str, DictConfig],
    task_weights: Optional[dict[str, float]] = None,
) -> TaskSet:
    """Helper function to create TaskSet from dictionary-based inputs.
    
    Args:
        curriculum_algorithm: Algorithm for updating weights based on task performance
        task_configs: Dict mapping task names to DictConfig objects
        task_weights: Optional dict mapping task names to weights
    
    Returns:
        TaskSet initialized with the given configuration
    """
    names = list(task_configs.keys())
    configs = [task_configs[name] for name in names]
    
    if task_weights is None:
        weights = None
    else:
        weights = np.array([task_weights.get(name, 1.0) for name in names], dtype=np.float32)
    
    return TaskSet(curriculum_algorithm, names, configs, weights)


class TaskGraph:
    def __init__(self, root: Node):
        """Initialize with a root node (either TaskSet or TaskSetGraph)."""
        self._root = root

    def get_root(self) -> Node:
        return self._root

    def sample_task(self) -> MettaGridTask:
        """Sample a task from the graph starting at the root."""
        return self._root.sample()

    def get_task_by_name(self, name: str) -> Optional[MettaGridTask]:
        """Find a task by name in the graph."""
        return self._find_task_in_node(self._root, name)
    
    def _find_task_in_node(self, node: Node, name: str) -> Optional[MettaGridTask]:
        """Recursively search for a task in the node hierarchy."""
        if isinstance(node, TaskSet):
            for task in node.tasks:
                if task.name == name:
                    return task
        elif isinstance(node, TaskSetGraph):
            for child in node.children:
                task = self._find_task_in_node(child, name)
                if task:
                    return task
        return None
