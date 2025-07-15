"""
TaskTrees are trees where nodes have two possible types:
* MettaGridTask: A specific environment configuration for MettaGrid
* TaskTree: The root of a task tree whose children are either MettaGridTask or TaskTrees

All leaves are MettaGridTasks; all parents are TaskTrees.

A TaskTree has an associated curriculum algorithm, which is used to update the weights of the children of the TaskTree.
Each TaskTree node can has its own curriculum algorithm.

Scores are propagated up the tree so that each rebalances
the weights for its children based on the score input at the leaf.

Sample queries are propagated down the tree so that each node can sample from its children. All TaskTrees sample
based on a weighted random distribution; curricula are responsible for rebalancing the weights. Weights are
automatically normalized to sum to 1 but should always be non-negative.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf


# TaskTreeNode is the base class for nodes in a task graph
class TaskTreeNode(ABC):
    parent: Optional["TaskTree"] = None
    child_index: Optional[int] = None
    name: Optional[str] = None

    @abstractmethod
    def sample(self) -> "MettaGridTask":
        """Sample a task from the node, either directly or by sampling a from a child."""
        pass

    def set_as_child(self, parent: "TaskTreeNode", index: int, name: Optional[str] = None):
        """Set the parent and child index of the node. Used by TaskTree.__init__ as
        graphs are constructed bottom-up."""
        self.parent = parent
        self.child_index = index
        self.name = name


class CurriculumAlgorithm(ABC):
    def update_weights(self, weights: np.ndarray, child_idx: int, score: float) -> None:
        """Update weights in-place based on task completion."""
        pass

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return statistics for logging purposes. Add prefix to all keys."""
        return dict()


class UniformRandomCurriculum(CurriculumAlgorithm):
    pass


class MettaGridTask(TaskTreeNode):
    def __init__(self, name: str, env_config: DictConfig):
        self.name = name
        self.env_config = env_config

    def sample(self) -> "MettaGridTask":
        return self

    def complete_task(self, score: float):
        if self.parent is not None:
            assert self.child_index is not None, "Child index must be set when completing task with parent"
            self.parent.complete_task(self.child_index, score, self.name)


class TaskTree(TaskTreeNode):
    def __init__(
        self,
        curriculum_algorithm: CurriculumAlgorithm,
        children: list[TaskTreeNode],
        names: Optional[list[str]] = None,
        weights: Optional[np.ndarray] = None,
        env_overrides: Optional[DictConfig] = None,
    ):
        self.children = children
        self.num_children = len(children)
        self.curriculum_algorithm = curriculum_algorithm
        self.completed_tasks = np.zeros(self.num_children, dtype=np.int32)
        self.sampled_tasks = np.zeros(self.num_children, dtype=np.int32)
        self.total_completed_tasks = 0
        self.total_sampled_tasks = 0
        if self.num_children == 0:
            raise ValueError("TaskTree must have at least one child")
        if names is not None:
            if len(names) != self.num_children:
                raise ValueError(f"Number of names ({len(names)}) must match number of children ({self.num_children})")
            self.names = names
        if weights is not None:
            if len(weights) != self.num_children:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match number of children ({self.num_children})"
                )
            self.weights = np.array(weights, dtype=np.float32)
        else:
            self.weights = np.ones(self.num_children, dtype=np.float32)
        if not np.all(self.weights >= 0):
            raise ValueError(f"Weights must be non-negative: {self.weights}")
        if self.weights.sum() <= 0:
            raise ValueError(f"Weights must be non-zero-sum. weights {self.weights} sum: {self.weights.sum()}")
        self.probabilities = self.weights / self.weights.sum()

        # Set parent references
        for i, child in enumerate(children):
            child.set_as_child(self, i, self.names[i] if self.names is not None else None)

    def sample(self) -> tuple[MettaGridTask, int]:
        child_idx = np.random.choice(self.num_children, p=self.probabilities)
        selected_child = self.children[child_idx]
        self.sampled_tasks[child_idx] += 1
        self.total_sampled_tasks += 1
        return selected_child.sample()

    def full_name(self, child_idx: int) -> str:
        name = self.children[child_idx].name
        if self.names is not None:
            return f"{self.names[child_idx]}/{name}"
        return name

    def complete_task(self, child_idx: int, score: float, name: Optional[str] = None):
        self.completed_tasks[child_idx] += 1
        self.total_completed_tasks += 1
        self.curriculum_algorithm.update_weights(self.weights, child_idx, score)
        self._update_probabilities()
        if self.parent is not None:
            full_name = name
            if self.names is not None:
                full_name = f"{self.names[child_idx]}/{name}"
            self.parent.complete_task(self.child_index, score, full_name)

    def get_completion_rates(self) -> dict[str, int]:
        if self.total_completed_tasks != 0:
            return self._completion_dict_with_prefix("task_completions/")
        else:
            return dict()

    def get_sample_rates(self) -> dict[str, int]:
        if self.total_sampled_tasks != 0:
            return self._sample_dict_with_prefix("task_samples/")
        else:
            return dict()

    def get_task_probabilities(self, relative_to_root: bool = False) -> dict[str, float]:
        return self._probability_dict_with_prefix(relative_to_root=relative_to_root)

    def get_curriculum_stats(self) -> dict[str, float]:
        return self.curriculum_algorithm.stats()

    def _update_probabilities(self):
        assert self.weights.sum() > 0, f"Weights must be non-zero-sum. weights {self.weights} sum: {self.weights.sum()}"
        assert np.all(self.weights >= 0), f"Weights must be non-negative. weights {self.weights}"
        self.probabilities = self.weights / self.weights.sum()

    def _completion_dict_with_prefix(self, prefix: str = "") -> dict[str, int]:
        result = dict()
        for child_idx in range(self.num_children):
            result[f"{prefix}{self.full_name(child_idx)}"] = self.completed_tasks[child_idx]
            if isinstance(self.children[child_idx], TaskTree):
                result.update(
                    self.children[child_idx]._completion_dict_with_prefix(f"{prefix}{self.full_name(child_idx)}/")
                )
        return result

    def _sample_dict_with_prefix(self, prefix: str = "") -> dict[str, int]:
        result = dict()
        for child_idx in range(self.num_children):
            result[f"{prefix}{self.full_name(child_idx)}"] = self.sampled_tasks[child_idx]
            if isinstance(self.children[child_idx], TaskTree):
                result.update(
                    self.children[child_idx]._sample_dict_with_prefix(f"{prefix}{self.full_name(child_idx)}/")
                )
        return result

    def _probability_dict_with_prefix(
        self, prefix: str = "", relative_to_root: bool = False, base_prob: float = 1.0
    ) -> dict[str, float]:
        """
        If relative_to_root is True, then the probabiltiies for each chld are for the full path from root to child.
        If relative_to_root is False, then the probabiltiies are for conditional on having reached this node.
        If base_prob is provided, then the probabilities are multiplied by this value.
        """
        probs = {}
        for child_idx in range(self.num_children):
            child_prob = self.probabilities[child_idx]
            if relative_to_root:
                child_prob = child_prob * base_prob
            probs[f"{prefix}{self.full_name(child_idx)}"] = child_prob
            if isinstance(self.children[child_idx], TaskTree):
                probs.update(
                    self.children[child_idx]._probability_dict_with_prefix(
                        f"{prefix}{self.full_name(child_idx)}/",
                        relative_to_root,
                        self.probabilities[child_idx] * base_prob,
                    )
                )

        return probs

    def __repr__(self) -> str:
        """Return a tree representation showing structure, weights, and algorithms."""
        return self._tree_repr(indent=0)

    def _tree_repr(self, indent: int = 0, prefix: str = "") -> str:
        """Recursive helper to build tree representation."""
        indent_str = "  " * indent
        lines = []

        # Current node info
        algo_name = type(self.curriculum_algorithm).__name__
        lines.append(f"{indent_str}{prefix}TaskTree({algo_name})")

        # Show weights and probabilities for children
        for i, child in enumerate(self.children):
            weight = self.weights[i]
            prob = self.probabilities[i]
            child_name = self.names[i] if self.names is not None else f"child_{i}"

            # Prefix for last child vs others
            is_last = i == len(self.children) - 1
            branch = "└─" if is_last else "├─"
            continuation = "  " if is_last else "│ "

            # Child info
            child_info = f"{indent_str}{branch} [{child_name}] w={weight:.3f} p={prob:.3f}"

            if isinstance(child, TaskTree):
                # Recursive case
                lines.append(child_info)
                subtree = child._tree_repr(indent + 1, prefix=continuation)
                lines.append(subtree)
            else:
                # Leaf case (MettaGridTask)
                lines.append(f"{child_info} -> {child.name}")

        return "\n".join(lines)


def task_set(
    task_configs: dict[str, DictConfig],
    env_overrides: Optional[DictConfig] = None,
    curriculum_algorithm: Optional[CurriculumAlgorithm] = None,
    task_weights: Optional[dict[str, float]] = None,
) -> TaskTree:
    """Helper function to create TaskTree from dictionary-based inputs.

    Args:
        curriculum_algorithm: Algorithm for updating weights based on task performance
        task_configs: Dict mapping config paths to DictConfig objects
        task_weights: Optional dict mapping task names to weights

    Returns:
        TaskTree initialized with the given configuration
    """

    def short_name(config_path: str) -> str:
        return config_path.split("/")[-1]

    def config_with_overrides(config: DictConfig, env_overrides: Optional[DictConfig]) -> DictConfig:
        """Create a new config with overrides applied."""
        if env_overrides is None:
            return config
        config = config.copy()
        OmegaConf.merge(config, env_overrides)
        return config

    tasks = [MettaGridTask(short_name(k), config_with_overrides(v, env_overrides)) for k, v in task_configs.items()]

    if curriculum_algorithm is None:
        curriculum_algorithm = UniformRandomCurriculum()

    return TaskTree(curriculum_algorithm, tasks, weights=task_weights)
