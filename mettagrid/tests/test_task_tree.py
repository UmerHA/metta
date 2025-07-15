"""Tests for TaskTree curriculum structure."""

import random
from collections import Counter

import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.task_tree import (
    CurriculumAlgorithm,
    MettaGridTask,
    TaskTree,
    UniformRandomCurriculum,
    task_set,
)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set all random seeds for deterministic test behavior."""
    random.seed(42)
    np.random.seed(42)
    yield
    # Reset after test
    random.seed()
    np.random.seed()


@pytest.fixture
def dummy_config():
    """Create a dummy DictConfig for testing."""
    return OmegaConf.create({"game": {"num_agents": 1}})


def print_sampling_results(tree: TaskTree, samples: list[MettaGridTask], test_name: str):
    """Pretty print sampling results for debugging."""
    print(f"\n{'=' * 60}")
    print(f"Test: {test_name}")
    print(f"{'=' * 60}")
    print("\nTree Structure:")
    print(tree)

    print(f"\nTotal samples: {len(samples)}")

    # Count samples
    sample_counts = Counter(task.name for task in samples)

    print("\nSampling Results:")
    print("-" * 40)
    print(f"{'Task Name':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 40)

    for task_name, count in sorted(sample_counts.items()):
        percentage = (count / len(samples)) * 100
        print(f"{task_name:<20} {count:<10} {percentage:<10.1f}%")

    # Get sample rates from tree
    sample_rates = tree.get_sample_rates()
    if sample_rates:
        print("\nTree Sample Rates:")
        print("-" * 40)
        for path, rate in sorted(sample_rates.items()):
            print(f"{path}: {rate:.3f}")

    # Get probabilities
    probs = tree.get_task_probabilities(relative_to_root=True)
    print("\nExpected Probabilities (relative to root):")
    print("-" * 40)
    for path, prob in sorted(probs.items()):
        print(f"{path}: {prob:.3f}")

    print("=" * 60)


def test_single_task_tree(dummy_config):
    """Test a TaskTree with a single task."""
    # Create a single task
    task = MettaGridTask("only_task", dummy_config)
    tree = TaskTree(name="root", curriculum_algorithm=UniformRandomCurriculum(), children=[task])

    # Sample 10 times
    samples = [tree.sample() for _ in range(10)]

    print_sampling_results(tree, samples, "Single Task Tree")

    # All samples should be the same task
    assert all(s.name == "only_task" for s in samples)
    assert tree.total_sampled_tasks == 10
    assert tree.sampled_tasks[0] == 10

    # Check sample rates
    rates = tree.get_sample_rates()
    assert rates["task_samples/only_task"] == 10


def test_three_tasks_uniform(dummy_config):
    """Test a TaskTree with 3 tasks and uniform weights."""
    # Create three tasks
    tasks = [
        MettaGridTask("task_a", dummy_config),
        MettaGridTask("task_b", dummy_config),
        MettaGridTask("task_c", dummy_config),
    ]

    tree = TaskTree(
        name="root",
        curriculum_algorithm=UniformRandomCurriculum(),
        children=tasks,
        weights=np.array([1.0, 1.0, 1.0]),  # Equal weights
    )

    # Sample 300 times (enough for reasonable distribution)
    samples = [tree.sample() for _ in range(300)]

    print_sampling_results(tree, samples, "Three Tasks - Uniform Weights")

    # Check that all tasks were sampled
    sample_counts = Counter(s.name for s in samples)
    assert len(sample_counts) == 3
    assert all(task_name in sample_counts for task_name in ["task_a", "task_b", "task_c"])

    # With uniform weights, each should get roughly 1/3
    for count in sample_counts.values():
        assert 80 < count < 120, f"Expected ~100 samples per task, got {count}"

    # Check sample rates (should be actual counts, not floats)
    rates = tree.get_sample_rates()
    total_samples = sum(rates.values())
    assert total_samples == 300


def test_three_tasks_skewed(dummy_config):
    """Test a TaskTree with 3 tasks and skewed weights."""
    tasks = [
        MettaGridTask("rare", dummy_config),
        MettaGridTask("common", dummy_config),
        MettaGridTask("very_common", dummy_config),
    ]

    # Very uneven weights: 1:4:15 ratio
    tree = TaskTree(
        name="root",
        curriculum_algorithm=UniformRandomCurriculum(),
        children=tasks,
        weights=np.array([1.0, 4.0, 15.0]),
    )

    # Sample 1000 times
    samples = [tree.sample() for _ in range(1000)]

    print_sampling_results(tree, samples, "Three Tasks - Skewed Weights")

    sample_counts = Counter(s.name for s in samples)

    # Check expected ratios (1:4:15 normalized = 0.05:0.2:0.75)
    assert 30 < sample_counts["rare"] < 70, f"Expected ~50 for rare, got {sample_counts['rare']}"
    assert 150 < sample_counts["common"] < 250, f"Expected ~200 for common, got {sample_counts['common']}"
    assert 700 < sample_counts["very_common"] < 800, (
        f"Expected ~750 for very_common, got {sample_counts['very_common']}"
    )

    # Check probabilities match weights
    probs = tree.get_task_probabilities()
    np.testing.assert_almost_equal(probs["rare"], 0.05, decimal=2)
    np.testing.assert_almost_equal(probs["common"], 0.2, decimal=2)
    np.testing.assert_almost_equal(probs["very_common"], 0.75, decimal=2)


def test_binary_tree_balanced(dummy_config):
    """Test a balanced binary tree of depth 3."""
    # Create leaf tasks (8 total for balanced binary tree of depth 3)
    leaf_tasks = [MettaGridTask(f"task_{i}", dummy_config) for i in range(8)]

    # Build tree bottom-up
    # Level 2: 4 nodes, each with 2 children
    level2_nodes = []
    for i in range(4):
        node = TaskTree(
            name=f"L2_{i}",
            curriculum_algorithm=UniformRandomCurriculum(),
            children=leaf_tasks[i * 2 : (i + 1) * 2],
            weights=np.array([1.0, 1.0]),  # Balanced
        )
        level2_nodes.append(node)

    # Level 1: 2 nodes, each with 2 children
    level1_nodes = []
    for i in range(2):
        node = TaskTree(
            name=f"L1_{i}",
            curriculum_algorithm=UniformRandomCurriculum(),
            children=level2_nodes[i * 2 : (i + 1) * 2],
            weights=np.array([1.0, 1.0]),  # Balanced
        )
        level1_nodes.append(node)

    # Root
    root = TaskTree(
        name="root",
        curriculum_algorithm=UniformRandomCurriculum(),
        children=level1_nodes,
        weights=np.array([1.0, 1.0]),  # Balanced
    )

    # Sample 1000 times
    samples = [root.sample() for _ in range(1000)]

    print_sampling_results(root, samples, "Binary Tree - Balanced")

    sample_counts = Counter(s.name for s in samples)

    # Each leaf should get roughly 1/8 of samples (125)
    for i in range(8):
        count = sample_counts[f"task_{i}"]
        assert 80 < count < 170, f"Expected ~125 for task_{i}, got {count}"

    # Check that probabilities are uniform
    probs = root.get_task_probabilities(relative_to_root=True)
    # Each leaf task should have probability 0.125 (1/8)
    leaf_probs = [v for k, v in probs.items() if k.endswith(tuple(f"/task_{i}" for i in range(8)))]
    assert len(leaf_probs) == 8
    for prob in leaf_probs:
        np.testing.assert_almost_equal(prob, 0.125, decimal=2)


def test_binary_tree_unbalanced(dummy_config):
    """Test an unbalanced binary tree where left branches are heavily weighted."""
    # Create leaf tasks
    leaf_tasks = [MettaGridTask(f"task_{i}", dummy_config) for i in range(8)]

    # Build tree with left-heavy weights
    # Level 2: 4 nodes
    level2_nodes = []
    for i in range(4):
        node = TaskTree(
            name=f"L2_{i}",
            curriculum_algorithm=UniformRandomCurriculum(),
            children=leaf_tasks[i * 2 : (i + 1) * 2],
            weights=np.array([3.0, 1.0]),  # Left child 3x more likely
        )
        level2_nodes.append(node)

    # Level 1: 2 nodes
    level1_nodes = []
    for i in range(2):
        node = TaskTree(
            name=f"L1_{i}",
            curriculum_algorithm=UniformRandomCurriculum(),
            children=level2_nodes[i * 2 : (i + 1) * 2],
            weights=np.array([3.0, 1.0]),  # Left child 3x more likely
        )
        level1_nodes.append(node)

    # Root
    root = TaskTree(
        name="root",
        curriculum_algorithm=UniformRandomCurriculum(),
        children=level1_nodes,
        weights=np.array([3.0, 1.0]),  # Left subtree 3x more likely
    )

    # Sample 1000 times
    samples = [root.sample() for _ in range(1000)]

    print_sampling_results(root, samples, "Binary Tree - Left-Heavy Unbalanced")

    sample_counts = Counter(s.name for s in samples)

    # Task 0 should be most common (left-left-left path)
    # Probability = 0.75 * 0.75 * 0.75 = 0.421875
    assert sample_counts["task_0"] > 350, f"task_0 should be most common, got {sample_counts['task_0']}"

    # Task 7 should be least common (right-right-right path)
    # Probability = 0.25 * 0.25 * 0.25 = 0.015625
    assert sample_counts["task_7"] < 50, f"task_7 should be least common, got {sample_counts['task_7']}"

    # Verify relative probabilities - look for the actual full paths
    probs = root.get_task_probabilities(relative_to_root=True)
    # Find task_0 and task_7 probabilities by searching through all paths
    task_0_prob = None
    task_7_prob = None
    for path, prob in probs.items():
        if path.endswith("/task_0"):
            task_0_prob = prob
        elif path.endswith("/task_7"):
            task_7_prob = prob
    
    assert task_0_prob is not None and task_0_prob > 0.4, f"task_0 probability should be > 0.4, got {task_0_prob}"
    assert task_7_prob is not None and task_7_prob < 0.02, f"task_7 probability should be < 0.02, got {task_7_prob}"


def test_task_set_helper(dummy_config):
    """Test the task_set helper function."""
    # Create task configs
    task_configs = {
        "/env/easy": OmegaConf.create({"difficulty": 1}),
        "/env/medium": OmegaConf.create({"difficulty": 2}),
        "/env/hard": OmegaConf.create({"difficulty": 3}),
    }

    # Create weights
    weights = {
        "/env/easy": 3.0,
        "/env/medium": 2.0,
        "/env/hard": 1.0,
    }

    tree = task_set(name="root", task_configs=task_configs, task_weights=weights, curriculum_algorithm=UniformRandomCurriculum())

    print("\nTask Set Helper Test:")
    print(tree)

    # Check structure
    assert tree.num_children == 3
    # Note: names are not set by task_set, they come from child names
    # Check weights are correctly assigned (order depends on dict iteration)
    assert tree.weights.sum() == 6.0  # 3 + 2 + 1

    # Sample and check distribution
    samples = [tree.sample() for _ in range(600)]
    sample_counts = Counter(s.name for s in samples)

    # With 3:2:1 weights, expect roughly 300:200:100
    assert 250 < sample_counts["easy"] < 350
    assert 150 < sample_counts["medium"] < 250
    assert 50 < sample_counts["hard"] < 150


def test_deep_tree_traversal(dummy_config):
    """Test that sampling correctly traverses deep trees."""
    # Create a deep tree: Root -> A -> B -> C -> task
    task = MettaGridTask("deep_task", dummy_config)

    c = TaskTree("C", UniformRandomCurriculum(), [task])
    b = TaskTree("B", UniformRandomCurriculum(), [c])
    a = TaskTree("A", UniformRandomCurriculum(), [b])
    root = TaskTree("root", UniformRandomCurriculum(), [a])

    # Sample should traverse all the way down
    sampled = root.sample()
    assert sampled.name == "deep_task"

    # Check that sample counts propagate correctly
    assert root.sampled_tasks[0] == 1
    assert a.sampled_tasks[0] == 1
    assert b.sampled_tasks[0] == 1
    assert c.sampled_tasks[0] == 1

    # Check the path in probabilities
    probs = root.get_task_probabilities(relative_to_root=True)
    # Find the deep_task probability
    deep_task_prob = None
    for path, prob in probs.items():
        if path.endswith("/deep_task"):
            deep_task_prob = prob
            break
    assert deep_task_prob == 1.0


def test_empty_tree_error():
    """Test that creating a tree with no children raises an error."""
    with pytest.raises(ValueError, match="TaskTree must have at least one child"):
        TaskTree("root", UniformRandomCurriculum(), [])


def test_weight_validation():
    """Test weight validation in TaskTree."""
    task = MettaGridTask("task", OmegaConf.create({}))

    # Negative weights should raise error
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        TaskTree("root", UniformRandomCurriculum(), [task], weights=np.array([-1.0]))

    # All-zero weights should raise error
    with pytest.raises(ValueError, match="Weights must be non-zero-sum"):
        TaskTree("root", UniformRandomCurriculum(), [task], weights=np.array([0.0]))


def test_probability_updates_after_weight_change(dummy_config):
    """Test that probabilities update correctly when weights change."""

    # Custom algorithm that zeros out a weight
    class ZeroingAlgorithm(CurriculumAlgorithm):
        def update_weights(self, weights: np.ndarray, child_idx: int, score: float):
            weights[child_idx] = 0.0

    tasks = [MettaGridTask(f"task_{i}", dummy_config) for i in range(3)]
    tree = TaskTree("root", ZeroingAlgorithm(), tasks, weights=np.array([1.0, 1.0, 1.0]))

    # Initially all equal
    np.testing.assert_array_almost_equal(tree.probabilities, [1 / 3, 1 / 3, 1 / 3])

    # Complete task 0 (which zeros its weight)
    tree.complete_task(0, 1.0)

    # Now task 0 should have 0 probability
    np.testing.assert_array_almost_equal(tree.probabilities, [0.0, 0.5, 0.5])

    # Sampling should never select task 0
    samples = [tree.sample() for _ in range(100)]
    assert all(s.name != "task_0" for s in samples)


if __name__ == "__main__":
    # Run with pretty output
    pytest.main([__file__, "-v", "-s"])
