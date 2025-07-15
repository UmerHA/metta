"""
Prioritize Regressed Curriculum Algorithm for TaskTree.

This module implements the prioritize regressed algorithm as a CurriculumAlgorithm
that can be used with TaskTree nodes to prioritize tasks where current performance
has regressed relative to peak performance.
"""

import logging
from typing import Dict

import numpy as np

from metta.mettagrid.curriculum.task_tree import CurriculumAlgorithm

logger = logging.getLogger(__name__)


class PrioritizeRegressedAlgorithm(CurriculumAlgorithm):
    """Curriculum algorithm that prioritizes tasks where performance has regressed from peak.

    This algorithm tracks both the maximum reward achieved and the moving average of rewards
    for each task. Tasks with high max/average ratios get higher weight, meaning tasks where
    we've seen good performance but are currently performing poorly get prioritized.

    Weight calculation: weight[i] = epsilon + max_reward[i] / (average_reward[i] + epsilon)

    This means:
    - Tasks with no history get epsilon weight (minimal)
    - Tasks with consistent performance get weight ≈ 1.0
    - Tasks with regression (max >> average) get higher weights
    """

    def __init__(self, num_tasks: int, moving_avg_decay_rate: float = 0.01):
        """Initialize prioritize regressed algorithm.

        Args:
            num_tasks: Number of tasks this algorithm will manage
            moving_avg_decay_rate: Smoothing factor for moving average (0 = no update, 1 = replace)
        """
        self.num_tasks = num_tasks
        self.moving_avg_decay_rate = moving_avg_decay_rate
        self.reward_averages = np.zeros(num_tasks, dtype=np.float32)
        self.reward_maxes = np.zeros(num_tasks, dtype=np.float32)
        self.task_completed_count = np.zeros(num_tasks, dtype=np.int32)

        # Reference to owning TaskTree (set by TaskTree during initialization)
        self.task_tree = None

        # Small epsilon to avoid division by zero (same as original: 1e-6)
        self.epsilon = 1e-6

    def update_weights(self, weights: np.ndarray, child_idx: int, score: float) -> None:
        """Update task weights based on regression from peak performance.

        Args:
            weights: Current weights array to update in-place
            child_idx: Index of the child that completed a task
            score: Score achieved (between 0 and 1)

        Note:
            The weights array is updated in-place. The TaskTree will handle
            normalization automatically via its _update_probabilities() method.
        """
        if child_idx >= self.num_tasks or child_idx < 0:
            logger.warning(f"Invalid child_idx {child_idx} for {self.num_tasks} tasks")
            return

        # Update moving average for the completed task
        old_average = self.reward_averages[child_idx]
        self.reward_averages[child_idx] = (1 - self.moving_avg_decay_rate) * self.reward_averages[
            child_idx
        ] + self.moving_avg_decay_rate * score

        # Update maximum reward seen for this task
        self.reward_maxes[child_idx] = max(self.reward_maxes[child_idx], score)

        # Track completion count
        self.task_completed_count[child_idx] += 1

        # Debug logging with task name from context
        task_name = self.get_task_name(child_idx)
        logger.debug(
            f"Updated task {child_idx} ({task_name}): "
            f"reward mean({old_average:.3f} -> {self.reward_averages[child_idx]:.3f}), "
            f"max({self.reward_maxes[child_idx]:.3f}), "
            f"count({self.task_completed_count[child_idx]})"
        )

        # Recalculate all weights based on max/average ratios
        # This matches the original implementation exactly
        for i in range(self.num_tasks):
            # Weight = epsilon + max / (average + epsilon)
            # This gives higher weight to tasks with max >> average (regression)
            weights[i] = self.epsilon + self.reward_maxes[i] / (self.reward_averages[i] + self.epsilon)

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return regression statistics for logging.

        Args:
            prefix: Prefix to add to all stat keys

        Returns:
            Dictionary of statistics with optional prefix
        """
        stats = {}

        # Overall statistics
        completed_tasks = self.task_completed_count > 0
        if np.any(completed_tasks):
            stats["pr/num_completed_tasks"] = int(np.sum(completed_tasks))
            stats["pr/total_completions"] = int(np.sum(self.task_completed_count))
            stats["pr/mean_reward_average"] = float(np.mean(self.reward_averages[completed_tasks]))
            stats["pr/mean_reward_max"] = float(np.mean(self.reward_maxes[completed_tasks]))

            # Calculate regression metrics
            avg_nonzero = self.reward_averages[completed_tasks]
            max_values = self.reward_maxes[completed_tasks]
            regression_ratios = max_values / (avg_nonzero + self.epsilon)
            stats["pr/mean_regression_ratio"] = float(np.mean(regression_ratios))
            stats["pr/max_regression_ratio"] = float(np.max(regression_ratios))

            # Individual task statistics (with names from TaskTree context if available)
            for i in range(min(3, self.num_tasks)):
                if self.task_completed_count[i] > 0:
                    task_name = self.get_task_name(i)
                    task_prefix = f"pr/{task_name}"
                    stats[f"{task_prefix}/avg"] = float(self.reward_averages[i])
                    stats[f"{task_prefix}/max"] = float(self.reward_maxes[i])
                    stats[f"{task_prefix}/count"] = int(self.task_completed_count[i])
                    stats[f"{task_prefix}/regression_ratio"] = float(
                        self.reward_maxes[i] / (self.reward_averages[i] + self.epsilon)
                    )
        else:
            stats["pr/num_completed_tasks"] = 0
            stats["pr/total_completions"] = 0

        # Add prefix if provided
        if prefix:
            return {f"{prefix}{k}": v for k, v in stats.items()}
        return stats

    def get_task_regression_ratios(self) -> np.ndarray:
        """Get current regression ratios for all tasks.

        Returns:
            Array of regression ratios (max/average) for each task
        """
        ratios = np.zeros(self.num_tasks)
        for i in range(self.num_tasks):
            if self.task_completed_count[i] > 0:
                ratios[i] = self.reward_maxes[i] / (self.reward_averages[i] + self.epsilon)
            else:
                ratios[i] = self.epsilon
        return ratios

    def get_task_name(self, child_idx: int) -> str:
        """Helper method to get task name from TaskTree context.

        Args:
            child_idx: Index of the child task

        Returns:
            Task name if TaskTree context is available, otherwise generic name
        """
        if self.task_tree is not None and hasattr(self.task_tree, "full_name"):
            try:
                return self.task_tree.full_name(child_idx)
            except (IndexError, AttributeError):
                pass
        return f"task_{child_idx}"

    def reset_task_stats(self, task_idx: int) -> None:
        """Reset statistics for a specific task (useful for testing).

        Args:
            task_idx: Index of task to reset
        """
        if 0 <= task_idx < self.num_tasks:
            self.reward_averages[task_idx] = 0.0
            self.reward_maxes[task_idx] = 0.0
            self.task_completed_count[task_idx] = 0
