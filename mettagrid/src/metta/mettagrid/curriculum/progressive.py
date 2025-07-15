"""
Progressive Curriculum Algorithm for TaskTree.

This module implements progressive curriculum algorithms as CurriculumAlgorithms
that can be used with TaskTree nodes to progressively advance through tasks
based on performance thresholds or time-based progression.
"""

import logging
from typing import Dict, Optional, Set

import numpy as np

from metta.mettagrid.curriculum.task_tree import CurriculumAlgorithm

logger = logging.getLogger(__name__)


class ProgressiveAlgorithm(CurriculumAlgorithm):
    """Curriculum algorithm that progressively advances through tasks based on performance.

    This algorithm blends multiple tasks using gating mechanisms and advances progression
    based on smoothed performance or time. Tasks are weighted using a gating function
    that creates smooth transitions between tasks as the agent progresses.

    The algorithm maintains a progress value [0, 1] that determines task weighting:
    - progress=0: Focus on first task
    - progress=0.5: Blend middle tasks
    - progress=1: Focus on last task

    Progress advances when:
    - performance mode: smoothed performance exceeds threshold
    - time mode: fixed progression rate per step
    """

    def __init__(
        self,
        num_tasks: int,
        performance_threshold: float = 0.8,
        smoothing: float = 0.1,
        progression_rate: float = 0.01,
        progression_mode: str = "perf",
        blending_smoothness: float = 0.5,
        blending_mode: str = "logistic",
    ):
        """Initialize progressive algorithm.

        Args:
            num_tasks: Number of tasks this algorithm will manage
            performance_threshold: Performance level required to advance (perf mode)
            smoothing: Smoothing factor for performance tracking
            progression_rate: Rate of progression per step/threshold crossing
            progression_mode: "perf" (performance-based) or "time" (time-based)
            blending_smoothness: Controls transition smoothness between tasks
            blending_mode: "logistic" or "linear" blending function
        """
        if progression_mode not in ["time", "perf"]:
            raise ValueError("progression_mode must be either 'time' or 'perf'")
        if blending_mode not in ["logistic", "linear"]:
            raise ValueError("blending_mode must be either 'logistic' or 'linear'")

        self.num_tasks = num_tasks
        self.performance_threshold = performance_threshold
        self.smoothing = smoothing
        self.progression_rate = progression_rate
        self.progression_mode = progression_mode
        self.blending_smoothness = blending_smoothness
        self.blending_mode = blending_mode

        # State tracking
        self.progress = 0.0  # Progress parameter [0, 1]
        self.smoothed_performance = 0.0
        self.step_count = 0
        self.last_score = None

        # Reference to owning TaskTree (set by TaskTree during initialization)
        self.task_tree = None

    def update_weights(self, weights: np.ndarray, child_idx: int, score: float) -> None:
        """Update task weights based on progressive curriculum logic.

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

        # Update smoothed performance
        self._update_smoothed_performance(score)

        # Advance progression based on mode
        self._advance_progression()

        # Update task weights using progressive gating
        self._update_progressive_weights(weights)

    def _update_smoothed_performance(self, score: float):
        """Update the smoothed performance estimate."""
        if self.last_score is None:
            self.smoothed_performance = score
        else:
            self.smoothed_performance = self.smoothing * score + (1 - self.smoothing) * self.smoothed_performance
        self.last_score = score

    def _advance_progression(self):
        """Advance the progression parameter based on the progression mode."""
        if self.progression_mode == "perf":
            if self.smoothed_performance >= self.performance_threshold:
                self.progress = min(1.0, self.progress + self.progression_rate)
        elif self.progression_mode == "time":
            self.step_count += 1
            self.progress = min(1.0, self.step_count * self.progression_rate)

    def _blending_function(self, x: float, xo: float, growing: bool = True) -> float:
        """Blending function that supports both logistic and linear modes."""
        if self.blending_mode == "logistic":
            return 1 / (1 + np.exp(-((-1) ** growing) * (x - xo) / self.blending_smoothness))
        elif self.blending_mode == "linear":
            # Linear blending with smoothness control
            if growing:
                # Growing function: 0 at xo, 1 at xo + blending_smoothness
                return max(0, min(1, (x - xo + self.blending_smoothness) / self.blending_smoothness))
            else:
                # Shrinking function: 1 at xo, 0 at xo + blending_smoothness
                return max(0, min(1, (xo + self.blending_smoothness - x) / self.blending_smoothness))

    def _update_progressive_weights(self, weights: np.ndarray):
        """Update weights using progressive gating mechanism."""
        # Scale progress to task space (0 to num_tasks-1)
        p = self.progress * (self.num_tasks - 1)

        # Task positions (0, 1, 2, ..., num_tasks-1)
        task_positions = np.arange(self.num_tasks)

        # Create gating matrix: tasks x progress points
        gating = np.zeros(self.num_tasks)

        for i, task_pos in enumerate(task_positions):
            # Double gating: activation and deactivation
            activation = self._blending_function(p, task_pos - self.blending_smoothness, growing=True)
            deactivation = self._blending_function(p, task_pos + self.blending_smoothness, growing=False)
            gating[i] = activation * deactivation

        # Normalize to get probabilities
        if np.sum(gating) > 0:
            probs = gating / np.sum(gating)
        else:
            # Fallback to uniform distribution if all gates are zero
            probs = np.ones(self.num_tasks) / self.num_tasks

        # Update weights array in-place
        for i in range(self.num_tasks):
            weights[i] = probs[i]

        logger.debug(
            f"Progress: {self.progress:.3f}, smoothed_perf: {self.smoothed_performance:.3f}, "
            f"weights: {weights[: min(5, self.num_tasks)]}"  # Log first 5 weights
        )

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return progressive curriculum statistics for logging.

        Args:
            prefix: Prefix to add to all stat keys

        Returns:
            Dictionary of statistics with optional prefix
        """
        stats = {
            "prog/smoothed_performance": self.smoothed_performance,
            "prog/progress": self.progress,
            "prog/step_count": float(self.step_count),
        }

        # Add mode-specific stats
        if self.progression_mode == "perf":
            stats["prog/threshold_diff"] = self.smoothed_performance - self.performance_threshold

        # Add prefix if provided
        if prefix:
            return {f"{prefix}{k}": v for k, v in stats.items()}
        return stats

    def reset_progress(self) -> None:
        """Reset progression state (useful for testing)."""
        self.progress = 0.0
        self.smoothed_performance = 0.0
        self.step_count = 0
        self.last_score = None


class SimpleProgressiveAlgorithm(CurriculumAlgorithm):
    """Simplified progressive algorithm that advances based on score thresholds.

    This is a simpler version that directly advances to the next task when
    a score threshold is met, similar to the original ProgressiveCurriculum
    but adapted for the TaskTree framework.
    """

    def __init__(self, num_tasks: int, score_threshold: float = 0.5):
        """Initialize simple progressive algorithm.

        Args:
            num_tasks: Number of tasks this algorithm will manage
            score_threshold: Score threshold to advance to next task
        """
        self.num_tasks = num_tasks
        self.score_threshold = score_threshold
        self.current_task = 0  # Index of current active task

        # Reference to owning TaskTree (set by TaskTree during initialization)
        self.task_tree = None

    def update_weights(self, weights: np.ndarray, child_idx: int, score: float) -> None:
        """Update task weights based on simple progression logic.

        Args:
            weights: Current weights array to update in-place
            child_idx: Index of the child that completed a task
            score: Score achieved (between 0 and 1)
        """
        if child_idx >= self.num_tasks or child_idx < 0:
            logger.warning(f"Invalid child_idx {child_idx} for {self.num_tasks} tasks")
            return

        # Check if we should advance to next task
        if score > self.score_threshold and child_idx == self.current_task:
            self.current_task = min(self.current_task + 1, self.num_tasks - 1)
            logger.debug(f"Advanced to task {self.current_task} after score {score:.3f}")

        # Set weights: current task gets weight 1, others get weight 0
        weights.fill(0.0)
        weights[self.current_task] = 1.0

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return simple progressive curriculum statistics.

        Args:
            prefix: Prefix to add to all stat keys

        Returns:
            Dictionary of statistics with optional prefix
        """
        stats = {
            "simple_prog/current_task": float(self.current_task),
            "simple_prog/progress_ratio": self.current_task / max(1, self.num_tasks - 1),
        }

        # Add prefix if provided
        if prefix:
            return {f"{prefix}{k}": v for k, v in stats.items()}
        return stats

    def reset_progress(self) -> None:
        """Reset to first task (useful for testing)."""
        self.current_task = 0
