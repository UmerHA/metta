#!/usr/bin/env python3
"""Simple memory leak test that tracks specific object sizes."""

import gc
import logging
import os
import sys
import psutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metta.util.memory_tracker import MemoryTracker, format_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def test_memory_tracker():
    """Test the memory tracker functionality."""
    tracker = MemoryTracker()

    # Test tracking different objects
    test_dict = {}
    test_list = []

    logger.info("Testing memory tracker...")

    # Simulate growing objects over "epochs"
    for epoch in range(0, 100, 10):
        # Grow the dictionary
        for i in range(100):
            test_dict[f"key_{epoch}_{i}"] = [0] * 1000

        # Grow the list
        test_list.extend([0] * 10000)

        # Track memory usage
        tracker.track("test_dict", test_dict, epoch)
        tracker.track("test_list", test_list, epoch)

        if epoch % 20 == 0:
            tracker.print_report(epoch, logger)

    # Check for potential leaks
    leaks = tracker.get_potential_leaks(threshold_bytes_per_epoch=50000)  # 50KB per epoch
    if leaks:
        logger.warning(f"Potential memory leaks detected in: {leaks}")
    else:
        logger.info("No significant memory leaks detected")

    # Final report
    logger.info("\nFinal Memory Report:")
    tracker.print_report(90, logger)

    # Clean up
    del test_dict
    del test_list
    gc.collect()


def test_experience_lstm_states():
    """Test LSTM state management in Experience class."""
    import torch
    from metta.rl.experience import Experience
    import numpy as np

    logger.info("\nTesting Experience LSTM state management...")

    # Create a mock environment space
    class MockSpace:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    obs_space = MockSpace((84, 84, 3), np.uint8)
    atn_space = MockSpace((2,), np.int32)

    # Create experience buffer
    experience = Experience(
        total_agents=10,
        batch_size=100,
        bptt_horizon=10,
        minibatch_size=10,
        max_minibatch_size=10,
        obs_space=obs_space,
        atn_space=atn_space,
        device="cpu",
        hidden_size=256,
        cpu_offload=False,
        num_lstm_layers=2,
    )

    tracker = MemoryTracker()

    # Initial tracking
    tracker.track("lstm_h", experience.lstm_h, 0)
    tracker.track("lstm_c", experience.lstm_c, 0)

    logger.info(f"Initial LSTM states: {len(experience.lstm_h)} env_ids")

    # Simulate adding more LSTM states (this shouldn't happen in normal operation)
    for i in range(1, 5):
        # Add some invalid env_ids
        experience.lstm_h[100 + i] = torch.zeros(2, 10, 256)
        experience.lstm_c[100 + i] = torch.zeros(2, 10, 256)

        tracker.track("lstm_h", experience.lstm_h, i * 10)
        tracker.track("lstm_c", experience.lstm_c, i * 10)

    logger.info(f"After adding invalid states: {len(experience.lstm_h)} env_ids")

    # Test cleanup
    valid_env_ids = {0}  # Only env_id 0 is valid
    experience.cleanup_lstm_states(valid_env_ids)

    logger.info(f"After cleanup: {len(experience.lstm_h)} env_ids")

    # Final tracking
    tracker.track("lstm_h", experience.lstm_h, 50)
    tracker.track("lstm_c", experience.lstm_c, 50)

    tracker.print_report(50, logger)


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Memory Tracker Test")
    logger.info("="*60)

    initial_memory = get_memory_mb()
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

    # Test memory tracker
    test_memory_tracker()

    # Test experience LSTM states
    test_experience_lstm_states()

    # Final memory
    gc.collect()
    final_memory = get_memory_mb()

    logger.info("\n" + "="*60)
    logger.info("Test Complete")
    logger.info("="*60)
    logger.info(f"Initial memory: {initial_memory:.2f} MB")
    logger.info(f"Final memory: {final_memory:.2f} MB")
    logger.info(f"Memory growth: {final_memory - initial_memory:.2f} MB")
