#!/usr/bin/env python3
"""Test script to check for memory leaks in the trainer.

This script runs a minimal training loop and monitors memory usage
to help identify potential memory leaks.
"""

import gc
import logging
import os
import sys

import psutil
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metta.rl.trainer import MettaTrainer
from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.wandb.wandb_context import WandbRun

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_memory_test(epochs: int = 500):
    """Run a short training session to test for memory leaks."""
    # Initialize Hydra configuration
    config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    config_dir = os.path.abspath(config_dir)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Load a minimal config for testing
        cfg = compose(
            config_name="train_job",
            overrides=[
                "env=mettagrid/ants",
                "agent=fast",
                "trainer=simple.medium",
                "trainer.total_timesteps=100000",
                "trainer.batch_size=512",
                "trainer.minibatch_size=128",
                "trainer.evaluate_interval=0",  # Disable evaluation
                "trainer.checkpoint_interval=100",
                "trainer.wandb_checkpoint_interval=0",  # Disable wandb uploads
                "trainer.replay_interval=0",  # Disable replay generation
                "device=cpu",  # Use CPU for testing
                "+run_dir=/tmp/memory_test",
            ]
        )

        # Create necessary directories
        os.makedirs(cfg.run_dir, exist_ok=True)

        # Initialize components
        policy_store = PolicyStore(cfg)
        sim_suite_config = SimulationSuiteConfig(simulations={})

        # Record initial memory
        gc.collect()
        initial_memory = get_memory_mb()
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

        # Create trainer
        trainer = MettaTrainer(
            cfg=cfg,
            wandb_run=None,  # No wandb for testing
            policy_store=policy_store,
            sim_suite_config=sim_suite_config,
            stats_client=None,
        )

        # Override total timesteps to run for specific number of epochs
        original_timesteps = cfg.trainer.total_timesteps
        trainer.trainer_cfg.total_timesteps = trainer.agent_step + (epochs * trainer._batch_size)

        # Run training
        logger.info(f"Running training for {epochs} epochs...")
        trainer.train()

        # Record final memory
        gc.collect()
        final_memory = get_memory_mb()
        memory_growth = final_memory - initial_memory

        logger.info(f"Final memory usage: {final_memory:.2f} MB")
        logger.info(f"Memory growth: {memory_growth:.2f} MB")
        logger.info(f"Memory growth per epoch: {memory_growth / epochs:.4f} MB/epoch")

        # Check for excessive memory growth
        growth_per_epoch = memory_growth / epochs
        if growth_per_epoch > 1.0:  # More than 1MB per epoch
            logger.warning(f"High memory growth detected: {growth_per_epoch:.4f} MB/epoch")
        else:
            logger.info(f"Memory growth within acceptable limits: {growth_per_epoch:.4f} MB/epoch")

        # Clean up
        trainer.close()
        del trainer
        gc.collect()

        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": memory_growth,
            "growth_per_epoch_mb": growth_per_epoch,
            "epochs": epochs,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test for memory leaks in the trainer")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to run")
    args = parser.parse_args()

    results = run_memory_test(epochs=args.epochs)

    print("\n" + "="*60)
    print("Memory Test Results:")
    print("="*60)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("="*60)
