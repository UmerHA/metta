# Memory Tracking During Training

The MettaTrainer includes built-in memory tracking to help identify memory leaks during training runs.

## How It Works

The memory tracker monitors the size of key Python objects every 10 epochs and generates detailed reports every 100 epochs (configurable). It tracks:

- **Experience Buffer Components**:
  - `experience` - The entire experience buffer object
  - `experience.obs` - Observation tensors
  - `experience.actions` - Action tensors
  - `experience.rewards` - Reward tensors
  - `experience.values` - Value estimates
  - `experience.lstm_h/lstm_c` - LSTM hidden states

- **Training State**:
  - `trainer.stats` - Statistics dictionary
  - `trainer.losses` - Loss tracking object
  - `trainer.evals` - Evaluation results
  - `policy.state_dict` - Policy model weights
  - `optimizer.state_dict` - Optimizer state

- **Infrastructure**:
  - `policy_store` - Policy storage system
  - `system_monitor._metrics` - System monitoring history
  - `curriculum` - Curriculum state
  - `wandb_run` - Wandb logging state
  - `torch_profiler` - PyTorch profiler state

## Configuration

Add these options to your trainer config to customize memory tracking:

```yaml
trainer:
  memory_tracking_enabled: true     # Enable/disable memory tracking (default: true)
  memory_report_interval: 100       # Report every N epochs (default: 100)
  memory_leak_threshold_mb: 5.0     # Warn if growth > N MB/epoch (default: 5.0)
```

## Example Usage

When running training with memory tracking enabled:

```bash
python tools/train.py trainer=simple.medium trainer.memory_report_interval=50
```

## Sample Output

Every report interval, you'll see output like:

```
============================================================
Memory Report - Epoch 100
============================================================

Current Memory Usage:
  experience                           1.23 GB
  experience.obs                     892.45 MB
  policy.state_dict                   45.23 MB
  optimizer.state_dict                90.46 MB
  trainer.stats                        2.34 MB

Memory Growth:
  experience                         500.00 MB (5.00 MB/epoch)
  trainer.stats                       1.20 MB (12.00 KB/epoch)
============================================================

WARNING: Potential memory leaks detected in: ['experience']
```

## Interpreting Results

- **Normal Growth**: Some objects like `trainer.stats` may grow slowly as they accumulate metrics
- **Potential Leaks**: Objects growing > 5MB/epoch (configurable) are flagged as potential leaks
- **Fixed Size**: Objects like policy weights should remain constant size

## Wandb Integration

If using Wandb, memory metrics are automatically logged:
- `memory/{object_name}/size_mb` - Current size in MB
- `memory/{object_name}/growth_mb_per_epoch` - Growth rate in MB/epoch

## Debugging Memory Leaks

If you identify a potential leak:

1. Check the growth rate - is it linear or does it plateau?
2. Look at what's being stored in that object
3. Check if old data is being properly cleared/overwritten
4. Use the detailed tracking to identify which component is growing

Common causes:
- Accumulating lists/dicts without clearing
- Storing references that prevent garbage collection
- LSTM states for non-existent environments
- Unbounded history/logging data
