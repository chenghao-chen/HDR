#!/bin/bash
#SBATCH --job-name=teacher_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --account=stanchan
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8      # <-- Increased from 4 to feed the bigger batches fast
#SBATCH --mem=80G              # <-- Scaled up memory headroom for batch data expansion
#SBATCH --time=48:00:00

# 1. Ensure directory structures exist
mkdir -p logs

# 2. Force Python to output terminal text immediately
export PYTHONUNBUFFERED=1

# 3. Securely pass your Weights & Biases API credentials to the automated worker node
# Replace the string below with the actual key from https://wandb.ai/authorize
set -a
source /scratch/gilbreth/chen4848/projects/HDR/.env
set +a

# 4. Set WANDB to online mode explicitly so it syncs up to the cloud dashboard
export WANDB_MODE=online

# 5. Direct Execution via the Absolute Path to your Environment Binary
/scratch/gilbreth/chen4848/.conda/envs/2025.06-py313/dl/bin/python train_A100_MoE_two_phase.py