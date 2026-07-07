#!/bin/bash
#SBATCH --job-name=teacher_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --account=stanchan
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00

# 1. Ensure directory structures exist
mkdir -p logs

# 2. Force Python to output terminal text immediately
export PYTHONUNBUFFERED=1

# 3. Credentials (needed only if W&B is used by the test script — safe to keep)
set -a
source /scratch/gilbreth/chen4848/projects/HDR/.env
set +a

# 4. Run evaluation
/scratch/gilbreth/chen4848/.conda/envs/2025.06-py313/dl/bin/python test_dual_MoE_two_phase.py
