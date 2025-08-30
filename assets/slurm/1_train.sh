#!/bin/bash
#SBATCH -J seflow
#SBATCH --gpus 4 -C "fat"
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qingwen@kth.se
#SBATCH --output /proj/berzelius-2023-154/users/x_qinzh/seflow/logs/slurm/%J_seflow.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_qinzh/seflow/logs/slurm/%J_seflow.err

PYTHON=/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/opensf/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib
cd /proj/berzelius-2023-364/users/x_qinzh/workspace/OpenSceneFlow


# ===> to transfer data into local node disk, it can be ignored. <===
SOURCE="/proj/berzelius-2023-364/users/x_qinzh/data/av2/autolabel"
DEST="/scratch/local/av2"
SUBDIRS=("sensor/train" "sensor/val")

start_time=$(date +%s)
for dir in "${SUBDIRS[@]}"; do
    mkdir -p "${DEST}/${dir}"
    find "${SOURCE}/${dir}" -type f -print0 | xargs -0 -n1 -P16 cp -t "${DEST}/${dir}" &
done
wait
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Copy ${SOURCE} to ${DEST} Total time: ${elapsed} seconds"
echo "Start training..."

# ====> leaderboard model = seflow_best
$PYTHON train.py slurm_id=$SLURM_JOB_ID wandb_mode=online train_data=/scratch/local/av2/sensor/train val_data=/scratch/local/av2/sensor/val \
    num_workers=16 model=deflow lr=2e-4 epochs=9 batch_size=16 "model.target.num_iters=2" "model.val_monitor=val/Dynamic/Mean" \
    loss_fn=seflowLoss "add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}"
