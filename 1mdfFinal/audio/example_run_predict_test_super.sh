#!/bin/sh
#SBATCH --job-name=a_predict_1m_test_with_last_batch_final
#SBATCH --output=predict_1m_test_with_last_batch_%j_%a.out
#SBATCH --error=predict_1m_test_with_last_batch_%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4  # Request one GPU per task
#SBATCH --mem=0
#SBATCH --cpus-per-task=30
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-3  # 8 workers, IDs range from 0 to 7

# Load Conda environment
source /ceph/hpc/data/st2207-pgp-users/ldragar/miniconda3/etc/profile.d/conda.sh
conda activate /ceph/hpc/data/st2207-pgp-users/ldragar/pytorch_env

# # Get the worker ID from SLURM_ARRAY_TASK_ID
# WORKER_ID=$SLURM_ARRAY_TASK_ID
# NUM_WORKERS=24  # Set this to match the --array range

# # Run the script with the specified number of workers
# srun --gpus-per-node=1 -p gpu --ntasks-per-node=1 python3 predict_test_with_last_batch_res2.py --num_workers $NUM_WORKERS --worker_id $WORKER_ID



# Get the worker ID from SLURM_ARRAY_TASK_ID
WORKER_ID=$SLURM_ARRAY_TASK_ID
NUM_WORKERS=4  # Set this to match the --array range
NUM_GPUS=4  # Number of GPUs per node

# Loop over the GPUs in the node and launch a separate process for each GPU
# Loop over the GPUs in the node and launch a separate process for each GPU
# for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
#     # Redirect the output to specific log files for each GPU process
#     OUTPUT_LOG=./predict_test_visual_logs_gpu_w_gpu/predict_1m_visual_${WORKER_ID}_gpu_${GPU_ID}_%j.out
#     ERROR_LOG=./predict_test_visual_logs_gpu_w_gpu/predict_1m_visual_${WORKER_ID}_gpu_${GPU_ID}_%j.err
#     srun --exclusive --gres=gpu:1 -p gpu python3 predict_timm_1MDF_on_test_only_visual_gpus.py --num_workers $NUM_WORKERS --worker_id $WORKER_ID --gpu_id $GPU_ID --num_gpus $NUM_GPUS > $OUTPUT_LOG 2> $ERROR_LOG &
# done
# Create an array to hold process IDs
# Create a temporary tasks file for this SLURM_ARRAY_TASK_ID
TASKS_FILE=tasks_$WORKER_ID.conf

# Generate the tasks.conf file dynamically
# cat <<EOL > $TASKS_FILE
# 0 python3 predict_timm_1MDF_on_test_only_visual_gpus.py --num_workers $NUM_WORKERS --worker_id $WORKER_ID --gpu_id 0 --num_gpus $NUM_GPUS 
# 1 python3 predict_timm_1MDF_on_test_only_visual_gpus.py --num_workers $NUM_WORKERS --worker_id $WORKER_ID --gpu_id 1 --num_gpus $NUM_GPUS 
# 2 python3 predict_timm_1MDF_on_test_only_visual_gpus.py --num_workers $NUM_WORKERS --worker_id $WORKER_ID --gpu_id 2 --num_gpus $NUM_GPUS 
# 3 python3 predict_timm_1MDF_on_test_only_visual_gpus.py --num_workers $NUM_WORKERS --worker_id $WORKER_ID --gpu_id 3 --num_gpus $NUM_GPUS
# EOL
#    parser.add_argument("--model", type=str, default="/ceph/hpc/data/st2207-pgp-users/models_luka_1mdf_audio_res_2/TDL_res2/er8pbob5/TDL_res2-epoch=31-val_loss=0.02-train_loss=0.02.ckpt")
    #parser.add_argument('--output_dir', type=str, default='/ceph/hpc/data/st2207-pgp-users/ldragar/TDL-ADD/predict_test_with_last_batch_er8pbob5_epoch31/')

model_path="/ceph/hpc/data/st2207-pgp-users/models_luka_1mdf_audio_res_2/TDL_res2/er8pbob5/TDL_res2-epoch=36-val_loss=0.02-train_loss=0.02.ckpt"
output_dir="/ceph/hpc/data/st2207-pgp-users/ldragar/TDL-ADD/predict_test_with_last_batch_er8pbob5_epoch36/"

cat <<EOL > $TASKS_FILE
0 bash -c 'CUDA_VISIBLE_DEVICES=0 python3 predict_test_with_last_batch_res2_super.py --num_workers $NUM_WORKERS --worker_id $WORKER_ID --gpu_id 0 --num_gpus $NUM_GPUS --model $model_path --output_dir $output_dir'
1 bash -c 'CUDA_VISIBLE_DEVICES=1 python3 predict_test_with_last_batch_res2_super.py --num_workers $NUM_WORKERS --worker_id $WORKER_ID --gpu_id 1 --num_gpus $NUM_GPUS --model $model_path --output_dir $output_dir'
2 bash -c 'CUDA_VISIBLE_DEVICES=2 python3 predict_test_with_last_batch_res2_super.py --num_workers $NUM_WORKERS --worker_id $WORKER_ID --gpu_id 2 --num_gpus $NUM_GPUS --model $model_path --output_dir $output_dir'
3 bash -c 'CUDA_VISIBLE_DEVICES=3 python3 predict_test_with_last_batch_res2_super.py --num_workers $NUM_WORKERS --worker_id $WORKER_ID --gpu_id 3 --num_gpus $NUM_GPUS --model $model_path --output_dir $output_dir'
EOL

# Run the tasks using srun with the --multi-prog option
srun -p gpu --multi-prog $TASKS_FILE 

# Clean up the temporary tasks file
rm $TASKS_FILE
# # Run the script with the specified number of workers
# srun --gpus-per-node=1 -p gpu --ntasks-per-node=1 python3 predict_timm_1MDF_on_test_only_visual_gpus.py --num_workers $NUM_WORKERS --worker_id $WORKER_ID --gpu_id 0
