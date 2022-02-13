#!/bin/bash -e
#SBATCH --job-name=LSR            # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/hoanglh88/LSR/slurms/Feb12.out      # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/hoanglh88/LSR/slurms/Feb12.err       # create a error file
#SBATCH --partition=research
#SBATCH --gres=gpu:1              # gpu count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --nodes=1                  # node count
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4          # cpu-cores per task (>1 if multi-threaded tasks)


export HTTP_PROXY=http://proxytc.vingroup.net:9090/ \
export HTTPS_PROXY=http://proxytc.vingroup.net:9090/ \
export http_proxy=http://proxytc.vingroup.net:9090/ \
export https_proxy=http://proxytc.vingroup.net:9090/



srun --container-image=/lustre/scratch/client/vinai/users/hoanglh88/dc-miniconda3-py:38-4.10.3-cuda11.4.2-cudnn8-ubuntu20.04.sqsh \
     --container-mounts=/lustre/scratch/client/vinai/users/hoanglh88/:/hoanglh88/ \
     /bin/bash -c \
     "
     export HTTP_PROXY=http://proxytc.vingroup.net:9090/
     export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
     export http_proxy=http://proxytc.vingroup.net:9090/
     export https_proxy=http://proxytc.vingroup.net:9090/

     source /opt/conda/bin/activate

     cd /hoanglh88/utils/
     conda activate igmc/

     cd /hoanglh88/IGMC
     python code/train.py\
          --appdx structure_mask\
          --wandb
     "