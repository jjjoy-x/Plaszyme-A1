#!/bin/bash
#SBATCH --job-name=plaszymer        # 作业名
#SBATCH --partition=gpu3090         # gpu3090队列
#SBATCH --qos=1gpu                  # gpu3090Qos
#SBATCH --nodes=1                   # 节点数量
#SBATCH --ntasks-per-node=1         # 每节点进程数
#SBATCH --cpus-per-task=4           # 1:4 的 GPU:CPU 配比 
#SBATCH --gres=gpu:1                # 2 块 GPU
#SBATCH --output=logs/%j.out        # 标准输出
#SBATCH --error=logs/%j.err         # 错误输出
#SBATCH --mail-user=yueqing.xing22@student.xjtlu.edu.cn
#SBATCH --mail-type=ALL

python test.py \
  --test_csv /gpfs/work/bio/yueqingxing22/igem/Plaszymer_ml/data/raw/PlaszymeDB_v1_sp_test_by_ids.csv \
  --model_dir /gpfs/work/bio/yueqingxing22/igem/Plaszymer_ml/model \
  --out_dir  /gpfs/work/bio/yueqingxing22/igem/Plaszymer_ml/results