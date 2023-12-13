#!/bin/bash
#SBATCH --job-name=sgg_motifs_tde_1 # nom du job
#SBATCH --output=job_logs/sgg_motifs_tde_1%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=job_logs/sgg_motifs_tde_1%j.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=v100-32g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 1 taches (ou processus)
#SBATCH --gres=gpu:2 # reserver 1 GPU
#SBATCH --cpus-per-task=4 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=100:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t4 # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=gtb@v100

module purge # nettoyer les modules herites par defaut
source ~/.bashrc
conda deactivate # desactiver les environnements herites par defaut
conda activate train
module load cuda/11.7.1
export WANDB_MODE=offline

set -x # activer l’echo des commandes
cd $WORK/Scene-Graph-Benchmark-Cuda11.7 # se deplacer dans le dossier de travail

export CUDA_VISIBLE_DEVICES=0,1 # selectionner les GPU a utiliser
srun python -m torch.distributed.launch --master_port 10026 --nproc_per_node=2 tools/relation_train_net.py --use-wandb --config-file "configs/VG178/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 30000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /gpfswork/rech/gtb/ukj95zg/glove MODEL.PRETRAINED_DETECTOR_CKPT /gpfswork/rech/gtb/ukj95zg/Scene-Graph-Benchmark-Cuda11.7/checkpoints/faster_rcnn/best_model_0058000.pth OUTPUT_DIR /gpfswork/rech/gtb/ukj95zg/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG178