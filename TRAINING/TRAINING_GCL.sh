##########################
### VG150-curated

# SGDET - Unbiased-Causal-Motifs-TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --use-wandb --save-best --config-file "configs/VG150/curated/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" --task "sgcls" MODEL.ROI_RELATION_HEAD.PREDICTOR TransLike_GCL GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 8 DTYPE "float16" SOLVER.MAX_ITER 30000 SOLVER.VAL_PERIOD 1000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False SOLVER.BASE_LR 0.001  GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/PREDCLS-SHA-GCL


##########################
### VG150-baseline

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --use-wandb --config-file "configs/VG150/baseline/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" --task "predcls" MODEL.ROI_RELATION_HEAD.PREDICTOR TransLike_GCL GLOBAL_SETTING.BASIC_ENCODER 'Cross-Attention' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 DTYPE "float16" SOLVER.MAX_ITER 60000 SOLVER.VAL_PERIOD 1000 SOLVER.CHECKPOINT_PERIOD 5000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/baseline/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/baseline/PREDCLS-SHA-GCL
