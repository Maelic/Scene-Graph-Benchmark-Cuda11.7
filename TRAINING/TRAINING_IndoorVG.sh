CUDA_VISIBLE_DEVICES=0 python tools/detector_pretrain_net.py --config-file "configs/IndoorVG/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" --use-wandb SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.STEPS "(30000, 45000)" SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.RELATION_ON False OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG/final/IndoorVG_faster_rcnn SOLVER.PRE_VAL False

# SGDET training IndoorVG - motifs  TDE, with part-whole
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/IndoorVG/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 35000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 4000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG/with_part_whole/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG/with_part_whole/SGDET-motifs-causal-tde

# PredCls training IndoorVG - motifs TDE, with part-whole
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/IndoorVG/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 35000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 4000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/IndoorVG/with_part_whole/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/IndoorVG/with_part_whole/PREDCLS-motifs-causal-tde

# SGCLs training IndoorVG - motifs TDE, with part-whole
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/IndoorVG/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 35000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 4000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/IndoorVG/with_part_whole/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/IndoorVG/with_part_whole/SGCLS-motifs-causal-tde

# SGDET training IndoorVG - TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --use-wandb --config-file "configs/IndoorVG/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ATTRIBUTE_ON False MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG/final/IndoorVG_faster_rcnn/best_model_0048000.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/motifs-IndoorVGAugmented

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/motif-precls-exmp OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp


# SGDET training VG200 original - TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 24000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/causal-motifs-vg150-tde

##### PREDCLS

# PREDCLS training IndoorVG - TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --use-tensorboard --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_indoor.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 20000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/release3/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/motif-precls-indoor-vg-tde


# Custom prediction - IndoorVG
CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR  /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/causal-motifs-sgdet-exmp OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/causal-motifs-sgdet-exmp TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/demo/images DETECTED_SGG_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/demo/outputs

# Custom prediction - VG200
CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --config-file "configs/VG200/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR  /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/upload_causal_motif_sgdet OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/upload_causal_motif_sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/demo/images DETECTED_SGG_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/demo/outputs/original_sgg


################## YOLOV8


# SGDET training IndoorVG - TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 26000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/Documents/PhD/MyModel/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG/causal-motifs-indoor-vg-tde

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 26000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/Documents/PhD/MyModel/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG/PredCLS/causal-motifs-tde MODEL.WEIGHT /home/maelic/Documents/PhD/MyModel/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG/PredCLS/causal-motifs-tde/best_model_0026000.pth



CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --use-wandb --save-best --config-file "configs/IndoorVG/e2e_relation_faster_rcnn.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 26000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/Documents/PhD/MyModel/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG/FasterRCNN-causal-motifs-tde

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --use-wandb --save-best --config-file "configs/IndoorVG/e2e_relation_faster_rcnn.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 26000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/Documents/PhD/MyModel/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG/FasterRCNN-causal-motifs-tde MODEL.WEIGHT /home/maelic/Documents/PhD/MyModel/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG/FasterRCNN-causal-motifs-tde/best_model_0008000.pth


# transformers

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 16000 SOLVER.STEPS "(10000,16000)" SOLVER.VAL_PERIOD 1000 SOLVER.CHECKPOINT_PERIOD 1000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/Documents/PhD/MyModel/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG/yolov8-transformer