CUDA_VISIBLE_DEVICES=0 python tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.STEPS "(30000, 45000)" SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.RELATION_ON False OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/release3 SOLVER.PRE_VAL False

# SGDET training VG150 -  Unbiased-Causal-TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 30000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/causal-motifs-sgdet-vg200

# PredCls training VG150 -  Unbiased-Causal-TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --use-tensorboard --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 20000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150-causal-motifs-predcls

# PredCls training VG150 - VCTree
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --use-tensorboard --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150-vctree-predcls

# SGDET training VG150 - VCTree TDE Sum
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 24 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150-vctree-sgdet

# SGDET training VG150 original - Motifs TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 24000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/causal-motifs-vg150-tde

# Custom prediction - VG150
CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --config-file "configs/VG200/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR  /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/upload_causal_motif_sgdet OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/upload_causal_motif_sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/demo/images DETECTED_SGG_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/demo/outputs/original_sgg


##########################
### VG150-curated

CUDA_VISIBLE_DEVICES=0 python tools/detector_pretrain_net.py --config-file "configs/VG150/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.STEPS "(30000, 45000)" SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 4000 MODEL.RELATION_ON False OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/pretrained_faster_rcnn SOLVER.PRE_VAL False


# SGDET - Unbiased-Causal-Motifs-TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 30000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/sgdet_motifs_causal_tde

# PredCLS - Unbiased-Causal-Motifs-TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 30000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/predcls_motifs_causal_tde

# SGCls - Unbiased-Causal-Motifs-TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 30000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/sgcls_motifs_causal_tde

# PredCLS - Unbiased-Causal-Vctree-TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree  SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 30000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/predcls_vctree_tde

# SGCls - Unbiased-Causal-Vctree-TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree  SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 30000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/sgcls_vctree_tde

# SGDet - Unbiased-Causal-Vctree-TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree  SOLVER.IMS_PER_BATCH 24 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 30000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/sgdet_vctree_tde