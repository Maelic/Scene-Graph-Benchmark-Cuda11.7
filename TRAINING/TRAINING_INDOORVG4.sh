### SGDET - INDOORVG4 - YOLOV8m

# Causal Motifs TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.VAL_PERIOD 1000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/Documents/PhD/MyModel/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG4/SGDET/causal-motifs-yolov8m

# Transformer
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --task sgdet --save-best --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.VAL_PERIOD 1000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/Documents/PhD/MyModel/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG4/SGDET/transformer-yolov8m

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --task sgdet --config-file "/home/maelic/ros2_humble/src/Robots-Scene-Understanding/rsu_scene_graph_generation/models/transformer" GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/ros2_humble/src/Robots-Scene-Understanding/rsu_scene_graph_generation/models/transformer

# Causal Vctree TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.VAL_PERIOD 1000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/Documents/PhD/MyModel/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG4/SGDET/causal-vctree-yolov8m

# GPS-Net
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR GPSNetPredictor SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/Documents/PhD/MyModel/Scene-Graph-Benchmark-Cuda11.7/checkpoints/IndoorVG4/SGDET/gpsnet-yolov8l