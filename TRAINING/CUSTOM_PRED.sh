# VG150 baseline

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --config-file "configs/VG150/baseline/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16"  GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/basline/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/baseline/upload_causal_motif_sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/demo/final_comparison/images DETECTED_SGG_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/demo/final_comparison/baseline


# VG150 curated

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --config-file "configs/VG150/curated/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16"  GLOVE_DIR /home/maelic/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/checkpoints/VG150/curated/sgdet_motifs_causal_tde TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/demo/final_comparison/images DETECTED_SGG_DIR /home/maelic/Documents/Scene-Graph-Benchmark-Cuda11.7/demo/final_comparison/curated

# GPS-Net
CUDA_VISIBLE_DEVICES=0 python tools/webcam.py --config-file "configs/IndoorVG/demo.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR GPSNetPredictor \
    GLOVE_DIR /home/maelic/glove \
    MODEL.WEIGHT /home/maelic/Documents/PhD/MyModel/Real_Time_SGG/GPS-Net/model_0002000.pth \
    MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/PhD/MyModel/Real_Time_SGG/Faster_RCNN/best_model_0048000.pth \
    TEST.IMS_PER_BATCH 1 \
    TEST.CUSTUM_EVAL True \
    TEST.CUSTUM_PATH /home/maelic/Documents/PhD/MyModel/Scene_Graphs_Visualization/custom_imgs_detection/outputs

# Motifs
CUDA_VISIBLE_DEVICES=0 python tools/webcam.py --config-file "configs/IndoorVG/demo.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    GLOVE_DIR /home/maelic/glove \
    MODEL.WEIGHT /home/maelic/Documents/PhD/MyModel/Real_Time_SGG/Motifs/model_0012000.pth \
    MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/PhD/MyModel/Real_Time_SGG/Faster_RCNN/best_model_0048000.pth \
    TEST.IMS_PER_BATCH 1 \
    TEST.CUSTUM_EVAL True \
    TEST.CUSTUM_PATH /home/maelic/Documents/PhD/MyModel/Scene_Graphs_Visualization/custom_imgs_detection/outputs

# Transformer
CUDA_VISIBLE_DEVICES=0 python tools/webcam.py --config-file "configs/IndoorVG/demo.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
    GLOVE_DIR /home/maelic/glove \
    MODEL.WEIGHT /home/maelic/Documents/PhD/MyModel/Real_Time_SGG/Transformer/model_0002000.pth \
    MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/PhD/MyModel/Real_Time_SGG/Faster_RCNN/best_model_0048000.pth \
    TEST.IMS_PER_BATCH 1 \
    TEST.CUSTUM_EVAL True \
    TEST.CUSTUM_PATH /home/maelic/Documents/PhD/MyModel/Scene_Graphs_Visualization/custom_imgs_detection/outputs

# export onnx

# GPS-Net
CUDA_VISIBLE_DEVICES=0 python tools/onnx_export.py --config-file "configs/IndoorVG/demo.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR GPSNetPredictor \
    GLOVE_DIR /home/maelic/glove \
    MODEL.WEIGHT /home/maelic/Documents/PhD/MyModel/Real_Time_SGG/GPS-Net/model_0002000.pth \
    MODEL.PRETRAINED_DETECTOR_CKPT /home/maelic/Documents/PhD/MyModel/Real_Time_SGG/Faster_RCNN/best_model_0048000.pth \
    TEST.IMS_PER_BATCH 1 
