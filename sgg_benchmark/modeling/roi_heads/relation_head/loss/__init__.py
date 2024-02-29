from .loss import FocalLoss, InformativeLoss, RelationLossComputation
from .hierarchical_loss import RelationHierarchicalLossComputation

def make_roi_relation_loss_evaluator(cfg):

    if "Hierarchical" in cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR:
        loss_evaluator = RelationHierarchicalLossComputation(
            cfg.MODEL.ATTRIBUTE_ON,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
            cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
            cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        )
    else:
        loss_evaluator = RelationLossComputation(
            cfg.MODEL.ATTRIBUTE_ON,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
            cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
            cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
            cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        )

    return loss_evaluator