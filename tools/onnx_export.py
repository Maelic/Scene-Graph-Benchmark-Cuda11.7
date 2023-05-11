# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import json
import torch
from tqdm import tqdm

from sgg_benchmark.config import cfg
from PIL import Image
from sgg_benchmark.modeling.detector import build_detection_model
import argparse
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.logger import setup_logger
from sgg_benchmark.utils.collect_env import collect_env_info
from sgg_benchmark.structures.image_list import to_image_list

from sgg_benchmark.data.build import build_transforms

def pre_process(image, target, cfg):
    transform = build_transforms(cfg, is_train=False)
    
    return transform(image, target)

def get_sorted_bbox_mapping(score_list):
    sorted_scoreidx = sorted([(s, i) for i, s in enumerate(score_list)], reverse=True)
    sorted2id = [item[1] for item in sorted_scoreidx]
    id2sorted = [item[1] for item in sorted([(j,i) for i, j in enumerate(sorted2id)])]
    return sorted2id, id2sorted

def custom_sgg_post_precessing(predictions):
    output_dict = {}
    for idx, boxlist in zip(predictions):
        xyxy_bbox = boxlist.convert('xyxy').bbox
        # current sgg info
        current_dict = {}
        # sort bbox based on confidence
        sortedid, id2sorted = get_sorted_bbox_mapping(boxlist.get_field('pred_scores').tolist())
        # sorted bbox label and score
        bbox = []
        bbox_labels = []
        bbox_scores = []
        # bbox = xyxy_bbox
        # bbox_labels = boxlist.get_field('pred_labels')
        # bbox_scores = boxlist.get_field('pred_scores')
        for i in sortedid:
            bbox.append(xyxy_bbox[i].tolist())
            bbox_labels.append(boxlist.get_field('pred_labels')[i].item())
            bbox_scores.append(boxlist.get_field('pred_scores')[i].item())
        current_dict['bbox'] = torch.tensor(bbox)
        current_dict['bbox_labels'] = torch.tensor(bbox_labels)
        current_dict['bbox_scores'] = torch.tensor(bbox_scores)
        # sorted relationships
        rel_sortedid, _ = get_sorted_bbox_mapping(boxlist.get_field('pred_rel_scores')[:,1:].max(1)[0].tolist())
        # sorted rel
        rel_pairs = []
        rel_labels = []
        rel_scores = []
        rel_all_scores = []
        # rel_labels = boxlist.get_field('pred_rel_scores')[:, 1:].max(1)[1] + 1
        # rel_scores = boxlist.get_field('pred_rel_scores')[:, 1:].max(1)[0]
        # rel_all_scores = boxlist.get_field('pred_rel_scores')
        # id2sorted = torch.tensor(id2sorted, dtype=torch.int32)
        # rel_pairs = [id2sorted[boxlist.get_field('rel_pair_idxs')[:, 0]], id2sorted[boxlist.get_field('rel_pair_idxs')[:, 1]]]
        for i in rel_sortedid:
            rel_labels.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[1].item() + 1)
            rel_scores.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[0].item())
            rel_all_scores.append(boxlist.get_field('pred_rel_scores')[i].tolist())
            old_pair = boxlist.get_field('rel_pair_idxs')[i].tolist()
            rel_pairs.append([id2sorted[old_pair[0]], id2sorted[old_pair[1]]])
        current_dict['rel_pairs'] = torch.tensor(rel_pairs)
        current_dict['rel_labels'] = torch.tensor(rel_labels)
        current_dict['rel_scores'] = torch.tensor(rel_scores)
        # current_dict['rel_all_scores'] = torch.tensor(rel_all_scores)
        output_dict[idx] = current_dict
    return output_dict

def predict(image, model, device):

    target = torch.LongTensor([-1])
    if isinstance(image, str):
        img = [Image.open(image).convert("RGB")]
    else:
        img = [image]

    device = torch.device(device)
    data_loader = data_loader.dataset

    model.eval()
    cpu_device = torch.device("cpu")
    torch.cuda.empty_cache()
    with torch.no_grad():
        target = [target.to(device)]
        # relation detection needs the targets
        output = model(img.to(device), target)
        output = [o.to(cpu_device) for o in output]

    torch.cuda.empty_cache()

    predictions = custom_sgg_post_precessing(output)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # logger = setup_logger("maskrcnn_benchmark", save_dir="", )
    # logger.info(cfg)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    # image_path = "/home/maelic/Documents/IETrans-SGG.pytorch/visualization/custom_imgs/image.jpg"
    # image = Image.open(image_path).convert("RGB")

    target = torch.LongTensor([-1])
    transform = build_transforms(cfg, is_train=False)
    
    # image, target = transform(image, target)

    device = torch.device(cfg.MODEL.DEVICE)
    image = torch.rand(1, 3, 600, 800)

    img_list = to_image_list(image)

    with torch.no_grad():
        model.eval()

        target = target.to(device)
        img_list = img_list.to(device)
        # relation detection needs the targets
        # output = model(image, target)

        #output = [o.to(cpu_device) for o in output]

        # predictions = custom_sgg_post_precessing(output)
        # Export the model
        torch.onnx.export(model.backbone,               # model being run
                    img_list.tensors,                         # model input (or a tuple for multiple inputs)
                    "motif_model_rwt_backbone.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input_images'],   # the model's input names
                    output_names = ['features'])#, # the model's output names
                    #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    #                 'output' : {0 : 'batch_size'}})
        features = model.backbone(img_list.tensors)

        torch.onnx.export(model.rpn,               # model being run
                    (img_list.tensors, img_list.image_sizes, features),      # model input (or a tuple for multiple inputs)
                    "motif_model_rwt_rpn.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['images', 'image_sizes', 'features'],   # the model's input names
                    output_names = ['output'])#, # the model's output names
                    #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    #                 'output' : {0 : 'batch_size'}})

        print("SUCCESS RPN")
        proposals, proposal_losses = model.rpn(img_list.tensors, img_list.image_sizes, features, target)

        torch.onnx.export(model.roi_heads,               # model being run
                    (features, proposals),                      # model input (or a tuple for multiple inputs)
                    "motif_model_rwt_relations_head.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    # opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=False,  # whether to execute constant folding for optimization
                    input_names = ['features', 'proposals'],   # the model's input names
                    output_names = ['output'])#, # the model's output names
                    #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    #                 'output' : {0 : 'batch_size'}})

        # x, result, detector_losses = model.roi_heads(features, proposals, target)

if __name__ == "__main__":
    main()