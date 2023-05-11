# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import cv2

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
from utils_custom_pred import graph_post_processing

import time

def get_sorted_bbox_mapping(score_list):
    sorted_scoreidx = sorted([(s, i) for i, s in enumerate(score_list)], reverse=True)
    sorted2id = [item[1] for item in sorted_scoreidx]
    id2sorted = [item[1] for item in sorted([(j,i) for i, j in enumerate(sorted2id)])]
    return sorted2id, id2sorted

def custom_sgg_post_precessing(boxlist):
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
    for i in sortedid[:100]:
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
    for i in rel_sortedid[:100]:
        rel_labels.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[1].item() + 1)
        rel_scores.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[0].item())
        rel_all_scores.append(boxlist.get_field('pred_rel_scores')[i].tolist())
        old_pair = boxlist.get_field('rel_pair_idxs')[i].tolist()
        rel_pairs.append([id2sorted[old_pair[0]], id2sorted[old_pair[1]]])
    current_dict['rel_pairs'] = torch.tensor(rel_pairs)
    current_dict['rel_labels'] = torch.tensor(rel_labels)
    current_dict['rel_scores'] = torch.tensor(rel_scores)
    # current_dict['rel_all_scores'] = torch.tensor(rel_all_scores)
    return current_dict

def main():
    indoor_vg_dict = json.load(open('/home/maelic/Documents/PhD/MyModel/PhD_Commonsense_Enrichment/VG_refinement/data_tools/IndoorVG/final/VG-SGG-dicts.json', 'r'))

    ind_to_classes = indoor_vg_dict['idx_to_label']
    ind_to_predicates = indoor_vg_dict['idx_to_predicate']

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

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    webcam = cv2.VideoCapture(0)
    # webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        image = Image.fromarray(frame)
        target = torch.LongTensor([-1])
        transform = build_transforms(cfg, is_train=False)
        
        image, target = transform(image, target)
        image = image[None,:]
        device = torch.device(cfg.MODEL.DEVICE)

        img_list = to_image_list(image)
        average_time = 0
        nb_dtections = 0
        with torch.no_grad():
            model.eval()

            target = target.to(device)
            img_list = img_list.to(device)

            time_start = time.time()
            features = model.backbone(img_list.tensors)
            print("Time inference backbone: ", time.time()-time_start)

            time_start = time.time()
            proposals, proposal_losses = model.rpn(img_list, features, target)
            print("Time inference RPN: ", time.time()-time_start)

            time_start = time.time()
            x, output, detector_losses = model.roi_heads(features, proposals, target, logger=None)
            print("Time inference relations: ", time.time()-time_start)
            average_time += time.time()-time_start
            nb_dtections += 1
            #output = model(img_list, target)
        time_start = time.time()

        sgg_detected = custom_sgg_post_precessing(output[0])
        print("Time post-process: ", time.time()-time_start)

        img_graph, img_boxes = graph_post_processing(sgg_detected, frame, ind_to_classes, ind_to_predicates, box_topk=17, rel_thres=0.0)

        cv2.imshow("Graphs", img_graph)
        cv2.imshow("Detected regions", img_boxes)
        print("Average time: ", average_time/nb_dtections)

        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    main()