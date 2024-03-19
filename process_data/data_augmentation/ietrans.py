import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import os
import time
import datetime
import numpy as np

import torch
import torch.distributed as dist

from sgg_benchmark.config import cfg
from sgg_benchmark.config.defaults_GCL import _C as cfg_GCL
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.solver import make_lr_scheduler
from sgg_benchmark.solver import make_optimizer
from sgg_benchmark.engine.trainer import reduce_loss_dict
from sgg_benchmark.engine.inference import inference
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.checkpoint import clip_grad_norm
from sgg_benchmark.utils.collect_env import collect_env_info
from sgg_benchmark.utils.comm import synchronize, get_rank, all_gather
from sgg_benchmark.utils.logger import setup_logger, logger_step
from sgg_benchmark.utils.miscellaneous import mkdir, save_config
from sgg_benchmark.utils.metric_logger import MetricLogger
from sgg_benchmark.utils.parser import default_argument_parser


def process(config_file, checkpoint_file, output_file=None):

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    logger = setup_logger("sgg_benchmark", verbose=True, steps=True)

    logger_step(logger, 'Building model...')
    model = build_detection_model(cfg) 

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    last_check = checkpointer.get_checkpoint_file()
    logger.info("Loading best checkpoint from {}...".format(last_check))
    _ = checkpointer.load(last_check)

    iou_types = ("bbox","relations", )


    dataset_names = cfg.DATASETS.TEST

    # This variable enables the script to run the test on any dataset split.
    if cfg.DATASETS.TO_TEST:
        assert cfg.DATASETS.TO_TEST in {'train', 'val', 'test', None}
        if cfg.DATASETS.TO_TEST == 'train':
            dataset_names = cfg.DATASETS.TRAIN
        elif cfg.DATASETS.TO_TEST == 'val':
            dataset_names = cfg.DATASETS.VAL

    output_folders = [None] * len(cfg.DATASETS.TEST)

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loader_train = make_data_loader(cfg=cfg, mode="train", is_distributed=distributed)

    model.eval()

    ############
    # for loop #
    ############
    gt_rels_count = 0
    internal_trans_count = 0
    external_trans_count = 0
    pbar = tqdm.tqdm(total=cfg.SOLVER.VAL_PERIOD)

    device = torch.device(cfg.MODEL.DEVICE)

    for _, batch in enumerate(tqdm(data_loader_train)):
        pbar.update(1)

        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]

            output = model(images.to(device), targets)

        # get gt
        ori_shape = targets
        gt_labels = data['gt_labels'][0].data[0][0]
        gt_bboxes = data['gt_bboxes'][0].data[0][0]
        gt_masks = data['gt_masks'][0].data[0][0]
        gt_masks = F.interpolate(gt_masks.unsqueeze(1),
                                    size=ori_shape[:2]).squeeze(1)
        gt_rels = data['gt_rels'][0].data[0][0]
        
        # get gt_no_rels
        sub_obj_pair_list = [[s, o] for s, o, r in gt_rels]
        gt_no_rels = []
        for s in range(len(gt_labels)):  # subject
            for o in range(len(gt_labels)):  # object
                if s == o:
                    continue
                if [s, o] not in sub_obj_pair_list:
                    gt_no_rels.append([s, o, 0])
        gt_no_rels = np.array(gt_no_rels)

        # get pred
        pd_labels = result[0].labels - 1
        pd_bboxes = result[0].refine_bboxes
        pd_masks = result[0].masks
        pd_rel_scores = result[0].rel_scores
        pd_rel_labels = result[0].rel_labels
        pd_rel_dists = result[0].rel_dists
        pd_rels = result[0].rels

        

        # get mask iou
        gt_masks = gt_masks.to(device='cuda:0').to(torch.float).flatten(1)
        pd_masks = torch.asarray(pd_masks, dtype=gt_masks.dtype, device=gt_masks.device).flatten(1)
        ious_list = []
        for mask_i in range(gt_masks.shape[0]):
            ious = gt_masks[mask_i:mask_i+1].mm(pd_masks.transpose(0, 1)) / ((gt_masks[mask_i:mask_i+1] + pd_masks) > 0).sum(-1)
            ious_list.append(ious)
        ious = torch.cat(ious_list, dim=0)

        # print('\n')
        # print(this_psg_data['relations'])
        ##################
        # internal trans #
        ##################
        gt_rels_count += gt_rels.shape[0]
        for i in range(gt_rels.shape[0]):
            gt_s_idx = gt_rels[i][0]
            gt_o_idx = gt_rels[i][1]
            gt_r_label = gt_rels[i][2]
            gt_s_label = gt_labels[gt_s_idx]
            gt_o_label = gt_labels[gt_o_idx]
            pd_r_dists_list = []
            for j in range(pd_rels.shape[0]):
                pd_s_idx = pd_rels[j][0]
                pd_o_idx = pd_rels[j][1]
                pd_r_label = pd_rels[j][2]
                pd_s_label = pd_labels[pd_s_idx]
                pd_o_label = pd_labels[pd_o_idx]
                pd_r_dists = pd_rel_dists[j]
                s_iou = ious[gt_s_idx, pd_s_idx]
                o_iou = ious[gt_o_idx, pd_o_idx]
                if gt_s_label == pd_s_label and gt_o_label == pd_o_label and \
                    s_iou > 0.5 and o_iou > 0.5:
                    pd_r_dists_list.append(pd_r_dists)
            if len(pd_r_dists_list) > 0:
                pd_r_dists = np.stack(pd_r_dists_list, axis=0)
                # 此处应该加权平均
                pd_r_dists = pd_r_dists.mean(axis=0)
                if gt_r_label != np.argmax(pd_r_dists):
                    r_sort = np.argsort(pd_r_dists)[::-1]
                    gt_r_idx = np.where(r_sort == gt_r_label.item())[0].item()
                    confusion_r_labels = r_sort[:gt_r_idx]
                    ori_attr = dataset.so_rel2freq[(gt_s_label.item(), gt_o_label.item(), gt_r_label.item())] / dataset.rel2freq[gt_r_label.item()]
                    for c_r_label in confusion_r_labels:
                        if c_r_label != 0:
                            new_attr = dataset.so_rel2freq[(gt_s_label.item(), gt_o_label.item(), c_r_label)] / dataset.rel2freq[c_r_label]
                            if new_attr < ori_attr:
                                this_psg_data['relations'].append([gt_s_idx.item(), gt_o_idx.item(), int(c_r_label - 1)])
                                internal_trans_count += 1
        # print(this_psg_data['relations'])
        # print(internal_trans_count)

        ##################
        # external trans #
        ##################
        for i in range(gt_no_rels.shape[0]):
            gt_s_idx = gt_no_rels[i][0]
            gt_o_idx = gt_no_rels[i][1]
            gt_r_label = gt_no_rels[i][2]
            gt_s_label = gt_labels[gt_s_idx]
            gt_o_label = gt_labels[gt_o_idx]
            pd_r_dists_list = []
            for j in range(pd_rels.shape[0]):
                pd_s_idx = pd_rels[j][0]
                pd_o_idx = pd_rels[j][1]
                pd_r_label = pd_rels[j][2]
                pd_s_label = pd_labels[pd_s_idx]
                pd_o_label = pd_labels[pd_o_idx]
                pd_r_dists = pd_rel_dists[j]
                s_iou = ious[gt_s_idx, pd_s_idx]
                o_iou = ious[gt_o_idx, pd_o_idx]
                if gt_s_label == pd_s_label and gt_o_label == pd_o_label and \
                    s_iou > 0.5 and o_iou > 0.5:
                    pd_r_dists_list.append(pd_r_dists)
            if len(pd_r_dists_list) > 0:
                pd_r_dists = np.stack(pd_r_dists_list, axis=0)
                # TODO: use weighted average
                pd_r_dists = pd_r_dists.mean(axis=0)
                if gt_r_label != np.argmax(pd_r_dists):
                    r_sort = np.argsort(pd_r_dists)[::-1]
                    gt_r_idx = np.where(r_sort == gt_r_label.item())[0].item()
                    confusion_r_labels = r_sort[:gt_r_idx]
                    for c_r_label in confusion_r_labels:
                        if c_r_label != 0:
                            new_attr = dataset.so_rel2freq[(gt_s_label.item(), gt_o_label.item(), c_r_label)]
                            if new_attr > 0 and c_r_label > 6:
                                this_psg_data['relations'].append([gt_s_idx.item(), gt_o_idx.item(), int(c_r_label - 1)])
                                external_trans_count += 1
        # print(this_psg_data['relations'])
        # print(external_trans_count)

        psg_data_list.append(this_psg_data)

    print('gt_rels_count: ', gt_rels_count)
    print('internal_trans_count: ', internal_trans_count)
    print('external_trans_count: ', external_trans_count)
    if output_file is not None:
        psg_data['data'] = psg_data_list
        fo = open(output_file, 'w')
        json.dump(psg_data, fo)

if __name__ == '__main__':
    # Modify self.data in test mode in openpsg/datasets/psg.py to all data, not just test.
    config_file = sys.argv[1]
    checkpoint_file = sys.argv[2]
    output_file = sys.argv[3]
    process(config_file, checkpoint_file, output_file)
    