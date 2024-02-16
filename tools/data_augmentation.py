import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from sgg_benchmark.config import cfg
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.parser import default_argument_parser
from sgg_benchmark.utils.comm import synchronize, get_rank, is_main_process, get_world_size, all_gather
from sgg_benchmark.utils.timer import Timer, get_time_str

from sgg_benchmark.utils.logger import setup_logger, logger_step
from sgg_benchmark.utils.collect_env import collect_env_info
from sgg_benchmark.data import make_data_loader

from tqdm import tqdm

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmcv.parallel import DataContainer as DC
from mmdet.core import encode_mask_results
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector
from openpsg.datasets import build_dataset


def enable_inplace_relu(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU):
            setattr(model, name, torch.nn.ReLU(inplace=True))
        else:
            enable_inplace_relu(module)

def process(config_file, logger, args):

    args = default_argument_parser()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR

    logger = setup_logger("sgg_benchmark", output_dir, get_rank(), filename="log.txt", steps=True, verbose=args.verbose)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.debug(args)

    logger_step(logger, "Collecting environment info...")
    logger.debug("\n" + collect_env_info())

    logger.info("Running with config:\n{}".format(cfg))

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    enable_inplace_relu(model)

    # Initialize mixed-precision if necessary
    use_amp = True if cfg.DTYPE == "float16" or args.amp else False

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    data_loader = make_data_loader(cfg=cfg, mode="test", is_distributed=distributed, dataset_to_test='train')[0]

    device = torch.device(device)
    num_devices = get_world_size()

    dataset = data_loader.dataset

    model.eval()

    ############
    # for loop #
    ############

    gt_rels_count = 0
    internal_trans_count = 0
    external_trans_count = 0

    timer = Timer()

    for _, batch in enumerate(tqdm(data_loader)):

        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]
            if timer:
                timer.tic()
            else:
                # relation detection needs the targets
                output = model(images.to(device), targets)
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        if synchronize_gather:
            synchronize()
            multi_gpu_predictions = all_gather({img_id: result for img_id, result in zip(image_ids, output)})
            if is_main_process():
                for p in multi_gpu_predictions:
                    results_dict.update(p)
        else:
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
            detected_sgg = custom_sgg_post_precessing(results_dict)
            clean_graph = generate_detect_sg(detected_sgg, vg_dict)



            

        img_metas = data['img_metas'][0].data[0][0]
        key = img_metas['ori_filename']
        this_psg_data = psg_data_dict[key]
        if this_psg_data['image_id'] in psg_data['test_image_ids']:
            psg_data_list.append(this_psg_data)
            continue

        with torch.no_grad():
            device = data['gt_labels'][0].data[0][0].device
            gt_masks = data['gt_masks'][0].data[0][0].to_tensor(
                torch.uint8, device)
            data['gt_masks'] = [DC([[gt_masks]])]
            result = model(return_loss=False, rescale=True, **data)
        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                        for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))
        
        # get gt
        ori_shape = img_metas['ori_shape']
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
    