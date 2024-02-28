# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from sgg_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

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

def train_one_epoch(model, optimizer, data_loader, device, epoch, max_norm, logger, cfg, scaler):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")

    header = 'Epoch: [{}]'.format(epoch)

    for i, (images, targets, _) in enumerate(metric_logger.log_every(data_loader, cfg.SOLVER.PRINT_FREQ, header)):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=cfg.DTYPE == "float16"):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def train(cfg, logger, args):
    available_metrics = {"mR": "_mean_recall", "R": "_recall", "zR": "_zeroshot_recall", "ng-zR": "_ng_zeroshot_recall", "ng-R": "_recall_nogc", "ng-mR": "_ng_mean_recall", "topA": ["_accuracy_hit", "_accuracy_count"]}

    best_epoch = 0
    best_metric = 0.0
    best_checkpoint = None

    metric_to_track = available_metrics[cfg.METRIC_TO_TRACK]

    logger_step(logger, 'Building model...')
    model = build_detection_model(cfg) 

    # get run name for logger
    if args['use_wandb']:
        import wandb
        run_name = cfg.OUTPUT_DIR.split('/')[-1]
        if args['distributed']:
            wandb.init(project="scene-graph-benchmark", entity="maelic", group="DDP", name=run_name, config=cfg)
        wandb.init(project="scene-graph-benchmark", entity="maelic", name=run_name, config=cfg)

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    if cfg.MODEL.BOX_HEAD:
        eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
    else:
        eval_modules = (model.backbone,)
 
    fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}
    
    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    logger_step(logger, 'Building optimizer and shcedule')

    # Initialize mixed-precision training
    use_amp = True if cfg.DTYPE == "float16" else False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args['local_rank']], output_device=args['local_rank'],
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(None, 
                                       update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
    else:
        if "FPN" in cfg.MODEL.BACKBONE.TYPE:
            # load_mapping is only used when we init current model from detection model.
            checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
        else:
            model.backbone.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT)
            model.backbone.model.to(device)
        # load backbone weights
        logger_step(logger, 'Loading Backbone weights from '+cfg.MODEL.PRETRAINED_DETECTOR_CKPT)
    
    mode = get_mode(cfg)

    # if mode == "predcls":
    #     model.backbone.model.eval()

    logger_step(logger, 'Building checkpointer')

    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=args['distributed'],
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=args['distributed'],
    )
    logger_step(logger, 'Building dataloader')

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, args['distributed'], logger, device=device)

    meters = MetricLogger(delimiter="  ")

    logger.info("Start training")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    print_first_grad = True
    pbar = tqdm.tqdm(total=cfg.SOLVER.VAL_PERIOD)

    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        pbar.update(1)
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.roi_heads.train()
        model.backbone.eval()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        # Note: If mixed precision is not used, this ends up doing nothing
        # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        meters.update(loss=losses_reduced, **loss_dict_reduced)
        if args['use_wandb']:
            wandb.log({"loss": losses_reduced}, step=iteration)

        optimizer.zero_grad()
        
        # Scaling loss
        scaler.scale(losses).backward()
        
        # Unscale the gradients of optimizer's assigned params in-place before cliping
        # from https://pytorch.org/docs/stable/notes/amp_examples.html
        scaler.unscale_(optimizer)

        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
        print_first_grad = False
        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

        scaler.step(optimizer)
        scaler.update()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 100 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if not args['save_best'] and iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        val_result = None # used for scheduler updating
        current_metric = None
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            # reset pbar
            pbar.close()
            pbar = tqdm.tqdm(total=cfg.SOLVER.VAL_PERIOD)
            logger.info("Start validating")
            val_result = run_val(cfg, model, val_data_loaders, args['distributed'], logger)
            if mode+metric_to_track not in val_result.keys():
                logger.error("Metric to track not found in validation result, default to R")
                metric_to_track = "_recall"
            results = val_result[mode+metric_to_track]
            current_metric = float(np.mean(list(results.values())))
            logger.info("Average validation Result for %s: %.4f" % (cfg.METRIC_TO_TRACK, current_metric))
            
            if current_metric > best_metric:
                best_epoch = iteration
                best_metric = current_metric
                if args['save_best']:
                    to_remove = best_checkpoint
                    checkpointer.save("best_model_{:07d}".format(iteration), **arguments)
                    best_checkpoint = os.path.join(cfg.OUTPUT_DIR, "best_model_{:07d}".format(iteration))

                    # We delete last checkpoint only after succesfuly writing a new one, in case of out of memory
                    if to_remove is not None:
                        os.remove(to_remove+".pth")
                        logger.info("New best model saved at iteration {}".format(iteration))
                
            logger.info("Now best epoch in {} is : {}, with value is {}".format(cfg.METRIC_TO_TRACK+"@k", best_epoch, best_metric))
            
            if args['use_wandb']:
                wandb.log({cfg.METRIC_TO_TRACK+"@k": val_result}, step=iteration)            
 
        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            # Using mean recall instead of traditionnal recall for scheduler
            scheduler.step(current_metric, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                break
        else:
            scheduler.step()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    name = "model_{:07d}".format(best_epoch)
    last_filename = os.path.join(cfg.OUTPUT_DIR, "{}.pth".format(name))
    output_folder = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
    with open(output_folder, "w") as f:
        f.write(last_filename)
    print('\n\n')
    logger.info("Best Epoch is : %.4f" % best_epoch)

    return model, last_filename 

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        # module.model.eval()
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def run_val(cfg, model, val_data_loaders, distributed, logger, device=None):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        # shrink data_loader to only 100 samples
        dataset_result = inference(
                            cfg,
                            model,
                            val_data_loader,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=None,
                            logger=logger,
                        )
        synchronize()

        val_result.append(dataset_result)

    # VG has only one val dataset
    dataset_result = val_result[0]
    if len(dataset_result) == 1:
        return dataset_result
    if distributed:
        for k1, v1 in dataset_result.items():
            for k2, v2 in v1.items():
                dataset_result[k1][k2] = torch.distributed.all_reduce(torch.tensor(np.mean(v2)).to(device).unsqueeze(0)).item() / torch.distributed.get_world_size()
    else:
        for k1, v1 in dataset_result.items():
            for k2, v2 in v1.items():
                if isinstance(v2, list):
                    # mean everything
                    v2 = [np.mean(v) for v in v2]
                dataset_result[k1][k2] = np.mean(v2)

    return dataset_result

def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()

def get_mode(cfg):
    task = "sgdet"
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        task = "sgcls"
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == True:
            task = "predcls"
    return task

def assert_mode(cfg, task):
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
    if task == "sgcls" or task == "predcls":
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = True
    if task == "predcls":
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = True

def main():
    args = default_argument_parser()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    for arg in args.opts:
        if "GCL_SETTING" in arg:
            cfg.set_new_allowed(True) # recursively update set_new_allowed to allow merging of configs and subconfigs
            cfg.merge_from_other_cfg(cfg_GCL)
            break
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.task:
        assert_mode(cfg, args.task)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("sgg_benchmark", output_dir, get_rank(), verbose=args.verbose, steps=True)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.debug(args)

    logger.info("Collecting environment info...")
    logger_step(logger, "\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.debug(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    training_args = {"task:": args.task, 
        "save_best": args.save_best, 
        "use_wandb": args.use_wandb, 
        "skip_test": args.skip_test, 
        "local_rank": args.local_rank, 
        "distributed": args.distributed
    }

    model, best_checkpoint = train(
        cfg=cfg,
        logger=logger,
        args=training_args
    )

    if not args.skip_test:
        if best_checkpoint is not None:
            logger.info("Loading best checkpoint from {}...".format(best_checkpoint))
            checkpointer = DetectronCheckpointer(cfg, model)
            _ = checkpointer.load(best_checkpoint)
            logger_step(logger, "Starting test with best checkpoint...")
            run_test(cfg, model, args.distributed, logger)
        else:
            logger_step(logger, "Starting test with last checkpoint...")
            run_test(cfg, model, args.distributed, logger)
    
    logger.info("#"*20+" END TRAINING "+"#"*20)


if __name__ == "__main__":
    main()