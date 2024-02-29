import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from sgg_benchmark.data.transforms import LetterBox

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import ops
from ultralytics.engine.results import Results

from sgg_benchmark.structures.bounding_box import BoxList

import numpy as np
import cv2

class YoloV8(DetectionModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        yolo_cfg = cfg.MODEL.YOLO.SIZE+'.yaml'
        super().__init__(yolo_cfg, nc=nc, verbose=verbose)
        # self.features_layers = [len(self.model) - 2]
        self.conf_thres = cfg.MODEL.BACKBONE.NMS_THRESH
        self.iou_thres = cfg.MODEL.ROI_HEADS.NMS
        self.device = cfg.MODEL.DEVICE
        self.input_size = cfg.INPUT.MIN_SIZE_TRAIN
        self.nc = nc

    def forward(self, x, profile=False, visualize=False, embed=None):
        y, feature_maps = [], []  # outputs
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            """
            We extract features from the following layers:
            15: 80x80
            18: 40x40
            21: 20x20
            For different object scales, as in original YOLOV8 implementation.
            """
            if embed:
                if i in {15, 18, 21}:  # if current layer is one of the feature extraction layers
                    # feature_maps.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                    feature_maps.append(x)
        if embed:
            return x, feature_maps
        else:
            return x

    def load(self, weights_path: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """

        weights, _ = attempt_load_one_weight(weights_path)

        if weights:
            super().load(weights)

        # args = get_cfg(overrides={'model': weights_path})
        # self.args = args  # attach hyperparameters to model

        # self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
        # self.ckpt_path = self.model.pt_path

        # self.overrides["model"] = weights
        # self.overrides["task"] = self.task

    def prepare_input(self, image, input_shape=(640,640), stride=32, auto=True):
        not_tensor = not isinstance(image, torch.Tensor)
        if not_tensor:
            same_shapes = all(x.shape == im[0].shape for x in image)
            letterbox = LetterBox(input_shape, auto=same_shapes, stride=self.model.stride)(image=image)
            im = np.stack([letterbox(image=x) for x in im])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device).float()
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0

        return im

    def features_extract(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, embeddings = [], []  # outputs
        for m in self.model.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
            if m.i == max(self.features_layers):
                return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    # def custom_forward(self, x):
    #     """
    #     This is a modified version of the original _forward_once() method in BaseModel,
    #     found in ultralytics/nn/tasks.py.
    #     The original method returns only the detection output, while this method returns
    #     both the detection output and the features extracted by the last convolutional layer.
    #     """
    #     y = []
    #     features = None
    #     if torch.is_tensor(x):
    #         x =  self.preprocess(x)
    #         for m in self.model:
    #             if m.f != -1:  # if not from previous layer
    #                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
    #             if torch.is_tensor(x):
    #                 features = x # keep the last tensor as features
    #             x = m(x)  # run
    #             if torch.is_tensor(x):
    #                 features = x # keep the last tensor as features
    #             y.append(x if m.i in self.save else None)  # save output
    #         if torch.is_tensor(x):
    #             features = x # keep the last tensor as features
    #         return features, x # return features and detection output
    #     else:
    #         x = self.preprocess(x)['img']
    #         for m in self.model:
    #             if m.f != -1:  # if not from previous layer
    #                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
    #             if torch.is_tensor(x):
    #                 features = x # keep the last tensor as features
    #             x = m(x)  # run
    #             if torch.is_tensor(x):
    #                 features = x # keep the last tensor as features
    #             y.append(x if m.i in self.save else None)  # save output
    #         if torch.is_tensor(x):
    #             features = x # keep the last tensor as features
    #         return features, x # return features and detection output
    
    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        if not(torch.is_tensor(batch)):
            batch['img'] = batch['img'].to(self.device, non_blocking=True)
            batch['img'] = (batch['img'].half() if self.half else batch['img'].float()) / 255
            for k in ['batch_idx', 'cls', 'bboxes']:
                batch[k] = batch[k].to(self.device)

            nb = len(batch['img'])
            self.lb = [torch.cat([batch['cls'], batch['bboxes']], dim=-1)[batch['batch_idx'] == i]
                    for i in range(nb)] if self.save_hybrid else []  # for autolabelling
        else:
            batch = batch.to(self.device, non_blocking=True)
            batch = (batch.half() if self.half else batch.float()) / 255
            nb = len(batch)

        return batch
    
    def visualize(self, preds, orig_imgs):
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    
        # get model input size
        imgsz = (self.input_size, self.input_size)
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(imgsz, pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
    
    def postprocess(self, preds, img, targets,visualize=True):
        """Post-processes predictions and returns a list of Results objects."""

        preds = ops.non_max_suppression(
            preds,
            nc=self.nc,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=50,
        )

        results = []
        for i, pred in enumerate(preds):
            # orig_img = orig_imgs[i]
            # pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img)

            orig_img_path = targets[i].get_field("image_path")
            image_size = targets[i].size
            print(image_size)
            print(img[i])
            boxlist = BoxList(pred[:, :4], (img[i][1], img[i][0]), mode="xyxy")
            # resize
            boxlist = boxlist.resize(image_size)
            boxlist.add_field("pred_scores", pred[:, 4])
            boxlist.add_field("labels", pred[:, 5])
            boxlist.add_field("pred_labels", pred[:, 5])

            # show boxes on image
            if visualize:
                img = cv2.imread(orig_img_path)

                for box, label in zip(boxlist.bbox, boxlist.get_field("pred_labels")):
                    # round + to list
                    bbox = list(map(int, box))
                    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    img = cv2.putText(img, str(int(label)), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                # save img to path
                cv2.imwrite("/home/maelic/Documents/poubelle/tests"+orig_img_path.split('/')[-1], img)

            assert len(boxlist.get_field("pred_labels")) == len(boxlist.get_field("pred_scores"))
            # boxlist.add_field("pred_logits", pred[:, 5:])

            results.append(boxlist)

        return results

    
    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}
    

def create_yolov8_model(model_name_or_path, nc, class_names):
    from ultralytics.nn.tasks import attempt_load_one_weight
    ckpt = None
    if str(model_name_or_path).endswith('.pt'):
        weights, ckpt = attempt_load_one_weight(model_name_or_path)
        cfg = ckpt['model'].yaml
    else:
        cfg = model_name_or_path
    model = YoloV8(cfg, nc=nc, verbose=True)
    if weights:
        model.load(weights)
    model.nc = nc
    model.names = class_names  # attach class names to model
    model.out_channels = 3

    return model