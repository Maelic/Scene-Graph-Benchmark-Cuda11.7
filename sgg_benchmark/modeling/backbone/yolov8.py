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
from PIL import Image


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
    
    def postprocess(self, preds, img_sizes, targets, visualize=False):
        """Post-processes predictions and returns a list of Results objects."""
        idx_to_label = {1: "bag", 2: "bar", 3: "basket", 4: "bathtub", 5: "bed", 6: "blanket", 7: "blind", 8: "board", 9: "book", 10: "bookshelf", 11: "bottle", 12: "bowl", 13: "box", 14: "brick", 15: "button", 16: "cabinet", 17: "can", 18: "candle", 19: "carpet", 20: "cat", 21: "ceiling", 22: "chair", 23: "child", 24: "clock", 25: "computer", 26: "container", 27: "cord", 28: "couch", 29: "counter", 30: "cup", 31: "curtain", 32: "cushion", 33: "design", 34: "desk", 35: "dog", 36: "door", 37: "drawer", 38: "fan", 39: "faucet", 40: "floor", 41: "flower", 42: "food", 43: "frame", 44: "glass", 45: "hand", 46: "handle", 47: "head", 48: "headboard", 49: "holder", 50: "jacket", 51: "jar", 52: "key", 53: "keyboard", 54: "knob", 55: "lamp", 56: "laptop", 57: "leaf", 58: "leg", 59: "lid", 60: "light", 61: "luggage", 62: "magazine", 63: "microwave", 64: "mirror", 65: "monitor", 66: "mouse", 67: "mug", 68: "outlet", 69: "oven", 70: "painting", 71: "paper", 72: "pen", 73: "person", 74: "phone", 75: "pillow", 76: "plant", 77: "plate", 78: "poster", 79: "pot", 80: "rack", 81: "refrigerator", 82: "remote", 83: "rug", 84: "scissors", 85: "screen", 86: "seat", 87: "sheet", 88: "shelf", 89: "shirt", 90: "shoe", 91: "shower", 92: "sink", 93: "speaker", 94: "stand", 95: "stove", 96: "suitcase", 97: "table", 98: "teddy bear", 99: "television", 100: "tile", 101: "toilet", 102: "toilet paper", 103: "toothbrush", 104: "towel", 105: "toy", 106: "urinal", 107: "vase", 108: "vent", 109: "wall", 110: "window"}
        idx_to_label = {k-1: v for k, v in idx_to_label.items()}

        preds = ops.non_max_suppression(
            preds,
            nc=self.nc,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=50,
        )

        results = []
        for i, pred in enumerate(preds):
            out_img_size = targets[i].size

            boxes = pred[:, :4]

            boxlist = BoxList(boxes, out_img_size, mode="xyxy")
            scores = pred[:, 4]
            labels = pred[:, 5]
            # resize
            boxlist.add_field("pred_scores", scores)
            boxlist.add_field("labels", labels)
            boxlist.add_field("pred_labels", labels)

            # show boxes on image
            if visualize:
                orig_img_path = targets[i].get_field("image_path")
                #boxes = ops.scale_boxes(img_sizes[i], pred[:, :4], (out_img_size[1], out_img_size[0]))

                orig_img = cv2.imread(orig_img_path)
                image_width, image_height = orig_img.shape[1], orig_img.shape[0]
                boxlist = boxlist.resize((image_width, image_height))
                boxes = boxlist.bbox

                # concat score and label to boxes
                scores = scores.unsqueeze(1)
                labels = labels.unsqueeze(1)
                boxes = torch.cat((boxes, scores, labels), 1)

                res = Results(orig_img, path=orig_img_path, names=idx_to_label, boxes=boxes)

                im_array = res.plot()  # plot results

                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                # write down to file
                im.save("/home/maelic/Documents/poubelle/ultr_"+orig_img_path.split('/')[-1])

            assert len(boxlist.get_field("pred_labels")) == len(boxlist.get_field("pred_scores"))
            # boxlist.add_field("pred_logits", pred[:, 5:])

            results.append(boxlist)
        del(preds)
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