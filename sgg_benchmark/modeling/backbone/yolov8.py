from ultralytics import YOLO
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel

from collections import OrderedDict

class YoloV8(nn.Module):
    def __init__(self, cfg, model_size):
        super(YoloV8, self).__init__()
        self.model_name = "yolov8" + str(model_size[0]) +".yaml"

        self.model = YOLO(self.model_name, task="detect")

    def embedding_concat(self, x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2) #.to(self.device)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z
    
    def forward2(self, frame):
        outputs = []
        test_outputs = OrderedDict([("layer1", []), ("layer2", []), ("layer3", [])])

        model = self.model
        model.to(self.device)

        def hook(module, input, output):
            outputs.append(output)

        # for i in range(len(model.model.model)):
        #     model.model.model[i].register_forward_hook(hook)
        model.model.model[4].register_forward_hook(hook)
        model.model.model[6].register_forward_hook(hook)
        model.model.model[8].register_forward_hook(hook)

        # get results and features
        results = model.predict(frame)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
                # Display the annotated frame
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_names = results[0].names
        classes = [class_names[c] for c in results[0].boxes.cls.cpu().numpy().astype(int)]

        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v)
        outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        embedding_vectors = (
            test_outputs["layer1"], # shape [1, 128, 80, 80]
            test_outputs["layer2"], # shape [1, 256, 40, 40]
            test_outputs["layer3"]  # shape [1, 512, 20, 20]
        )
        # we scale the first layer from 128 to 256
        embedding_vectors = self.embedding_concat(embedding_vectors[0], test_outputs["layer2"])

        # for layer_name in ["layer2", "layer3"]:
        #     embedding_vectors = self.embedding_concat(embedding_vectors, test_outputs[layer_name])

        # convert to cpu
        return boxes, classes, embedding_vectors, annotated_frame.cpu().numpy().astype(int)