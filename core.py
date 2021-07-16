import torch
import numpy as np


class Core:
    def __init__(self, device):
        self.num_points = 1024
        self.device = device
        if device != 'cpu':
            raise NotImplementedError('Only CPU device supported')
        self.model = torch.jit.load('./weights/model_cpu.pt', map_location=device)
        self.model.eval()

        self.label2num = {
            'cone': 0,
            'cube': 1,
            'cylinder': 2,
            'plane': 3,
            'torus': 4,
            'uv_sphere': 5,
        }

        self.num2label = {v: k for k, v in self.label2num.items()}


    def predict(self, pointcloud):
        if not isinstance(pointcloud, np.ndarray):
            raise ValueError('Input must be numpy array')

        with torch.no_grad():
            pad_to = self.num_points - pointcloud.shape[0]
            pointcloud = np.pad(pointcloud, [(0, pad_to), (0, 0)], mode='constant')

            pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
            pointcloud = pointcloud.to(self.device)
            pointcloud = pointcloud.unsqueeze(0)
            pointcloud = pointcloud.permute(0, 2, 1)

            res = self.model(pointcloud)
            res = torch.nn.functional.softmax(res)

        return res
