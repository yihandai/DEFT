import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()


# Custom layers with IR methods for FG-CAM
def forward_hook(self, input, output):
    self.X = input[0].detach()
    self.X.requires_grad = True


def divide_with_zero(a, b):
    b_nozero = torch.where(b == 0, torch.ones_like(b), b)
    c = a / b_nozero
    return torch.where(b == 0, torch.zeros_like(b), c)


class ReLU(nn.ReLU):
    def IR(self, I):
        return I


class MaxPool2d(nn.MaxPool2d):
    def IR(self, I):
        X = torch.clamp(self.X, min=0)
        Y = self.forward(X)
        I = divide_with_zero(I, Y)
        I = torch.autograd.grad(Y, X, I, retain_graph=True)[0] * X
        return I


class AvgPool2d(nn.AvgPool2d):
    def IR(self, I):
        X = torch.clamp(self.X, min=0)
        Y = self.forward(X)
        I = divide_with_zero(I, Y)
        I = torch.autograd.grad(Y, X, I, retain_graph=True)[0] * X
        return I


class BatchNorm2d(nn.BatchNorm2d):
    def IR(self, I):
        return I


class Conv2d(nn.Conv2d):
    def IR(self, I):
        X = self.X
        positive_weight = torch.clamp(self.weight, min=0)
        nagative_weight = torch.clamp(self.weight, max=0)
        positive_input = torch.clamp(X, min=0)
        nagative_input = torch.clamp(X, max=0)
        if X.shape[1] == 3:
            B = (
                X * 0
                + torch.min(
                    torch.min(
                        torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True
                    )[0],
                    dim=3,
                    keepdim=True,
                )[0]
            )
            H = (
                X * 0
                + torch.max(
                    torch.max(
                        torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True
                    )[0],
                    dim=3,
                    keepdim=True,
                )[0]
            )

            Y1 = torch.conv2d(
                X, self.weight, bias=None, stride=self.stride, padding=self.padding
            )
            Y2 = torch.conv2d(
                B, positive_weight, bias=None, stride=self.stride, padding=self.padding
            )
            Y3 = torch.conv2d(
                H, nagative_weight, bias=None, stride=self.stride, padding=self.padding
            )
            I = divide_with_zero(I, Y1 - Y2 - Y3)
            I = (
                X * torch.autograd.grad(Y1, X, I, retain_graph=True)[0]
                - B * torch.autograd.grad(Y2, B, I, retain_graph=True)[0]
                - H * torch.autograd.grad(Y3, H, I, retain_graph=True)[0]
            )
        else:
            Y1 = F.conv2d(
                positive_input,
                positive_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
            )
            Y2 = F.conv2d(
                nagative_input,
                nagative_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
            )
            I = divide_with_zero(I, Y1 + Y2)
            I = (
                positive_input
                * torch.autograd.grad(Y1, positive_input, I, retain_graph=True)[0]
                + nagative_input
                * torch.autograd.grad(Y2, nagative_input, I, retain_graph=True)[0]
            )
        return I


class Linear(nn.Linear):
    def IR(self, I):
        X = self.X
        positive_weight = torch.clamp(self.weight, min=0)
        nagative_weight = torch.clamp(self.weight, max=0)
        positive_input = torch.clamp(X, min=0)
        nagative_input = torch.clamp(X, max=0)
        Y1 = F.linear(positive_input, positive_weight, bias=None)
        Y2 = F.linear(nagative_input, nagative_weight, bias=None)
        I = divide_with_zero(I, Y1 + Y2)
        I = (
            positive_input
            * torch.autograd.grad(Y1, positive_input, I, retain_graph=True)[0]
            + nagative_input
            * torch.autograd.grad(Y2, nagative_input, I, retain_graph=True)[0]
        )
        return I


class Pad(nn.Module):
    def __init__(self, pad, value=0):
        super(Pad, self).__init__()
        self.pad = pad
        self.value = value

    def forward(self, x):
        return F.pad(x, self.pad, value=self.value)

    def IR(self, I):
        return I


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
        self.X1 = None
        self.X2 = None
        self.output = None

    def forward(self, x1, x2):
        # Store inputs for IR propagation
        if x1.requires_grad or x2.requires_grad:
            self.X1 = x1
            self.X2 = x2
        else:
            self.X1 = x1.detach()
            self.X1.requires_grad = True
            self.X2 = x2.detach()
            self.X2.requires_grad = True
        self.output = self.X1 + self.X2
        return self.output

    def IR(self, I):
        # For addition, distribute I to both inputs
        # The gradient of sum w.r.t. each input is 1, so we multiply by the input
        if self.X1 is not None and self.X2 is not None:
            I1 = torch.autograd.grad(
                self.output, self.X1, I, retain_graph=True, allow_unused=True
            )[0]
            I2 = torch.autograd.grad(
                self.output, self.X2, I, retain_graph=True, allow_unused=True
            )[0]
            if I1 is not None:
                I1 = I1 * self.X1
            else:
                I1 = torch.zeros_like(self.X1)
            if I2 is not None:
                I2 = I2 * self.X2
            else:
                I2 = torch.zeros_like(self.X2)
            # Return the sum (both paths contribute)
            return I1 + I2
        return I


def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding="bytes").item()

    return weights_dict


class CNN(nn.Module):

    def __init__(self, weight_file):
        super(CNN, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)
        # Track layers for IR propagation (similar to VGG's self.features)
        self.features = []
        self.hook_handles = []

        self.conv1 = self.__conv(
            2,
            name="conv1",
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            groups=1,
            bias=False,
        )
        self.bn_conv1 = self.__batch_normalization(
            2, "bn_conv1", num_features=64, eps=9.999999747378752e-06, momentum=0.0
        )
        self.res2a_branch1 = self.__conv(
            2,
            name="res2a_branch1",
            in_channels=64,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.res2a_branch2a = self.__conv(
            2,
            name="res2a_branch2a",
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn2a_branch1 = self.__batch_normalization(
            2, "bn2a_branch1", num_features=256, eps=9.999999747378752e-06, momentum=0.0
        )
        self.bn2a_branch2a = self.__batch_normalization(
            2, "bn2a_branch2a", num_features=64, eps=9.999999747378752e-06, momentum=0.0
        )
        self.res2a_branch2b = self.__conv(
            2,
            name="res2a_branch2b",
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn2a_branch2b = self.__batch_normalization(
            2, "bn2a_branch2b", num_features=64, eps=9.999999747378752e-06, momentum=0.0
        )
        self.res2a_branch2c = self.__conv(
            2,
            name="res2a_branch2c",
            in_channels=64,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn2a_branch2c = self.__batch_normalization(
            2,
            "bn2a_branch2c",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res2b_branch2a = self.__conv(
            2,
            name="res2b_branch2a",
            in_channels=256,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn2b_branch2a = self.__batch_normalization(
            2, "bn2b_branch2a", num_features=64, eps=9.999999747378752e-06, momentum=0.0
        )
        self.res2b_branch2b = self.__conv(
            2,
            name="res2b_branch2b",
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn2b_branch2b = self.__batch_normalization(
            2, "bn2b_branch2b", num_features=64, eps=9.999999747378752e-06, momentum=0.0
        )
        self.res2b_branch2c = self.__conv(
            2,
            name="res2b_branch2c",
            in_channels=64,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn2b_branch2c = self.__batch_normalization(
            2,
            "bn2b_branch2c",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res2c_branch2a = self.__conv(
            2,
            name="res2c_branch2a",
            in_channels=256,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn2c_branch2a = self.__batch_normalization(
            2, "bn2c_branch2a", num_features=64, eps=9.999999747378752e-06, momentum=0.0
        )
        self.res2c_branch2b = self.__conv(
            2,
            name="res2c_branch2b",
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn2c_branch2b = self.__batch_normalization(
            2, "bn2c_branch2b", num_features=64, eps=9.999999747378752e-06, momentum=0.0
        )
        self.res2c_branch2c = self.__conv(
            2,
            name="res2c_branch2c",
            in_channels=64,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn2c_branch2c = self.__batch_normalization(
            2,
            "bn2c_branch2c",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3a_branch1 = self.__conv(
            2,
            name="res3a_branch1",
            in_channels=256,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(2, 2),
            groups=1,
            bias=False,
        )
        self.res3a_branch2a = self.__conv(
            2,
            name="res3a_branch2a",
            in_channels=256,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(2, 2),
            groups=1,
            bias=False,
        )
        self.bn3a_branch1 = self.__batch_normalization(
            2, "bn3a_branch1", num_features=512, eps=9.999999747378752e-06, momentum=0.0
        )
        self.bn3a_branch2a = self.__batch_normalization(
            2,
            "bn3a_branch2a",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3a_branch2b = self.__conv(
            2,
            name="res3a_branch2b",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3a_branch2b = self.__batch_normalization(
            2,
            "bn3a_branch2b",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3a_branch2c = self.__conv(
            2,
            name="res3a_branch2c",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3a_branch2c = self.__batch_normalization(
            2,
            "bn3a_branch2c",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b1_branch2a = self.__conv(
            2,
            name="res3b1_branch2a",
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b1_branch2a = self.__batch_normalization(
            2,
            "bn3b1_branch2a",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b1_branch2b = self.__conv(
            2,
            name="res3b1_branch2b",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b1_branch2b = self.__batch_normalization(
            2,
            "bn3b1_branch2b",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b1_branch2c = self.__conv(
            2,
            name="res3b1_branch2c",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b1_branch2c = self.__batch_normalization(
            2,
            "bn3b1_branch2c",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b2_branch2a = self.__conv(
            2,
            name="res3b2_branch2a",
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b2_branch2a = self.__batch_normalization(
            2,
            "bn3b2_branch2a",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b2_branch2b = self.__conv(
            2,
            name="res3b2_branch2b",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b2_branch2b = self.__batch_normalization(
            2,
            "bn3b2_branch2b",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b2_branch2c = self.__conv(
            2,
            name="res3b2_branch2c",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b2_branch2c = self.__batch_normalization(
            2,
            "bn3b2_branch2c",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b3_branch2a = self.__conv(
            2,
            name="res3b3_branch2a",
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b3_branch2a = self.__batch_normalization(
            2,
            "bn3b3_branch2a",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b3_branch2b = self.__conv(
            2,
            name="res3b3_branch2b",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b3_branch2b = self.__batch_normalization(
            2,
            "bn3b3_branch2b",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b3_branch2c = self.__conv(
            2,
            name="res3b3_branch2c",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b3_branch2c = self.__batch_normalization(
            2,
            "bn3b3_branch2c",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b4_branch2a = self.__conv(
            2,
            name="res3b4_branch2a",
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b4_branch2a = self.__batch_normalization(
            2,
            "bn3b4_branch2a",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b4_branch2b = self.__conv(
            2,
            name="res3b4_branch2b",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b4_branch2b = self.__batch_normalization(
            2,
            "bn3b4_branch2b",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b4_branch2c = self.__conv(
            2,
            name="res3b4_branch2c",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b4_branch2c = self.__batch_normalization(
            2,
            "bn3b4_branch2c",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b5_branch2a = self.__conv(
            2,
            name="res3b5_branch2a",
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b5_branch2a = self.__batch_normalization(
            2,
            "bn3b5_branch2a",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b5_branch2b = self.__conv(
            2,
            name="res3b5_branch2b",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b5_branch2b = self.__batch_normalization(
            2,
            "bn3b5_branch2b",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b5_branch2c = self.__conv(
            2,
            name="res3b5_branch2c",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b5_branch2c = self.__batch_normalization(
            2,
            "bn3b5_branch2c",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b6_branch2a = self.__conv(
            2,
            name="res3b6_branch2a",
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b6_branch2a = self.__batch_normalization(
            2,
            "bn3b6_branch2a",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b6_branch2b = self.__conv(
            2,
            name="res3b6_branch2b",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b6_branch2b = self.__batch_normalization(
            2,
            "bn3b6_branch2b",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b6_branch2c = self.__conv(
            2,
            name="res3b6_branch2c",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b6_branch2c = self.__batch_normalization(
            2,
            "bn3b6_branch2c",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b7_branch2a = self.__conv(
            2,
            name="res3b7_branch2a",
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b7_branch2a = self.__batch_normalization(
            2,
            "bn3b7_branch2a",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b7_branch2b = self.__conv(
            2,
            name="res3b7_branch2b",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b7_branch2b = self.__batch_normalization(
            2,
            "bn3b7_branch2b",
            num_features=128,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res3b7_branch2c = self.__conv(
            2,
            name="res3b7_branch2c",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn3b7_branch2c = self.__batch_normalization(
            2,
            "bn3b7_branch2c",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4a_branch1 = self.__conv(
            2,
            name="res4a_branch1",
            in_channels=512,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(2, 2),
            groups=1,
            bias=False,
        )
        self.res4a_branch2a = self.__conv(
            2,
            name="res4a_branch2a",
            in_channels=512,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(2, 2),
            groups=1,
            bias=False,
        )
        self.bn4a_branch1 = self.__batch_normalization(
            2,
            "bn4a_branch1",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.bn4a_branch2a = self.__batch_normalization(
            2,
            "bn4a_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4a_branch2b = self.__conv(
            2,
            name="res4a_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4a_branch2b = self.__batch_normalization(
            2,
            "bn4a_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4a_branch2c = self.__conv(
            2,
            name="res4a_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4a_branch2c = self.__batch_normalization(
            2,
            "bn4a_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b1_branch2a = self.__conv(
            2,
            name="res4b1_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b1_branch2a = self.__batch_normalization(
            2,
            "bn4b1_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b1_branch2b = self.__conv(
            2,
            name="res4b1_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b1_branch2b = self.__batch_normalization(
            2,
            "bn4b1_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b1_branch2c = self.__conv(
            2,
            name="res4b1_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b1_branch2c = self.__batch_normalization(
            2,
            "bn4b1_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b2_branch2a = self.__conv(
            2,
            name="res4b2_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b2_branch2a = self.__batch_normalization(
            2,
            "bn4b2_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b2_branch2b = self.__conv(
            2,
            name="res4b2_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b2_branch2b = self.__batch_normalization(
            2,
            "bn4b2_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b2_branch2c = self.__conv(
            2,
            name="res4b2_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b2_branch2c = self.__batch_normalization(
            2,
            "bn4b2_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b3_branch2a = self.__conv(
            2,
            name="res4b3_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b3_branch2a = self.__batch_normalization(
            2,
            "bn4b3_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b3_branch2b = self.__conv(
            2,
            name="res4b3_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b3_branch2b = self.__batch_normalization(
            2,
            "bn4b3_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b3_branch2c = self.__conv(
            2,
            name="res4b3_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b3_branch2c = self.__batch_normalization(
            2,
            "bn4b3_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b4_branch2a = self.__conv(
            2,
            name="res4b4_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b4_branch2a = self.__batch_normalization(
            2,
            "bn4b4_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b4_branch2b = self.__conv(
            2,
            name="res4b4_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b4_branch2b = self.__batch_normalization(
            2,
            "bn4b4_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b4_branch2c = self.__conv(
            2,
            name="res4b4_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b4_branch2c = self.__batch_normalization(
            2,
            "bn4b4_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b5_branch2a = self.__conv(
            2,
            name="res4b5_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b5_branch2a = self.__batch_normalization(
            2,
            "bn4b5_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b5_branch2b = self.__conv(
            2,
            name="res4b5_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b5_branch2b = self.__batch_normalization(
            2,
            "bn4b5_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b5_branch2c = self.__conv(
            2,
            name="res4b5_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b5_branch2c = self.__batch_normalization(
            2,
            "bn4b5_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b6_branch2a = self.__conv(
            2,
            name="res4b6_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b6_branch2a = self.__batch_normalization(
            2,
            "bn4b6_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b6_branch2b = self.__conv(
            2,
            name="res4b6_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b6_branch2b = self.__batch_normalization(
            2,
            "bn4b6_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b6_branch2c = self.__conv(
            2,
            name="res4b6_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b6_branch2c = self.__batch_normalization(
            2,
            "bn4b6_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b7_branch2a = self.__conv(
            2,
            name="res4b7_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b7_branch2a = self.__batch_normalization(
            2,
            "bn4b7_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b7_branch2b = self.__conv(
            2,
            name="res4b7_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b7_branch2b = self.__batch_normalization(
            2,
            "bn4b7_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b7_branch2c = self.__conv(
            2,
            name="res4b7_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b7_branch2c = self.__batch_normalization(
            2,
            "bn4b7_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b8_branch2a = self.__conv(
            2,
            name="res4b8_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b8_branch2a = self.__batch_normalization(
            2,
            "bn4b8_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b8_branch2b = self.__conv(
            2,
            name="res4b8_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b8_branch2b = self.__batch_normalization(
            2,
            "bn4b8_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b8_branch2c = self.__conv(
            2,
            name="res4b8_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b8_branch2c = self.__batch_normalization(
            2,
            "bn4b8_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b9_branch2a = self.__conv(
            2,
            name="res4b9_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b9_branch2a = self.__batch_normalization(
            2,
            "bn4b9_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b9_branch2b = self.__conv(
            2,
            name="res4b9_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b9_branch2b = self.__batch_normalization(
            2,
            "bn4b9_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b9_branch2c = self.__conv(
            2,
            name="res4b9_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b9_branch2c = self.__batch_normalization(
            2,
            "bn4b9_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b10_branch2a = self.__conv(
            2,
            name="res4b10_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b10_branch2a = self.__batch_normalization(
            2,
            "bn4b10_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b10_branch2b = self.__conv(
            2,
            name="res4b10_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b10_branch2b = self.__batch_normalization(
            2,
            "bn4b10_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b10_branch2c = self.__conv(
            2,
            name="res4b10_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b10_branch2c = self.__batch_normalization(
            2,
            "bn4b10_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b11_branch2a = self.__conv(
            2,
            name="res4b11_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b11_branch2a = self.__batch_normalization(
            2,
            "bn4b11_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b11_branch2b = self.__conv(
            2,
            name="res4b11_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b11_branch2b = self.__batch_normalization(
            2,
            "bn4b11_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b11_branch2c = self.__conv(
            2,
            name="res4b11_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b11_branch2c = self.__batch_normalization(
            2,
            "bn4b11_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b12_branch2a = self.__conv(
            2,
            name="res4b12_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b12_branch2a = self.__batch_normalization(
            2,
            "bn4b12_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b12_branch2b = self.__conv(
            2,
            name="res4b12_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b12_branch2b = self.__batch_normalization(
            2,
            "bn4b12_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b12_branch2c = self.__conv(
            2,
            name="res4b12_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b12_branch2c = self.__batch_normalization(
            2,
            "bn4b12_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b13_branch2a = self.__conv(
            2,
            name="res4b13_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b13_branch2a = self.__batch_normalization(
            2,
            "bn4b13_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b13_branch2b = self.__conv(
            2,
            name="res4b13_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b13_branch2b = self.__batch_normalization(
            2,
            "bn4b13_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b13_branch2c = self.__conv(
            2,
            name="res4b13_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b13_branch2c = self.__batch_normalization(
            2,
            "bn4b13_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b14_branch2a = self.__conv(
            2,
            name="res4b14_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b14_branch2a = self.__batch_normalization(
            2,
            "bn4b14_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b14_branch2b = self.__conv(
            2,
            name="res4b14_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b14_branch2b = self.__batch_normalization(
            2,
            "bn4b14_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b14_branch2c = self.__conv(
            2,
            name="res4b14_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b14_branch2c = self.__batch_normalization(
            2,
            "bn4b14_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b15_branch2a = self.__conv(
            2,
            name="res4b15_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b15_branch2a = self.__batch_normalization(
            2,
            "bn4b15_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b15_branch2b = self.__conv(
            2,
            name="res4b15_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b15_branch2b = self.__batch_normalization(
            2,
            "bn4b15_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b15_branch2c = self.__conv(
            2,
            name="res4b15_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b15_branch2c = self.__batch_normalization(
            2,
            "bn4b15_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b16_branch2a = self.__conv(
            2,
            name="res4b16_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b16_branch2a = self.__batch_normalization(
            2,
            "bn4b16_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b16_branch2b = self.__conv(
            2,
            name="res4b16_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b16_branch2b = self.__batch_normalization(
            2,
            "bn4b16_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b16_branch2c = self.__conv(
            2,
            name="res4b16_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b16_branch2c = self.__batch_normalization(
            2,
            "bn4b16_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b17_branch2a = self.__conv(
            2,
            name="res4b17_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b17_branch2a = self.__batch_normalization(
            2,
            "bn4b17_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b17_branch2b = self.__conv(
            2,
            name="res4b17_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b17_branch2b = self.__batch_normalization(
            2,
            "bn4b17_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b17_branch2c = self.__conv(
            2,
            name="res4b17_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b17_branch2c = self.__batch_normalization(
            2,
            "bn4b17_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b18_branch2a = self.__conv(
            2,
            name="res4b18_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b18_branch2a = self.__batch_normalization(
            2,
            "bn4b18_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b18_branch2b = self.__conv(
            2,
            name="res4b18_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b18_branch2b = self.__batch_normalization(
            2,
            "bn4b18_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b18_branch2c = self.__conv(
            2,
            name="res4b18_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b18_branch2c = self.__batch_normalization(
            2,
            "bn4b18_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b19_branch2a = self.__conv(
            2,
            name="res4b19_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b19_branch2a = self.__batch_normalization(
            2,
            "bn4b19_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b19_branch2b = self.__conv(
            2,
            name="res4b19_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b19_branch2b = self.__batch_normalization(
            2,
            "bn4b19_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b19_branch2c = self.__conv(
            2,
            name="res4b19_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b19_branch2c = self.__batch_normalization(
            2,
            "bn4b19_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b20_branch2a = self.__conv(
            2,
            name="res4b20_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b20_branch2a = self.__batch_normalization(
            2,
            "bn4b20_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b20_branch2b = self.__conv(
            2,
            name="res4b20_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b20_branch2b = self.__batch_normalization(
            2,
            "bn4b20_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b20_branch2c = self.__conv(
            2,
            name="res4b20_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b20_branch2c = self.__batch_normalization(
            2,
            "bn4b20_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b21_branch2a = self.__conv(
            2,
            name="res4b21_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b21_branch2a = self.__batch_normalization(
            2,
            "bn4b21_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b21_branch2b = self.__conv(
            2,
            name="res4b21_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b21_branch2b = self.__batch_normalization(
            2,
            "bn4b21_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b21_branch2c = self.__conv(
            2,
            name="res4b21_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b21_branch2c = self.__batch_normalization(
            2,
            "bn4b21_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b22_branch2a = self.__conv(
            2,
            name="res4b22_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b22_branch2a = self.__batch_normalization(
            2,
            "bn4b22_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b22_branch2b = self.__conv(
            2,
            name="res4b22_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b22_branch2b = self.__batch_normalization(
            2,
            "bn4b22_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b22_branch2c = self.__conv(
            2,
            name="res4b22_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b22_branch2c = self.__batch_normalization(
            2,
            "bn4b22_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b23_branch2a = self.__conv(
            2,
            name="res4b23_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b23_branch2a = self.__batch_normalization(
            2,
            "bn4b23_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b23_branch2b = self.__conv(
            2,
            name="res4b23_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b23_branch2b = self.__batch_normalization(
            2,
            "bn4b23_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b23_branch2c = self.__conv(
            2,
            name="res4b23_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b23_branch2c = self.__batch_normalization(
            2,
            "bn4b23_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b24_branch2a = self.__conv(
            2,
            name="res4b24_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b24_branch2a = self.__batch_normalization(
            2,
            "bn4b24_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b24_branch2b = self.__conv(
            2,
            name="res4b24_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b24_branch2b = self.__batch_normalization(
            2,
            "bn4b24_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b24_branch2c = self.__conv(
            2,
            name="res4b24_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b24_branch2c = self.__batch_normalization(
            2,
            "bn4b24_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b25_branch2a = self.__conv(
            2,
            name="res4b25_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b25_branch2a = self.__batch_normalization(
            2,
            "bn4b25_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b25_branch2b = self.__conv(
            2,
            name="res4b25_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b25_branch2b = self.__batch_normalization(
            2,
            "bn4b25_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b25_branch2c = self.__conv(
            2,
            name="res4b25_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b25_branch2c = self.__batch_normalization(
            2,
            "bn4b25_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b26_branch2a = self.__conv(
            2,
            name="res4b26_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b26_branch2a = self.__batch_normalization(
            2,
            "bn4b26_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b26_branch2b = self.__conv(
            2,
            name="res4b26_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b26_branch2b = self.__batch_normalization(
            2,
            "bn4b26_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b26_branch2c = self.__conv(
            2,
            name="res4b26_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b26_branch2c = self.__batch_normalization(
            2,
            "bn4b26_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b27_branch2a = self.__conv(
            2,
            name="res4b27_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b27_branch2a = self.__batch_normalization(
            2,
            "bn4b27_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b27_branch2b = self.__conv(
            2,
            name="res4b27_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b27_branch2b = self.__batch_normalization(
            2,
            "bn4b27_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b27_branch2c = self.__conv(
            2,
            name="res4b27_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b27_branch2c = self.__batch_normalization(
            2,
            "bn4b27_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b28_branch2a = self.__conv(
            2,
            name="res4b28_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b28_branch2a = self.__batch_normalization(
            2,
            "bn4b28_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b28_branch2b = self.__conv(
            2,
            name="res4b28_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b28_branch2b = self.__batch_normalization(
            2,
            "bn4b28_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b28_branch2c = self.__conv(
            2,
            name="res4b28_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b28_branch2c = self.__batch_normalization(
            2,
            "bn4b28_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b29_branch2a = self.__conv(
            2,
            name="res4b29_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b29_branch2a = self.__batch_normalization(
            2,
            "bn4b29_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b29_branch2b = self.__conv(
            2,
            name="res4b29_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b29_branch2b = self.__batch_normalization(
            2,
            "bn4b29_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b29_branch2c = self.__conv(
            2,
            name="res4b29_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b29_branch2c = self.__batch_normalization(
            2,
            "bn4b29_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b30_branch2a = self.__conv(
            2,
            name="res4b30_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b30_branch2a = self.__batch_normalization(
            2,
            "bn4b30_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b30_branch2b = self.__conv(
            2,
            name="res4b30_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b30_branch2b = self.__batch_normalization(
            2,
            "bn4b30_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b30_branch2c = self.__conv(
            2,
            name="res4b30_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b30_branch2c = self.__batch_normalization(
            2,
            "bn4b30_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b31_branch2a = self.__conv(
            2,
            name="res4b31_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b31_branch2a = self.__batch_normalization(
            2,
            "bn4b31_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b31_branch2b = self.__conv(
            2,
            name="res4b31_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b31_branch2b = self.__batch_normalization(
            2,
            "bn4b31_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b31_branch2c = self.__conv(
            2,
            name="res4b31_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b31_branch2c = self.__batch_normalization(
            2,
            "bn4b31_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b32_branch2a = self.__conv(
            2,
            name="res4b32_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b32_branch2a = self.__batch_normalization(
            2,
            "bn4b32_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b32_branch2b = self.__conv(
            2,
            name="res4b32_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b32_branch2b = self.__batch_normalization(
            2,
            "bn4b32_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b32_branch2c = self.__conv(
            2,
            name="res4b32_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b32_branch2c = self.__batch_normalization(
            2,
            "bn4b32_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b33_branch2a = self.__conv(
            2,
            name="res4b33_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b33_branch2a = self.__batch_normalization(
            2,
            "bn4b33_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b33_branch2b = self.__conv(
            2,
            name="res4b33_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b33_branch2b = self.__batch_normalization(
            2,
            "bn4b33_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b33_branch2c = self.__conv(
            2,
            name="res4b33_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b33_branch2c = self.__batch_normalization(
            2,
            "bn4b33_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b34_branch2a = self.__conv(
            2,
            name="res4b34_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b34_branch2a = self.__batch_normalization(
            2,
            "bn4b34_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b34_branch2b = self.__conv(
            2,
            name="res4b34_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b34_branch2b = self.__batch_normalization(
            2,
            "bn4b34_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b34_branch2c = self.__conv(
            2,
            name="res4b34_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b34_branch2c = self.__batch_normalization(
            2,
            "bn4b34_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b35_branch2a = self.__conv(
            2,
            name="res4b35_branch2a",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b35_branch2a = self.__batch_normalization(
            2,
            "bn4b35_branch2a",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b35_branch2b = self.__conv(
            2,
            name="res4b35_branch2b",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b35_branch2b = self.__batch_normalization(
            2,
            "bn4b35_branch2b",
            num_features=256,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res4b35_branch2c = self.__conv(
            2,
            name="res4b35_branch2c",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn4b35_branch2c = self.__batch_normalization(
            2,
            "bn4b35_branch2c",
            num_features=1024,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res5a_branch1 = self.__conv(
            2,
            name="res5a_branch1",
            in_channels=1024,
            out_channels=2048,
            kernel_size=(1, 1),
            stride=(2, 2),
            groups=1,
            bias=False,
        )
        self.res5a_branch2a = self.__conv(
            2,
            name="res5a_branch2a",
            in_channels=1024,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(2, 2),
            groups=1,
            bias=False,
        )
        self.bn5a_branch1 = self.__batch_normalization(
            2,
            "bn5a_branch1",
            num_features=2048,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.bn5a_branch2a = self.__batch_normalization(
            2,
            "bn5a_branch2a",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res5a_branch2b = self.__conv(
            2,
            name="res5a_branch2b",
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn5a_branch2b = self.__batch_normalization(
            2,
            "bn5a_branch2b",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res5a_branch2c = self.__conv(
            2,
            name="res5a_branch2c",
            in_channels=512,
            out_channels=2048,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn5a_branch2c = self.__batch_normalization(
            2,
            "bn5a_branch2c",
            num_features=2048,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res5b_branch2a = self.__conv(
            2,
            name="res5b_branch2a",
            in_channels=2048,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn5b_branch2a = self.__batch_normalization(
            2,
            "bn5b_branch2a",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res5b_branch2b = self.__conv(
            2,
            name="res5b_branch2b",
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn5b_branch2b = self.__batch_normalization(
            2,
            "bn5b_branch2b",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res5b_branch2c = self.__conv(
            2,
            name="res5b_branch2c",
            in_channels=512,
            out_channels=2048,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn5b_branch2c = self.__batch_normalization(
            2,
            "bn5b_branch2c",
            num_features=2048,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res5c_branch2a = self.__conv(
            2,
            name="res5c_branch2a",
            in_channels=2048,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn5c_branch2a = self.__batch_normalization(
            2,
            "bn5c_branch2a",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res5c_branch2b = self.__conv(
            2,
            name="res5c_branch2b",
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn5c_branch2b = self.__batch_normalization(
            2,
            "bn5c_branch2b",
            num_features=512,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.res5c_branch2c = self.__conv(
            2,
            name="res5c_branch2c",
            in_channels=512,
            out_channels=2048,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=False,
        )
        self.bn5c_branch2c = self.__batch_normalization(
            2,
            "bn5c_branch2c",
            num_features=2048,
            eps=9.999999747378752e-06,
            momentum=0.0,
        )
        self.fc365_1 = self.__dense(
            name="fc365_1", in_features=2048, out_features=365, bias=True
        )

        # Create wrapper modules for functional operations
        self.pad_conv1 = Pad((3, 3, 3, 3))
        self.relu_conv1 = ReLU(inplace=False)
        self.pad_pool1 = Pad((0, 1, 0, 1), value=float("-inf"))
        self.maxpool1 = MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.relu_wrapper = ReLU(inplace=False)
        self.avgpool5 = AvgPool2d(kernel_size=7, stride=1, padding=0)

        # Add module for residual connections
        self.add_module_ = Add()

        # Reusable Pad instances for common padding patterns
        self.pad_1x1 = Pad((1, 1, 1, 1))  # For 3x3 conv padding

        # Create all Pad instances needed for residual blocks (all use (1,1,1,1) padding)
        # We'll create enough instances for all residual blocks
        num_residual_blocks = (
            3 + 7 + 1 + 35 + 1 + 1 + 1
        )  # res2:3, res3:7, res4:35, res5:3
        self.pad_instances = [
            Pad((1, 1, 1, 1)) for _ in range(num_residual_blocks * 2)
        ]  # 2 pads per block
        self.pad_idx = 0  # Index to track which pad instance to use

        # Build features list similar to VGG's self.features - all operations in execution order
        # This list contains all operations that will be executed in forward, in order
        self.features = self._build_features_list()

    def forward(self, x):
        # Track pad index for residual blocks
        pad_idx = 0
        # conv1_pad = F.pad(x, (3, 3, 3, 3))
        conv1_pad = self.pad_conv1(x)
        conv1 = self.conv1(conv1_pad)
        bn_conv1 = self.bn_conv1(conv1)
        # conv1_relu = self.relu_wrapper(bn_conv1)
        conv1_relu = self.relu_conv1(bn_conv1)
        # pool1_pad = F.pad(conv1_relu, (0, 1, 0, 1), value=float("-inf"))
        pool1_pad = self.pad_pool1(conv1_relu)
        # pool1, pool1_idx = F.max_pool2d(
        #     pool1_pad,
        #     kernel_size=(3, 3),
        #     stride=(2, 2),
        #     padding=0,
        #     ceil_mode=False,
        #     return_indices=True,
        # )
        pool1 = self.maxpool1(pool1_pad)
        pool1_idx = None
        res2a_branch1 = self.res2a_branch1(pool1)
        res2a_branch2a = self.res2a_branch2a(pool1)
        bn2a_branch1 = self.bn2a_branch1(res2a_branch1)
        bn2a_branch2a = self.bn2a_branch2a(res2a_branch2a)
        # res2a_branch2a_relu = self.relu_wrapper(bn2a_branch2a)
        # res2a_branch2b_pad = self.pad_instances[0](res2a_branch2a_relu)
        res2a_branch2a_relu = self.relu_wrapper(bn2a_branch2a)
        res2a_branch2b_pad = self.pad_instances[0](res2a_branch2a_relu)
        res2a_branch2b = self.res2a_branch2b(res2a_branch2b_pad)
        bn2a_branch2b = self.bn2a_branch2b(res2a_branch2b)
        # res2a_branch2b_relu = self.relu_wrapper(bn2a_branch2b)
        res2a_branch2b_relu = self.relu_wrapper(bn2a_branch2b)
        res2a_branch2c = self.res2a_branch2c(res2a_branch2b_relu)
        bn2a_branch2c = self.bn2a_branch2c(res2a_branch2c)
        res2a = self.add_module_(bn2a_branch1, bn2a_branch2c)
        # res2a_relu = self.relu_wrapper(res2a)
        res2a_relu = self.relu_wrapper(res2a)
        res2b_branch2a = self.res2b_branch2a(res2a_relu)
        bn2b_branch2a = self.bn2b_branch2a(res2b_branch2a)
        # res2b_branch2a_relu = self.relu_wrapper(bn2b_branch2a)
        res2b_branch2a_relu = self.relu_wrapper(bn2b_branch2a)
        # res2b_branch2b_pad = self.pad_instances[1](res2b_branch2a_relu)
        res2b_branch2b_pad = self.pad_instances[1](res2b_branch2a_relu)
        res2b_branch2b = self.res2b_branch2b(res2b_branch2b_pad)
        bn2b_branch2b = self.bn2b_branch2b(res2b_branch2b)
        # res2b_branch2b_relu = self.relu_wrapper(bn2b_branch2b)
        res2b_branch2b_relu = self.relu_wrapper(bn2b_branch2b)
        res2b_branch2c = self.res2b_branch2c(res2b_branch2b_relu)
        bn2b_branch2c = self.bn2b_branch2c(res2b_branch2c)
        res2b = self.add_module_(res2a_relu, bn2b_branch2c)
        # res2b_relu = self.relu_wrapper(res2b)
        res2b_relu = self.relu_wrapper(res2b)
        res2c_branch2a = self.res2c_branch2a(res2b_relu)
        bn2c_branch2a = self.bn2c_branch2a(res2c_branch2a)
        # res2c_branch2a_relu = self.relu_wrapper(bn2c_branch2a)
        res2c_branch2a_relu = self.relu_wrapper(bn2c_branch2a)
        # res2c_branch2b_pad = self.pad_instances[2](res2c_branch2a_relu)
        res2c_branch2b_pad = self.pad_instances[2](res2c_branch2a_relu)
        res2c_branch2b = self.res2c_branch2b(res2c_branch2b_pad)
        bn2c_branch2b = self.bn2c_branch2b(res2c_branch2b)
        # res2c_branch2b_relu = self.relu_wrapper(bn2c_branch2b)
        res2c_branch2b_relu = self.relu_wrapper(bn2c_branch2b)
        res2c_branch2c = self.res2c_branch2c(res2c_branch2b_relu)
        bn2c_branch2c = self.bn2c_branch2c(res2c_branch2c)
        res2c = self.add_module_(res2b_relu, bn2c_branch2c)
        # res2c_relu = self.relu_wrapper(res2c)
        res2c_relu = self.relu_wrapper(res2c)
        res3a_branch1 = self.res3a_branch1(res2c_relu)
        res3a_branch2a = self.res3a_branch2a(res2c_relu)
        bn3a_branch1 = self.bn3a_branch1(res3a_branch1)
        bn3a_branch2a = self.bn3a_branch2a(res3a_branch2a)
        # res3a_branch2a_relu = self.relu_wrapper(bn3a_branch2a)
        res3a_branch2a_relu = self.relu_wrapper(bn3a_branch2a)
        # res3a_branch2b_pad = self.pad_instances[3](res3a_branch2a_relu)
        res3a_branch2b_pad = self.pad_instances[3](res3a_branch2a_relu)
        res3a_branch2b = self.res3a_branch2b(res3a_branch2b_pad)
        bn3a_branch2b = self.bn3a_branch2b(res3a_branch2b)
        # res3a_branch2b_relu = self.relu_wrapper(bn3a_branch2b)
        res3a_branch2b_relu = self.relu_wrapper(bn3a_branch2b)
        res3a_branch2c = self.res3a_branch2c(res3a_branch2b_relu)
        bn3a_branch2c = self.bn3a_branch2c(res3a_branch2c)
        res3a = self.add_module_(bn3a_branch1, bn3a_branch2c)
        # res3a_relu = self.relu_wrapper(res3a)
        res3a_relu = self.relu_wrapper(res3a)
        res3b1_branch2a = self.res3b1_branch2a(res3a_relu)
        bn3b1_branch2a = self.bn3b1_branch2a(res3b1_branch2a)
        # res3b1_branch2a_relu = self.relu_wrapper(bn3b1_branch2a)
        res3b1_branch2a_relu = self.relu_wrapper(bn3b1_branch2a)
        # res3b1_branch2b_pad = self.pad_instances[4](res3b1_branch2a_relu)
        res3b1_branch2b_pad = self.pad_instances[4](res3b1_branch2a_relu)
        res3b1_branch2b = self.res3b1_branch2b(res3b1_branch2b_pad)
        bn3b1_branch2b = self.bn3b1_branch2b(res3b1_branch2b)
        res3b1_branch2b_relu = self.relu_wrapper(bn3b1_branch2b)
        res3b1_branch2c = self.res3b1_branch2c(res3b1_branch2b_relu)
        bn3b1_branch2c = self.bn3b1_branch2c(res3b1_branch2c)
        res3b1 = self.add_module_(res3a_relu, bn3b1_branch2c)
        res3b1_relu = self.relu_wrapper(res3b1)
        res3b2_branch2a = self.res3b2_branch2a(res3b1_relu)
        bn3b2_branch2a = self.bn3b2_branch2a(res3b2_branch2a)
        res3b2_branch2a_relu = self.relu_wrapper(bn3b2_branch2a)
        res3b2_branch2b_pad = self.pad_instances[5](res3b2_branch2a_relu)
        res3b2_branch2b = self.res3b2_branch2b(res3b2_branch2b_pad)
        bn3b2_branch2b = self.bn3b2_branch2b(res3b2_branch2b)
        res3b2_branch2b_relu = self.relu_wrapper(bn3b2_branch2b)
        res3b2_branch2c = self.res3b2_branch2c(res3b2_branch2b_relu)
        bn3b2_branch2c = self.bn3b2_branch2c(res3b2_branch2c)
        res3b2 = self.add_module_(res3b1_relu, bn3b2_branch2c)
        res3b2_relu = self.relu_wrapper(res3b2)
        res3b3_branch2a = self.res3b3_branch2a(res3b2_relu)
        bn3b3_branch2a = self.bn3b3_branch2a(res3b3_branch2a)
        res3b3_branch2a_relu = self.relu_wrapper(bn3b3_branch2a)
        res3b3_branch2b_pad = self.pad_instances[6](res3b3_branch2a_relu)
        res3b3_branch2b = self.res3b3_branch2b(res3b3_branch2b_pad)
        bn3b3_branch2b = self.bn3b3_branch2b(res3b3_branch2b)
        res3b3_branch2b_relu = self.relu_wrapper(bn3b3_branch2b)
        res3b3_branch2c = self.res3b3_branch2c(res3b3_branch2b_relu)
        bn3b3_branch2c = self.bn3b3_branch2c(res3b3_branch2c)
        res3b3 = self.add_module_(res3b2_relu, bn3b3_branch2c)
        res3b3_relu = self.relu_wrapper(res3b3)
        res3b4_branch2a = self.res3b4_branch2a(res3b3_relu)
        bn3b4_branch2a = self.bn3b4_branch2a(res3b4_branch2a)
        res3b4_branch2a_relu = self.relu_wrapper(bn3b4_branch2a)
        res3b4_branch2b_pad = self.pad_instances[7](res3b4_branch2a_relu)
        res3b4_branch2b = self.res3b4_branch2b(res3b4_branch2b_pad)
        bn3b4_branch2b = self.bn3b4_branch2b(res3b4_branch2b)
        res3b4_branch2b_relu = self.relu_wrapper(bn3b4_branch2b)
        res3b4_branch2c = self.res3b4_branch2c(res3b4_branch2b_relu)
        bn3b4_branch2c = self.bn3b4_branch2c(res3b4_branch2c)
        res3b4 = self.add_module_(res3b3_relu, bn3b4_branch2c)
        res3b4_relu = self.relu_wrapper(res3b4)
        res3b5_branch2a = self.res3b5_branch2a(res3b4_relu)
        bn3b5_branch2a = self.bn3b5_branch2a(res3b5_branch2a)
        res3b5_branch2a_relu = self.relu_wrapper(bn3b5_branch2a)
        res3b5_branch2b_pad = self.pad_instances[8](res3b5_branch2a_relu)
        res3b5_branch2b = self.res3b5_branch2b(res3b5_branch2b_pad)
        bn3b5_branch2b = self.bn3b5_branch2b(res3b5_branch2b)
        res3b5_branch2b_relu = self.relu_wrapper(bn3b5_branch2b)
        res3b5_branch2c = self.res3b5_branch2c(res3b5_branch2b_relu)
        bn3b5_branch2c = self.bn3b5_branch2c(res3b5_branch2c)
        res3b5 = self.add_module_(res3b4_relu, bn3b5_branch2c)
        res3b5_relu = self.relu_wrapper(res3b5)
        res3b6_branch2a = self.res3b6_branch2a(res3b5_relu)
        bn3b6_branch2a = self.bn3b6_branch2a(res3b6_branch2a)
        res3b6_branch2a_relu = self.relu_wrapper(bn3b6_branch2a)
        res3b6_branch2b_pad = self.pad_instances[9](res3b6_branch2a_relu)
        res3b6_branch2b = self.res3b6_branch2b(res3b6_branch2b_pad)
        bn3b6_branch2b = self.bn3b6_branch2b(res3b6_branch2b)
        res3b6_branch2b_relu = self.relu_wrapper(bn3b6_branch2b)
        res3b6_branch2c = self.res3b6_branch2c(res3b6_branch2b_relu)
        bn3b6_branch2c = self.bn3b6_branch2c(res3b6_branch2c)
        res3b6 = self.add_module_(res3b5_relu, bn3b6_branch2c)
        res3b6_relu = self.relu_wrapper(res3b6)
        res3b7_branch2a = self.res3b7_branch2a(res3b6_relu)
        bn3b7_branch2a = self.bn3b7_branch2a(res3b7_branch2a)
        res3b7_branch2a_relu = self.relu_wrapper(bn3b7_branch2a)
        res3b7_branch2b_pad = self.pad_instances[10](res3b7_branch2a_relu)
        res3b7_branch2b = self.res3b7_branch2b(res3b7_branch2b_pad)
        bn3b7_branch2b = self.bn3b7_branch2b(res3b7_branch2b)
        res3b7_branch2b_relu = self.relu_wrapper(bn3b7_branch2b)
        res3b7_branch2c = self.res3b7_branch2c(res3b7_branch2b_relu)
        bn3b7_branch2c = self.bn3b7_branch2c(res3b7_branch2c)
        res3b7 = self.add_module_(res3b6_relu, bn3b7_branch2c)
        res3b7_relu = self.relu_wrapper(res3b7)
        res4a_branch1 = self.res4a_branch1(res3b7_relu)
        res4a_branch2a = self.res4a_branch2a(res3b7_relu)
        bn4a_branch1 = self.bn4a_branch1(res4a_branch1)
        bn4a_branch2a = self.bn4a_branch2a(res4a_branch2a)
        res4a_branch2a_relu = self.relu_wrapper(bn4a_branch2a)
        res4a_branch2b_pad = self.pad_instances[11](res4a_branch2a_relu)
        res4a_branch2b = self.res4a_branch2b(res4a_branch2b_pad)
        bn4a_branch2b = self.bn4a_branch2b(res4a_branch2b)
        res4a_branch2b_relu = self.relu_wrapper(bn4a_branch2b)
        res4a_branch2c = self.res4a_branch2c(res4a_branch2b_relu)
        bn4a_branch2c = self.bn4a_branch2c(res4a_branch2c)
        res4a = self.add_module_(bn4a_branch1, bn4a_branch2c)
        res4a_relu = self.relu_wrapper(res4a)
        res4b1_branch2a = self.res4b1_branch2a(res4a_relu)
        bn4b1_branch2a = self.bn4b1_branch2a(res4b1_branch2a)
        res4b1_branch2a_relu = self.relu_wrapper(bn4b1_branch2a)
        res4b1_branch2b_pad = self.pad_instances[12](res4b1_branch2a_relu)
        res4b1_branch2b = self.res4b1_branch2b(res4b1_branch2b_pad)
        bn4b1_branch2b = self.bn4b1_branch2b(res4b1_branch2b)
        res4b1_branch2b_relu = self.relu_wrapper(bn4b1_branch2b)
        res4b1_branch2c = self.res4b1_branch2c(res4b1_branch2b_relu)
        bn4b1_branch2c = self.bn4b1_branch2c(res4b1_branch2c)
        res4b1 = self.add_module_(res4a_relu, bn4b1_branch2c)
        res4b1_relu = self.relu_wrapper(res4b1)
        res4b2_branch2a = self.res4b2_branch2a(res4b1_relu)
        bn4b2_branch2a = self.bn4b2_branch2a(res4b2_branch2a)
        res4b2_branch2a_relu = self.relu_wrapper(bn4b2_branch2a)
        res4b2_branch2b_pad = self.pad_instances[13](res4b2_branch2a_relu)
        res4b2_branch2b = self.res4b2_branch2b(res4b2_branch2b_pad)
        bn4b2_branch2b = self.bn4b2_branch2b(res4b2_branch2b)
        res4b2_branch2b_relu = self.relu_wrapper(bn4b2_branch2b)
        res4b2_branch2c = self.res4b2_branch2c(res4b2_branch2b_relu)
        bn4b2_branch2c = self.bn4b2_branch2c(res4b2_branch2c)
        res4b2 = self.add_module_(res4b1_relu, bn4b2_branch2c)
        res4b2_relu = self.relu_wrapper(res4b2)
        res4b3_branch2a = self.res4b3_branch2a(res4b2_relu)
        bn4b3_branch2a = self.bn4b3_branch2a(res4b3_branch2a)
        res4b3_branch2a_relu = self.relu_wrapper(bn4b3_branch2a)
        res4b3_branch2b_pad = self.pad_instances[14](res4b3_branch2a_relu)
        res4b3_branch2b = self.res4b3_branch2b(res4b3_branch2b_pad)
        bn4b3_branch2b = self.bn4b3_branch2b(res4b3_branch2b)
        res4b3_branch2b_relu = self.relu_wrapper(bn4b3_branch2b)
        res4b3_branch2c = self.res4b3_branch2c(res4b3_branch2b_relu)
        bn4b3_branch2c = self.bn4b3_branch2c(res4b3_branch2c)
        res4b3 = self.add_module_(res4b2_relu, bn4b3_branch2c)
        res4b3_relu = self.relu_wrapper(res4b3)
        res4b4_branch2a = self.res4b4_branch2a(res4b3_relu)
        bn4b4_branch2a = self.bn4b4_branch2a(res4b4_branch2a)
        res4b4_branch2a_relu = self.relu_wrapper(bn4b4_branch2a)
        res4b4_branch2b_pad = self.pad_instances[15](res4b4_branch2a_relu)
        res4b4_branch2b = self.res4b4_branch2b(res4b4_branch2b_pad)
        bn4b4_branch2b = self.bn4b4_branch2b(res4b4_branch2b)
        res4b4_branch2b_relu = self.relu_wrapper(bn4b4_branch2b)
        res4b4_branch2c = self.res4b4_branch2c(res4b4_branch2b_relu)
        bn4b4_branch2c = self.bn4b4_branch2c(res4b4_branch2c)
        res4b4 = self.add_module_(res4b3_relu, bn4b4_branch2c)
        res4b4_relu = self.relu_wrapper(res4b4)
        res4b5_branch2a = self.res4b5_branch2a(res4b4_relu)
        bn4b5_branch2a = self.bn4b5_branch2a(res4b5_branch2a)
        res4b5_branch2a_relu = self.relu_wrapper(bn4b5_branch2a)
        res4b5_branch2b_pad = self.pad_instances[16](res4b5_branch2a_relu)
        res4b5_branch2b = self.res4b5_branch2b(res4b5_branch2b_pad)
        bn4b5_branch2b = self.bn4b5_branch2b(res4b5_branch2b)
        res4b5_branch2b_relu = self.relu_wrapper(bn4b5_branch2b)
        res4b5_branch2c = self.res4b5_branch2c(res4b5_branch2b_relu)
        bn4b5_branch2c = self.bn4b5_branch2c(res4b5_branch2c)
        res4b5 = self.add_module_(res4b4_relu, bn4b5_branch2c)
        res4b5_relu = self.relu_wrapper(res4b5)
        res4b6_branch2a = self.res4b6_branch2a(res4b5_relu)
        bn4b6_branch2a = self.bn4b6_branch2a(res4b6_branch2a)
        res4b6_branch2a_relu = self.relu_wrapper(bn4b6_branch2a)
        res4b6_branch2b_pad = self.pad_instances[17](res4b6_branch2a_relu)
        res4b6_branch2b = self.res4b6_branch2b(res4b6_branch2b_pad)
        bn4b6_branch2b = self.bn4b6_branch2b(res4b6_branch2b)
        res4b6_branch2b_relu = self.relu_wrapper(bn4b6_branch2b)
        res4b6_branch2c = self.res4b6_branch2c(res4b6_branch2b_relu)
        bn4b6_branch2c = self.bn4b6_branch2c(res4b6_branch2c)
        res4b6 = self.add_module_(res4b5_relu, bn4b6_branch2c)
        res4b6_relu = self.relu_wrapper(res4b6)
        res4b7_branch2a = self.res4b7_branch2a(res4b6_relu)
        bn4b7_branch2a = self.bn4b7_branch2a(res4b7_branch2a)
        res4b7_branch2a_relu = self.relu_wrapper(bn4b7_branch2a)
        res4b7_branch2b_pad = self.pad_instances[18](res4b7_branch2a_relu)
        res4b7_branch2b = self.res4b7_branch2b(res4b7_branch2b_pad)
        bn4b7_branch2b = self.bn4b7_branch2b(res4b7_branch2b)
        res4b7_branch2b_relu = self.relu_wrapper(bn4b7_branch2b)
        res4b7_branch2c = self.res4b7_branch2c(res4b7_branch2b_relu)
        bn4b7_branch2c = self.bn4b7_branch2c(res4b7_branch2c)
        res4b7 = self.add_module_(res4b6_relu, bn4b7_branch2c)
        res4b7_relu = self.relu_wrapper(res4b7)
        res4b8_branch2a = self.res4b8_branch2a(res4b7_relu)
        bn4b8_branch2a = self.bn4b8_branch2a(res4b8_branch2a)
        res4b8_branch2a_relu = self.relu_wrapper(bn4b8_branch2a)
        res4b8_branch2b_pad = self.pad_instances[19](res4b8_branch2a_relu)
        res4b8_branch2b = self.res4b8_branch2b(res4b8_branch2b_pad)
        bn4b8_branch2b = self.bn4b8_branch2b(res4b8_branch2b)
        res4b8_branch2b_relu = self.relu_wrapper(bn4b8_branch2b)
        res4b8_branch2c = self.res4b8_branch2c(res4b8_branch2b_relu)
        bn4b8_branch2c = self.bn4b8_branch2c(res4b8_branch2c)
        res4b8 = self.add_module_(res4b7_relu, bn4b8_branch2c)
        res4b8_relu = self.relu_wrapper(res4b8)
        res4b9_branch2a = self.res4b9_branch2a(res4b8_relu)
        bn4b9_branch2a = self.bn4b9_branch2a(res4b9_branch2a)
        res4b9_branch2a_relu = self.relu_wrapper(bn4b9_branch2a)
        res4b9_branch2b_pad = self.pad_instances[20](res4b9_branch2a_relu)
        res4b9_branch2b = self.res4b9_branch2b(res4b9_branch2b_pad)
        bn4b9_branch2b = self.bn4b9_branch2b(res4b9_branch2b)
        res4b9_branch2b_relu = self.relu_wrapper(bn4b9_branch2b)
        res4b9_branch2c = self.res4b9_branch2c(res4b9_branch2b_relu)
        bn4b9_branch2c = self.bn4b9_branch2c(res4b9_branch2c)
        res4b9 = self.add_module_(res4b8_relu, bn4b9_branch2c)
        res4b9_relu = self.relu_wrapper(res4b9)
        res4b10_branch2a = self.res4b10_branch2a(res4b9_relu)
        bn4b10_branch2a = self.bn4b10_branch2a(res4b10_branch2a)
        res4b10_branch2a_relu = self.relu_wrapper(bn4b10_branch2a)
        res4b10_branch2b_pad = self.pad_instances[21](res4b10_branch2a_relu)
        res4b10_branch2b = self.res4b10_branch2b(res4b10_branch2b_pad)
        bn4b10_branch2b = self.bn4b10_branch2b(res4b10_branch2b)
        res4b10_branch2b_relu = self.relu_wrapper(bn4b10_branch2b)
        res4b10_branch2c = self.res4b10_branch2c(res4b10_branch2b_relu)
        bn4b10_branch2c = self.bn4b10_branch2c(res4b10_branch2c)
        res4b10 = self.add_module_(res4b9_relu, bn4b10_branch2c)
        res4b10_relu = self.relu_wrapper(res4b10)
        res4b11_branch2a = self.res4b11_branch2a(res4b10_relu)
        bn4b11_branch2a = self.bn4b11_branch2a(res4b11_branch2a)
        res4b11_branch2a_relu = self.relu_wrapper(bn4b11_branch2a)
        res4b11_branch2b_pad = self.pad_instances[22](res4b11_branch2a_relu)
        res4b11_branch2b = self.res4b11_branch2b(res4b11_branch2b_pad)
        bn4b11_branch2b = self.bn4b11_branch2b(res4b11_branch2b)
        res4b11_branch2b_relu = self.relu_wrapper(bn4b11_branch2b)
        res4b11_branch2c = self.res4b11_branch2c(res4b11_branch2b_relu)
        bn4b11_branch2c = self.bn4b11_branch2c(res4b11_branch2c)
        res4b11 = self.add_module_(res4b10_relu, bn4b11_branch2c)
        res4b11_relu = self.relu_wrapper(res4b11)
        res4b12_branch2a = self.res4b12_branch2a(res4b11_relu)
        bn4b12_branch2a = self.bn4b12_branch2a(res4b12_branch2a)
        res4b12_branch2a_relu = self.relu_wrapper(bn4b12_branch2a)
        res4b12_branch2b_pad = self.pad_instances[23](res4b12_branch2a_relu)
        res4b12_branch2b = self.res4b12_branch2b(res4b12_branch2b_pad)
        bn4b12_branch2b = self.bn4b12_branch2b(res4b12_branch2b)
        res4b12_branch2b_relu = self.relu_wrapper(bn4b12_branch2b)
        res4b12_branch2c = self.res4b12_branch2c(res4b12_branch2b_relu)
        bn4b12_branch2c = self.bn4b12_branch2c(res4b12_branch2c)
        res4b12 = self.add_module_(res4b11_relu, bn4b12_branch2c)
        res4b12_relu = self.relu_wrapper(res4b12)
        res4b13_branch2a = self.res4b13_branch2a(res4b12_relu)
        bn4b13_branch2a = self.bn4b13_branch2a(res4b13_branch2a)
        res4b13_branch2a_relu = self.relu_wrapper(bn4b13_branch2a)
        res4b13_branch2b_pad = self.pad_instances[24](res4b13_branch2a_relu)
        res4b13_branch2b = self.res4b13_branch2b(res4b13_branch2b_pad)
        bn4b13_branch2b = self.bn4b13_branch2b(res4b13_branch2b)
        res4b13_branch2b_relu = self.relu_wrapper(bn4b13_branch2b)
        res4b13_branch2c = self.res4b13_branch2c(res4b13_branch2b_relu)
        bn4b13_branch2c = self.bn4b13_branch2c(res4b13_branch2c)
        res4b13 = self.add_module_(res4b12_relu, bn4b13_branch2c)
        res4b13_relu = self.relu_wrapper(res4b13)
        res4b14_branch2a = self.res4b14_branch2a(res4b13_relu)
        bn4b14_branch2a = self.bn4b14_branch2a(res4b14_branch2a)
        res4b14_branch2a_relu = self.relu_wrapper(bn4b14_branch2a)
        res4b14_branch2b_pad = self.pad_instances[25](res4b14_branch2a_relu)
        res4b14_branch2b = self.res4b14_branch2b(res4b14_branch2b_pad)
        bn4b14_branch2b = self.bn4b14_branch2b(res4b14_branch2b)
        res4b14_branch2b_relu = self.relu_wrapper(bn4b14_branch2b)
        res4b14_branch2c = self.res4b14_branch2c(res4b14_branch2b_relu)
        bn4b14_branch2c = self.bn4b14_branch2c(res4b14_branch2c)
        res4b14 = self.add_module_(res4b13_relu, bn4b14_branch2c)
        res4b14_relu = self.relu_wrapper(res4b14)
        res4b15_branch2a = self.res4b15_branch2a(res4b14_relu)
        bn4b15_branch2a = self.bn4b15_branch2a(res4b15_branch2a)
        res4b15_branch2a_relu = self.relu_wrapper(bn4b15_branch2a)
        res4b15_branch2b_pad = self.pad_instances[26](res4b15_branch2a_relu)
        res4b15_branch2b = self.res4b15_branch2b(res4b15_branch2b_pad)
        bn4b15_branch2b = self.bn4b15_branch2b(res4b15_branch2b)
        res4b15_branch2b_relu = self.relu_wrapper(bn4b15_branch2b)
        res4b15_branch2c = self.res4b15_branch2c(res4b15_branch2b_relu)
        bn4b15_branch2c = self.bn4b15_branch2c(res4b15_branch2c)
        res4b15 = self.add_module_(res4b14_relu, bn4b15_branch2c)
        res4b15_relu = self.relu_wrapper(res4b15)
        res4b16_branch2a = self.res4b16_branch2a(res4b15_relu)
        bn4b16_branch2a = self.bn4b16_branch2a(res4b16_branch2a)
        res4b16_branch2a_relu = self.relu_wrapper(bn4b16_branch2a)
        res4b16_branch2b_pad = self.pad_instances[27](res4b16_branch2a_relu)
        res4b16_branch2b = self.res4b16_branch2b(res4b16_branch2b_pad)
        bn4b16_branch2b = self.bn4b16_branch2b(res4b16_branch2b)
        res4b16_branch2b_relu = self.relu_wrapper(bn4b16_branch2b)
        res4b16_branch2c = self.res4b16_branch2c(res4b16_branch2b_relu)
        bn4b16_branch2c = self.bn4b16_branch2c(res4b16_branch2c)
        res4b16 = self.add_module_(res4b15_relu, bn4b16_branch2c)
        res4b16_relu = self.relu_wrapper(res4b16)
        res4b17_branch2a = self.res4b17_branch2a(res4b16_relu)
        bn4b17_branch2a = self.bn4b17_branch2a(res4b17_branch2a)
        res4b17_branch2a_relu = self.relu_wrapper(bn4b17_branch2a)
        res4b17_branch2b_pad = self.pad_instances[28](res4b17_branch2a_relu)
        res4b17_branch2b = self.res4b17_branch2b(res4b17_branch2b_pad)
        bn4b17_branch2b = self.bn4b17_branch2b(res4b17_branch2b)
        res4b17_branch2b_relu = self.relu_wrapper(bn4b17_branch2b)
        res4b17_branch2c = self.res4b17_branch2c(res4b17_branch2b_relu)
        bn4b17_branch2c = self.bn4b17_branch2c(res4b17_branch2c)
        res4b17 = self.add_module_(res4b16_relu, bn4b17_branch2c)
        res4b17_relu = self.relu_wrapper(res4b17)
        res4b18_branch2a = self.res4b18_branch2a(res4b17_relu)
        bn4b18_branch2a = self.bn4b18_branch2a(res4b18_branch2a)
        res4b18_branch2a_relu = self.relu_wrapper(bn4b18_branch2a)
        res4b18_branch2b_pad = self.pad_instances[29](res4b18_branch2a_relu)
        res4b18_branch2b = self.res4b18_branch2b(res4b18_branch2b_pad)
        bn4b18_branch2b = self.bn4b18_branch2b(res4b18_branch2b)
        res4b18_branch2b_relu = self.relu_wrapper(bn4b18_branch2b)
        res4b18_branch2c = self.res4b18_branch2c(res4b18_branch2b_relu)
        bn4b18_branch2c = self.bn4b18_branch2c(res4b18_branch2c)
        res4b18 = self.add_module_(res4b17_relu, bn4b18_branch2c)
        res4b18_relu = self.relu_wrapper(res4b18)
        res4b19_branch2a = self.res4b19_branch2a(res4b18_relu)
        bn4b19_branch2a = self.bn4b19_branch2a(res4b19_branch2a)
        res4b19_branch2a_relu = self.relu_wrapper(bn4b19_branch2a)
        res4b19_branch2b_pad = self.pad_instances[30](res4b19_branch2a_relu)
        res4b19_branch2b = self.res4b19_branch2b(res4b19_branch2b_pad)
        bn4b19_branch2b = self.bn4b19_branch2b(res4b19_branch2b)
        res4b19_branch2b_relu = self.relu_wrapper(bn4b19_branch2b)
        res4b19_branch2c = self.res4b19_branch2c(res4b19_branch2b_relu)
        bn4b19_branch2c = self.bn4b19_branch2c(res4b19_branch2c)
        res4b19 = self.add_module_(res4b18_relu, bn4b19_branch2c)
        res4b19_relu = self.relu_wrapper(res4b19)
        res4b20_branch2a = self.res4b20_branch2a(res4b19_relu)
        bn4b20_branch2a = self.bn4b20_branch2a(res4b20_branch2a)
        res4b20_branch2a_relu = self.relu_wrapper(bn4b20_branch2a)
        res4b20_branch2b_pad = self.pad_instances[31](res4b20_branch2a_relu)
        res4b20_branch2b = self.res4b20_branch2b(res4b20_branch2b_pad)
        bn4b20_branch2b = self.bn4b20_branch2b(res4b20_branch2b)
        res4b20_branch2b_relu = self.relu_wrapper(bn4b20_branch2b)
        res4b20_branch2c = self.res4b20_branch2c(res4b20_branch2b_relu)
        bn4b20_branch2c = self.bn4b20_branch2c(res4b20_branch2c)
        res4b20 = self.add_module_(res4b19_relu, bn4b20_branch2c)
        res4b20_relu = self.relu_wrapper(res4b20)
        res4b21_branch2a = self.res4b21_branch2a(res4b20_relu)
        bn4b21_branch2a = self.bn4b21_branch2a(res4b21_branch2a)
        res4b21_branch2a_relu = self.relu_wrapper(bn4b21_branch2a)
        res4b21_branch2b_pad = self.pad_instances[32](res4b21_branch2a_relu)
        res4b21_branch2b = self.res4b21_branch2b(res4b21_branch2b_pad)
        bn4b21_branch2b = self.bn4b21_branch2b(res4b21_branch2b)
        res4b21_branch2b_relu = self.relu_wrapper(bn4b21_branch2b)
        res4b21_branch2c = self.res4b21_branch2c(res4b21_branch2b_relu)
        bn4b21_branch2c = self.bn4b21_branch2c(res4b21_branch2c)
        res4b21 = self.add_module_(res4b20_relu, bn4b21_branch2c)
        res4b21_relu = self.relu_wrapper(res4b21)
        res4b22_branch2a = self.res4b22_branch2a(res4b21_relu)
        bn4b22_branch2a = self.bn4b22_branch2a(res4b22_branch2a)
        res4b22_branch2a_relu = self.relu_wrapper(bn4b22_branch2a)
        res4b22_branch2b_pad = self.pad_instances[33](res4b22_branch2a_relu)
        res4b22_branch2b = self.res4b22_branch2b(res4b22_branch2b_pad)
        bn4b22_branch2b = self.bn4b22_branch2b(res4b22_branch2b)
        res4b22_branch2b_relu = self.relu_wrapper(bn4b22_branch2b)
        res4b22_branch2c = self.res4b22_branch2c(res4b22_branch2b_relu)
        bn4b22_branch2c = self.bn4b22_branch2c(res4b22_branch2c)
        res4b22 = self.add_module_(res4b21_relu, bn4b22_branch2c)
        res4b22_relu = self.relu_wrapper(res4b22)
        res4b23_branch2a = self.res4b23_branch2a(res4b22_relu)
        bn4b23_branch2a = self.bn4b23_branch2a(res4b23_branch2a)
        res4b23_branch2a_relu = self.relu_wrapper(bn4b23_branch2a)
        res4b23_branch2b_pad = self.pad_instances[34](res4b23_branch2a_relu)
        res4b23_branch2b = self.res4b23_branch2b(res4b23_branch2b_pad)
        bn4b23_branch2b = self.bn4b23_branch2b(res4b23_branch2b)
        res4b23_branch2b_relu = self.relu_wrapper(bn4b23_branch2b)
        res4b23_branch2c = self.res4b23_branch2c(res4b23_branch2b_relu)
        bn4b23_branch2c = self.bn4b23_branch2c(res4b23_branch2c)
        res4b23 = self.add_module_(res4b22_relu, bn4b23_branch2c)
        res4b23_relu = self.relu_wrapper(res4b23)
        res4b24_branch2a = self.res4b24_branch2a(res4b23_relu)
        bn4b24_branch2a = self.bn4b24_branch2a(res4b24_branch2a)
        res4b24_branch2a_relu = self.relu_wrapper(bn4b24_branch2a)
        res4b24_branch2b_pad = self.pad_instances[35](res4b24_branch2a_relu)
        res4b24_branch2b = self.res4b24_branch2b(res4b24_branch2b_pad)
        bn4b24_branch2b = self.bn4b24_branch2b(res4b24_branch2b)
        res4b24_branch2b_relu = self.relu_wrapper(bn4b24_branch2b)
        res4b24_branch2c = self.res4b24_branch2c(res4b24_branch2b_relu)
        bn4b24_branch2c = self.bn4b24_branch2c(res4b24_branch2c)
        res4b24 = self.add_module_(res4b23_relu, bn4b24_branch2c)
        res4b24_relu = self.relu_wrapper(res4b24)
        res4b25_branch2a = self.res4b25_branch2a(res4b24_relu)
        bn4b25_branch2a = self.bn4b25_branch2a(res4b25_branch2a)
        res4b25_branch2a_relu = self.relu_wrapper(bn4b25_branch2a)
        res4b25_branch2b_pad = self.pad_instances[36](res4b25_branch2a_relu)
        res4b25_branch2b = self.res4b25_branch2b(res4b25_branch2b_pad)
        bn4b25_branch2b = self.bn4b25_branch2b(res4b25_branch2b)
        res4b25_branch2b_relu = self.relu_wrapper(bn4b25_branch2b)
        res4b25_branch2c = self.res4b25_branch2c(res4b25_branch2b_relu)
        bn4b25_branch2c = self.bn4b25_branch2c(res4b25_branch2c)
        res4b25 = self.add_module_(res4b24_relu, bn4b25_branch2c)
        res4b25_relu = self.relu_wrapper(res4b25)
        res4b26_branch2a = self.res4b26_branch2a(res4b25_relu)
        bn4b26_branch2a = self.bn4b26_branch2a(res4b26_branch2a)
        res4b26_branch2a_relu = self.relu_wrapper(bn4b26_branch2a)
        res4b26_branch2b_pad = self.pad_instances[37](res4b26_branch2a_relu)
        res4b26_branch2b = self.res4b26_branch2b(res4b26_branch2b_pad)
        bn4b26_branch2b = self.bn4b26_branch2b(res4b26_branch2b)
        res4b26_branch2b_relu = self.relu_wrapper(bn4b26_branch2b)
        res4b26_branch2c = self.res4b26_branch2c(res4b26_branch2b_relu)
        bn4b26_branch2c = self.bn4b26_branch2c(res4b26_branch2c)
        res4b26 = self.add_module_(res4b25_relu, bn4b26_branch2c)
        res4b26_relu = self.relu_wrapper(res4b26)
        res4b27_branch2a = self.res4b27_branch2a(res4b26_relu)
        bn4b27_branch2a = self.bn4b27_branch2a(res4b27_branch2a)
        res4b27_branch2a_relu = self.relu_wrapper(bn4b27_branch2a)
        res4b27_branch2b_pad = self.pad_instances[38](res4b27_branch2a_relu)
        res4b27_branch2b = self.res4b27_branch2b(res4b27_branch2b_pad)
        bn4b27_branch2b = self.bn4b27_branch2b(res4b27_branch2b)
        res4b27_branch2b_relu = self.relu_wrapper(bn4b27_branch2b)
        res4b27_branch2c = self.res4b27_branch2c(res4b27_branch2b_relu)
        bn4b27_branch2c = self.bn4b27_branch2c(res4b27_branch2c)
        res4b27 = self.add_module_(res4b26_relu, bn4b27_branch2c)
        res4b27_relu = self.relu_wrapper(res4b27)
        res4b28_branch2a = self.res4b28_branch2a(res4b27_relu)
        bn4b28_branch2a = self.bn4b28_branch2a(res4b28_branch2a)
        res4b28_branch2a_relu = self.relu_wrapper(bn4b28_branch2a)
        res4b28_branch2b_pad = self.pad_instances[39](res4b28_branch2a_relu)
        res4b28_branch2b = self.res4b28_branch2b(res4b28_branch2b_pad)
        bn4b28_branch2b = self.bn4b28_branch2b(res4b28_branch2b)
        res4b28_branch2b_relu = self.relu_wrapper(bn4b28_branch2b)
        res4b28_branch2c = self.res4b28_branch2c(res4b28_branch2b_relu)
        bn4b28_branch2c = self.bn4b28_branch2c(res4b28_branch2c)
        res4b28 = self.add_module_(res4b27_relu, bn4b28_branch2c)
        res4b28_relu = self.relu_wrapper(res4b28)
        res4b29_branch2a = self.res4b29_branch2a(res4b28_relu)
        bn4b29_branch2a = self.bn4b29_branch2a(res4b29_branch2a)
        res4b29_branch2a_relu = self.relu_wrapper(bn4b29_branch2a)
        res4b29_branch2b_pad = self.pad_instances[40](res4b29_branch2a_relu)
        res4b29_branch2b = self.res4b29_branch2b(res4b29_branch2b_pad)
        bn4b29_branch2b = self.bn4b29_branch2b(res4b29_branch2b)
        res4b29_branch2b_relu = self.relu_wrapper(bn4b29_branch2b)
        res4b29_branch2c = self.res4b29_branch2c(res4b29_branch2b_relu)
        bn4b29_branch2c = self.bn4b29_branch2c(res4b29_branch2c)
        res4b29 = self.add_module_(res4b28_relu, bn4b29_branch2c)
        res4b29_relu = self.relu_wrapper(res4b29)
        res4b30_branch2a = self.res4b30_branch2a(res4b29_relu)
        bn4b30_branch2a = self.bn4b30_branch2a(res4b30_branch2a)
        res4b30_branch2a_relu = self.relu_wrapper(bn4b30_branch2a)
        res4b30_branch2b_pad = self.pad_instances[41](res4b30_branch2a_relu)
        res4b30_branch2b = self.res4b30_branch2b(res4b30_branch2b_pad)
        bn4b30_branch2b = self.bn4b30_branch2b(res4b30_branch2b)
        res4b30_branch2b_relu = self.relu_wrapper(bn4b30_branch2b)
        res4b30_branch2c = self.res4b30_branch2c(res4b30_branch2b_relu)
        bn4b30_branch2c = self.bn4b30_branch2c(res4b30_branch2c)
        res4b30 = self.add_module_(res4b29_relu, bn4b30_branch2c)
        res4b30_relu = self.relu_wrapper(res4b30)
        res4b31_branch2a = self.res4b31_branch2a(res4b30_relu)
        bn4b31_branch2a = self.bn4b31_branch2a(res4b31_branch2a)
        res4b31_branch2a_relu = self.relu_wrapper(bn4b31_branch2a)
        res4b31_branch2b_pad = self.pad_instances[42](res4b31_branch2a_relu)
        res4b31_branch2b = self.res4b31_branch2b(res4b31_branch2b_pad)
        bn4b31_branch2b = self.bn4b31_branch2b(res4b31_branch2b)
        res4b31_branch2b_relu = self.relu_wrapper(bn4b31_branch2b)
        res4b31_branch2c = self.res4b31_branch2c(res4b31_branch2b_relu)
        bn4b31_branch2c = self.bn4b31_branch2c(res4b31_branch2c)
        res4b31 = self.add_module_(res4b30_relu, bn4b31_branch2c)
        res4b31_relu = self.relu_wrapper(res4b31)
        res4b32_branch2a = self.res4b32_branch2a(res4b31_relu)
        bn4b32_branch2a = self.bn4b32_branch2a(res4b32_branch2a)
        res4b32_branch2a_relu = self.relu_wrapper(bn4b32_branch2a)
        res4b32_branch2b_pad = self.pad_instances[43](res4b32_branch2a_relu)
        res4b32_branch2b = self.res4b32_branch2b(res4b32_branch2b_pad)
        bn4b32_branch2b = self.bn4b32_branch2b(res4b32_branch2b)
        res4b32_branch2b_relu = self.relu_wrapper(bn4b32_branch2b)
        res4b32_branch2c = self.res4b32_branch2c(res4b32_branch2b_relu)
        bn4b32_branch2c = self.bn4b32_branch2c(res4b32_branch2c)
        res4b32 = self.add_module_(res4b31_relu, bn4b32_branch2c)
        res4b32_relu = self.relu_wrapper(res4b32)
        res4b33_branch2a = self.res4b33_branch2a(res4b32_relu)
        bn4b33_branch2a = self.bn4b33_branch2a(res4b33_branch2a)
        res4b33_branch2a_relu = self.relu_wrapper(bn4b33_branch2a)
        res4b33_branch2b_pad = self.pad_instances[44](res4b33_branch2a_relu)
        res4b33_branch2b = self.res4b33_branch2b(res4b33_branch2b_pad)
        bn4b33_branch2b = self.bn4b33_branch2b(res4b33_branch2b)
        res4b33_branch2b_relu = self.relu_wrapper(bn4b33_branch2b)
        res4b33_branch2c = self.res4b33_branch2c(res4b33_branch2b_relu)
        bn4b33_branch2c = self.bn4b33_branch2c(res4b33_branch2c)
        res4b33 = self.add_module_(res4b32_relu, bn4b33_branch2c)
        res4b33_relu = self.relu_wrapper(res4b33)
        res4b34_branch2a = self.res4b34_branch2a(res4b33_relu)
        bn4b34_branch2a = self.bn4b34_branch2a(res4b34_branch2a)
        res4b34_branch2a_relu = self.relu_wrapper(bn4b34_branch2a)
        res4b34_branch2b_pad = self.pad_instances[45](res4b34_branch2a_relu)
        res4b34_branch2b = self.res4b34_branch2b(res4b34_branch2b_pad)
        bn4b34_branch2b = self.bn4b34_branch2b(res4b34_branch2b)
        res4b34_branch2b_relu = self.relu_wrapper(bn4b34_branch2b)
        res4b34_branch2c = self.res4b34_branch2c(res4b34_branch2b_relu)
        bn4b34_branch2c = self.bn4b34_branch2c(res4b34_branch2c)
        res4b34 = self.add_module_(res4b33_relu, bn4b34_branch2c)
        res4b34_relu = self.relu_wrapper(res4b34)
        res4b35_branch2a = self.res4b35_branch2a(res4b34_relu)
        bn4b35_branch2a = self.bn4b35_branch2a(res4b35_branch2a)
        res4b35_branch2a_relu = self.relu_wrapper(bn4b35_branch2a)
        res4b35_branch2b_pad = self.pad_instances[46](res4b35_branch2a_relu)
        res4b35_branch2b = self.res4b35_branch2b(res4b35_branch2b_pad)
        bn4b35_branch2b = self.bn4b35_branch2b(res4b35_branch2b)
        res4b35_branch2b_relu = self.relu_wrapper(bn4b35_branch2b)
        res4b35_branch2c = self.res4b35_branch2c(res4b35_branch2b_relu)
        bn4b35_branch2c = self.bn4b35_branch2c(res4b35_branch2c)
        res4b35 = self.add_module_(res4b34_relu, bn4b35_branch2c)
        res4b35_relu = self.relu_wrapper(res4b35)
        res5a_branch1 = self.res5a_branch1(res4b35_relu)
        res5a_branch2a = self.res5a_branch2a(res4b35_relu)
        bn5a_branch1 = self.bn5a_branch1(res5a_branch1)
        bn5a_branch2a = self.bn5a_branch2a(res5a_branch2a)
        res5a_branch2a_relu = self.relu_wrapper(bn5a_branch2a)
        res5a_branch2b_pad = self.pad_instances[47](res5a_branch2a_relu)
        res5a_branch2b = self.res5a_branch2b(res5a_branch2b_pad)
        bn5a_branch2b = self.bn5a_branch2b(res5a_branch2b)
        res5a_branch2b_relu = self.relu_wrapper(bn5a_branch2b)
        res5a_branch2c = self.res5a_branch2c(res5a_branch2b_relu)
        bn5a_branch2c = self.bn5a_branch2c(res5a_branch2c)
        res5a = self.add_module_(bn5a_branch1, bn5a_branch2c)
        res5a_relu = self.relu_wrapper(res5a)
        res5b_branch2a = self.res5b_branch2a(res5a_relu)
        bn5b_branch2a = self.bn5b_branch2a(res5b_branch2a)
        res5b_branch2a_relu = self.relu_wrapper(bn5b_branch2a)
        res5b_branch2b_pad = self.pad_instances[48](res5b_branch2a_relu)
        res5b_branch2b = self.res5b_branch2b(res5b_branch2b_pad)
        bn5b_branch2b = self.bn5b_branch2b(res5b_branch2b)
        res5b_branch2b_relu = self.relu_wrapper(bn5b_branch2b)
        res5b_branch2c = self.res5b_branch2c(res5b_branch2b_relu)
        bn5b_branch2c = self.bn5b_branch2c(res5b_branch2c)
        res5b = self.add_module_(res5a_relu, bn5b_branch2c)
        res5b_relu = self.relu_wrapper(res5b)
        res5c_branch2a = self.res5c_branch2a(res5b_relu)
        bn5c_branch2a = self.bn5c_branch2a(res5c_branch2a)
        res5c_branch2a_relu = self.relu_wrapper(bn5c_branch2a)
        res5c_branch2b_pad = self.pad_instances[49](res5c_branch2a_relu)
        res5c_branch2b = self.res5c_branch2b(res5c_branch2b_pad)
        bn5c_branch2b = self.bn5c_branch2b(res5c_branch2b)
        res5c_branch2b_relu = self.relu_wrapper(bn5c_branch2b)
        res5c_branch2c = self.res5c_branch2c(res5c_branch2b_relu)
        bn5c_branch2c = self.bn5c_branch2c(res5c_branch2c)
        res5c = self.add_module_(res5b_relu, bn5c_branch2c)
        res5c_relu = self.relu_wrapper(res5c)
        pool5 = self.avgpool5(res5c_relu)
        fc365_0 = pool5.view(pool5.size(0), -1)
        fc365_1 = self.fc365_1(fc365_0)
        prob = F.softmax(fc365_1)
        # return prob
        return pool5, prob

    def _build_features_list(self):
        """
        Build complete features list with all operations in execution order
        Similar to VGG's create_features_modules, but for ResNet architecture
        """
        features = []

        # Initial layers
        features.append(self.pad_conv1)
        features.append(self.conv1)
        features.append(self.bn_conv1)
        features.append(self.relu_conv1)
        features.append(self.pad_pool1)
        features.append(self.maxpool1)

        # Res2a block
        features.append(self.res2a_branch1)
        features.append(self.res2a_branch2a)
        features.append(self.bn2a_branch1)
        features.append(self.bn2a_branch2a)
        features.append(self.relu_wrapper)  # res2a_branch2a_relu
        features.append(self.pad_instances[0])  # res2a_branch2b_pad
        features.append(self.res2a_branch2b)
        features.append(self.bn2a_branch2b)
        features.append(self.relu_wrapper)  # res2a_branch2b_relu
        features.append(self.res2a_branch2c)
        features.append(self.bn2a_branch2c)
        features.append(self.add_module_)  # res2a
        features.append(self.relu_wrapper)  # res2a_relu

        # Res2b block
        features.append(self.res2b_branch2a)
        features.append(self.bn2b_branch2a)
        features.append(self.relu_wrapper)
        features.append(self.pad_instances[1])
        features.append(self.res2b_branch2b)
        features.append(self.bn2b_branch2b)
        features.append(self.relu_wrapper)
        features.append(self.res2b_branch2c)
        features.append(self.bn2b_branch2c)
        features.append(self.add_module_)  # res2b
        features.append(self.relu_wrapper)  # res2b_relu

        # Res2c block
        features.append(self.res2c_branch2a)
        features.append(self.bn2c_branch2a)
        features.append(self.relu_wrapper)
        features.append(self.pad_instances[2])
        features.append(self.res2c_branch2b)
        features.append(self.bn2c_branch2b)
        features.append(self.relu_wrapper)
        features.append(self.res2c_branch2c)
        features.append(self.bn2c_branch2c)
        features.append(self.add_module_)  # res2c
        features.append(self.relu_wrapper)  # res2c_relu

        # Res3a block
        features.append(self.res3a_branch1)
        features.append(self.res3a_branch2a)
        features.append(self.bn3a_branch1)
        features.append(self.bn3a_branch2a)
        features.append(self.relu_wrapper)
        features.append(self.pad_instances[3])
        features.append(self.res3a_branch2b)
        features.append(self.bn3a_branch2b)
        features.append(self.relu_wrapper)
        features.append(self.res3a_branch2c)
        features.append(self.bn3a_branch2c)
        features.append(self.add_module_)  # res3a
        features.append(self.relu_wrapper)  # res3a_relu

        # Res3b1 through res3b7 blocks (7 blocks)
        pad_idx = 4
        for i in range(1, 8):
            features.append(getattr(self, f"res3b{i}_branch2a"))
            features.append(getattr(self, f"bn3b{i}_branch2a"))
            features.append(self.relu_wrapper)
            features.append(self.pad_instances[pad_idx])
            pad_idx += 1
            features.append(getattr(self, f"res3b{i}_branch2b"))
            features.append(getattr(self, f"bn3b{i}_branch2b"))
            features.append(self.relu_wrapper)
            features.append(getattr(self, f"res3b{i}_branch2c"))
            features.append(getattr(self, f"bn3b{i}_branch2c"))
            features.append(self.add_module_)
            features.append(self.relu_wrapper)

        # Res4a block
        features.append(self.res4a_branch1)
        features.append(self.res4a_branch2a)
        features.append(self.bn4a_branch1)
        features.append(self.bn4a_branch2a)
        features.append(self.relu_wrapper)
        features.append(self.pad_instances[pad_idx])
        pad_idx += 1
        features.append(self.res4a_branch2b)
        features.append(self.bn4a_branch2b)
        features.append(self.relu_wrapper)
        features.append(self.res4a_branch2c)
        features.append(self.bn4a_branch2c)
        features.append(self.add_module_)  # res4a
        features.append(self.relu_wrapper)  # res4a_relu

        # Res4b1 through res4b35 blocks (35 blocks)
        for i in range(1, 36):
            features.append(getattr(self, f"res4b{i}_branch2a"))
            features.append(getattr(self, f"bn4b{i}_branch2a"))
            features.append(self.relu_wrapper)
            features.append(self.pad_instances[pad_idx])
            pad_idx += 1
            features.append(getattr(self, f"res4b{i}_branch2b"))
            features.append(getattr(self, f"bn4b{i}_branch2b"))
            features.append(self.relu_wrapper)
            features.append(getattr(self, f"res4b{i}_branch2c"))
            features.append(getattr(self, f"bn4b{i}_branch2c"))
            features.append(self.add_module_)
            features.append(self.relu_wrapper)

        # Res5a block
        features.append(self.res5a_branch1)
        features.append(self.res5a_branch2a)
        features.append(self.bn5a_branch1)
        features.append(self.bn5a_branch2a)
        features.append(self.relu_wrapper)
        features.append(self.pad_instances[pad_idx])
        pad_idx += 1
        features.append(self.res5a_branch2b)
        features.append(self.bn5a_branch2b)
        features.append(self.relu_wrapper)
        features.append(self.res5a_branch2c)
        features.append(self.bn5a_branch2c)
        features.append(self.add_module_)  # res5a
        features.append(self.relu_wrapper)  # res5a_relu

        # Res5b block
        features.append(self.res5b_branch2a)
        features.append(self.bn5b_branch2a)
        features.append(self.relu_wrapper)
        features.append(self.pad_instances[pad_idx])
        pad_idx += 1
        features.append(self.res5b_branch2b)
        features.append(self.bn5b_branch2b)
        features.append(self.relu_wrapper)
        features.append(self.res5b_branch2c)
        features.append(self.bn5b_branch2c)
        features.append(self.add_module_)  # res5b
        features.append(self.relu_wrapper)  # res5b_relu

        # Res5c block
        features.append(self.res5c_branch2a)
        features.append(self.bn5c_branch2a)
        features.append(self.relu_wrapper)
        features.append(self.pad_instances[pad_idx])
        pad_idx += 1
        features.append(self.res5c_branch2b)
        features.append(self.bn5c_branch2b)
        features.append(self.relu_wrapper)
        features.append(self.res5c_branch2c)
        features.append(self.bn5c_branch2c)
        features.append(self.add_module_)  # res5c
        features.append(self.relu_wrapper)  # res5c_relu

        # Final layers
        features.append(self.avgpool5)
        features.append(self.fc365_1)

        return features

    def improve_resolution(self, I, target_layer):
        """
        Improve resolution by propagating I backwards through layers
        Similar to VGG's improve_resolution, using self.features
        target_layer: index in features to stop at
        """
        # Similar to VGG: iterate backwards through features
        # Skip FC (Linear) layers since they have incompatible shapes with conv explanations
        for i in range(len(self.features) - 1, target_layer, -1):
            layer = self.features[i]
            # Skip Linear (FC) layers - they expect flattened input, not spatial features
            if isinstance(layer, Linear):
                continue
            if hasattr(layer, "IR"):
                try:
                    I = layer.IR(I)
                except RuntimeError as e:
                    # If shape mismatch, skip this layer (likely FC or incompatible layer)
                    if "size" in str(e).lower() or "shape" in str(e).lower():
                        continue
                    raise
        return I

    def register_hook(self):
        """Register forward hooks on all layers that need them (similar to VGG)"""
        self.hook_handles = []
        for m in self.features:
            if hasattr(m, "register_forward_hook"):
                handle = m.register_forward_hook(forward_hook)
                self.hook_handles.append(handle)

    def remove_hook(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if dim == 0 or dim == 1:
            layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:
            layer = BatchNorm2d(**kwargs)
        elif dim == 3:
            layer = nn.BatchNorm3d(**kwargs)
        else:
            raise NotImplementedError()

        if "scale" in _weights_dict[name]:
            layer.state_dict()["weight"].copy_(
                torch.from_numpy(_weights_dict[name]["scale"].squeeze())
            )
        else:
            layer.weight.data.fill_(1)

        if "bias" in _weights_dict[name]:
            layer.state_dict()["bias"].copy_(
                torch.from_numpy(_weights_dict[name]["bias"])
            )
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()["running_mean"].copy_(
            torch.from_numpy(_weights_dict[name]["mean"])
        )
        layer.state_dict()["running_var"].copy_(
            torch.from_numpy(_weights_dict[name]["var"])
        )
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()

        layer.state_dict()["weight"].copy_(
            torch.from_numpy(_weights_dict[name]["weights"])
        )
        if "bias" in _weights_dict[name]:
            layer.state_dict()["bias"].copy_(
                torch.from_numpy(_weights_dict[name]["bias"])
            )
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = Linear(**kwargs)
        layer.state_dict()["weight"].copy_(
            torch.from_numpy(_weights_dict[name]["weights"])
        )
        if "bias" in _weights_dict[name]:
            layer.state_dict()["bias"].copy_(
                torch.from_numpy(_weights_dict[name]["bias"])
            )
        return layer


if __name__ == "__main__":
    classes = list()
    class_file_namme = "/Users/ian/Project/VLN/R2R/checkpoints/categories_places365.txt"
    with open(class_file_namme) as class_file:
        for line in class_file:
            classes.append(line.strip().split(" ")[0][3:])
    classes = tuple(classes)

    wt_path = "/Users/ian/Project/VLN/R2R/30913b5b6a4c411bb1b6020f492e5862.npy"
    model = CNN(weight_file=wt_path)
    model.eval()

    # img_path = "/Users/ian/Downloads/dataset-cover.jpeg"
    # img_path = "/Users/ian/Downloads/IMG_0404.jpeg"
    img_path = "/Users/ian/Downloads/IMG_0494.jpeg"
    # img_path = "/Users/ian/Downloads/IMG_9736.jpeg"
    img_size = 224
    from PIL import Image
    import torch.nn.functional as F

    def ZeroCenter(path, size, BGRTranspose=False):
        img = Image.open(path)
        if isinstance(size, tuple):
            h, w = size[0], size[1]
        else:
            h, w = size, size
        img = img.resize((h, w))
        x = np.array(img, dtype=np.float32)

        # [103.1, 115.9, 123.2]
        x[..., 0] -= 123.2
        x[..., 1] -= 115.9
        x[..., 2] -= 103.1
        if BGRTranspose == True:
            x = x[..., ::-1]

        return x

    img = ZeroCenter(img_path, img_size, True)
    img = np.expand_dims(img, 0).copy()
    input_data = torch.from_numpy(img)
    # input_data = torch.cat([input_data, input_data])
    print(input_data.shape)  # (bs, 224, 224, 3)
    input_data = input_data.permute(0, 3, 1, 2)  # (bs, 3, 224, 224)
    data = torch.autograd.Variable(input_data, requires_grad=False)
    # do forward pass
    _, logit = model(data)
    h_x = F.softmax(logit, 1).data.squeeze()
    print(h_x.shape)
    probs, idx = h_x.sort(0, True)
    for i in range(0, 5):
        print("{:.3f} -> {}".format(probs[i], classes[idx[i]]))

    # # do explanation
    # explanations = explanation_model(model, data)
    # print(explanations.shape)
    # print(explanations)
