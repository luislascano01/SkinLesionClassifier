import torch
import torch.nn as nn

class LargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LargeKernelConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=31, stride=1, padding=15)

    def forward(self, x):
        return self.conv(x)
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, num_channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(num_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, num_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2) for _ in range(num_capsules)]
            )

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = torch.zeros(*priors.size()).to(x.device)
            for i in range(self.num_iterations):
                probs = torch.softmax(logits, dim=2)
                outputs = squash((probs * priors).sum(dim=2, keepdim=True))
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1)
                    logits = logits + delta_logits
            return outputs.squeeze(3)
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            return squash(outputs)

def squash(input_tensor):
    squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * input_tensor / torch.sqrt(squared_norm)
class FixCapsNet(nn.Module):
    def __init__(self):
        super(FixCapsNet, self).__init__()
        self.large_kernel_conv = LargeKernelConv(3, 256)
        self.cbam = CBAM(256)
        self.capsule_layer = CapsuleLayer(num_capsules=10, num_route_nodes=-1, in_channels=256, out_channels=32, kernel_size=9, stride=2)

    def forward(self, x):
        x = self.large_kernel_conv(x)
        x = self.cbam(x)
        x = self.capsule_layer(x)
        return x
