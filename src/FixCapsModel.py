import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformation for input images
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # Include any additional transformations as needed
])

class FixCapsNet(nn.Module):
    def __init__(self, num_classes=7):
        super(FixCapsNet, self).__init__()

        # Input image dimensions
        self.image_height = 128
        self.image_width = 128

        # Updated configurations for the network to handle RGB images
        conv_inputs = 3  # For RGB images
        conv_outputs = 64  # Reduced to manage memory
        primary_units = 8  # Number of primary capsules
        primary_caps_dim = 8  # Dimension of each primary capsule
        output_unit_size = 16  # Dimension of the output vector from each digit capsule

        # First convolutional layer
        self.convolution = nn.Conv2d(conv_inputs, conv_outputs, kernel_size=9, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(channel=conv_outputs)

        # Calculate the output size after the first convolution
        conv_output_size = self.compute_output_size(self.image_height, kernel_size=9, stride=1)

        # Primary Capsules Layer
        self.primary_caps = PrimaryCaps(
            in_channels=conv_outputs,
            num_capsules=primary_units,
            capsule_dim=primary_caps_dim,
            kernel_size=9,
            stride=2
        )

        # Calculate the output size after PrimaryCaps layer
        primary_caps_output_size = self.compute_output_size(conv_output_size, kernel_size=9, stride=2)

        # Total number of capsules after PrimaryCaps layer
        num_input_capsules = primary_units * primary_caps_output_size * primary_caps_output_size

        # Digit Capsules Layer
        self.digit_caps = DigitCaps(
            num_input_capsules=num_input_capsules,
            input_dim=primary_caps_dim,
            num_capsules=num_classes,
            capsule_dim=output_unit_size
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.convolution(x))
        x = self.cbam(x)
        x = self.primary_caps(x)  # Output shape: [batch_size, num_input_capsules, capsule_dim]
        x = self.digit_caps(x)    # Output shape: [batch_size, num_capsules, capsule_dim]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def compute_output_size(input_size, kernel_size, stride, padding=0):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, num_capsules, capsule_dim, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.capsules = nn.Conv2d(
            in_channels,
            num_capsules * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )

    def forward(self, x):
        x = self.capsules(x)  # Shape: [batch_size, num_capsules * capsule_dim, h, w]
        batch_size = x.size(0)
        h = x.size(2)
        w = x.size(3)
        x = x.view(batch_size, self.num_capsules, self.capsule_dim, h, w)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # Shape: [batch_size, num_capsules, h, w, capsule_dim]
        x = x.view(batch_size, self.num_capsules * h * w, self.capsule_dim)
        x = self.squash(x)
        return x  # Shape: [batch_size, num_input_capsules, capsule_dim]

    @staticmethod
    def squash(input_tensor):
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm + 1e-8)
        return output_tensor

class DigitCaps(nn.Module):
    def __init__(self, num_input_capsules, input_dim, num_capsules, capsule_dim, routing_iters=3):
        super(DigitCaps, self).__init__()
        self.num_capsules = num_capsules
        self.routing_iters = routing_iters
        self.W = nn.Parameter(
            0.01 * torch.randn(1, num_input_capsules, num_capsules, capsule_dim, input_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)  # Shape: [batch_size, num_input_capsules, 1, input_dim, 1]
        x = x.expand(batch_size, -1, self.num_capsules, -1, 1)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x)  # Shape: [batch_size, num_input_capsules, num_capsules, capsule_dim, 1]
        b_ij = torch.zeros(batch_size, x.size(1), self.num_capsules, 1, 1, device=x.device)

        for iteration in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)  # Softmax over num_capsules
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # Sum over num_input_capsules
            v_j = self.squash(s_j)
            if iteration < self.routing_iters - 1:
                delta_b_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
                b_ij = b_ij + delta_b_ij

        v_j = v_j.squeeze(1).squeeze(-1)  # Shape: [batch_size, num_capsules, capsule_dim]
        return v_j

    @staticmethod
    def squash(s):
        s_squared_norm = (s ** 2).sum(dim=-2, keepdim=True)
        scale = s_squared_norm / (1 + s_squared_norm)
        v = scale * s / torch.sqrt(s_squared_norm + 1e-8)
        return v

# CBAM components (ChannelAttention, SpatialAttention, CBAM) remain the same as before

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooled = self.avg_pool(x).view(x.size(0), -1)
        max_pooled = self.max_pool(x).view(x.size(0), -1)
        avg_out = self.fc(avg_pooled)
        max_out = self.fc(max_pooled)
        out = avg_out + max_out
        out = out.view(x.size(0), x.size(1), 1, 1)
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x_cat)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channel, reduction_ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = self.channel_attention(x) * x
        x_out = self.spatial_attention(x_out) * x_out
        return x_out

# Example usage:
# model = FixCapsNet(num_classes=7)
# output = model(torch.randn(16, 3, 128, 128))  # For a batch of 16 images
