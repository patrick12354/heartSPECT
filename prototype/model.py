"""
3D U-Net model for left ventricle segmentation on cardiac SPECT imaging.
Architecture: Encoder (16->32->64->128) + Bottleneck (256) + Decoder + Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Double Conv: (Conv3D -> BN -> ReLU) x 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D U-Net for left ventricle SPECT segmentation.

    Args:
        in_channels  : Input channels (default 1 = grayscale)
        out_channels : Output channels (default 1 = binary mask)
        features     : Filter counts at each encoder level
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: list = [16, 32, 64, 128]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock3D(ch, f))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            ch = f

        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)

        rev_features = features[::-1]
        for f in rev_features:
            self.upconvs.append(
                nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock3D(f * 2, f))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for upconv, decoder, skip in zip(self.upconvs, self.decoders,
                                          skip_connections):
            x = upconv(x)
            if x.shape != skip.shape:
                diff = [s - x_ for s, x_ in zip(skip.shape[2:], x.shape[2:])]
                x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return torch.sigmoid(self.final_conv(x))


def load_model(checkpoint_path: str, device: torch.device) -> UNet3D:
    """Load trained UNet3D from checkpoint."""
    model = UNet3D(in_channels=1, out_channels=1, features=[16, 32, 64, 128]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    info = {
        'epoch': ckpt.get('epoch', 'N/A'),
        'val_dice': ckpt.get('val_dice', 'N/A'),
    }
    return model, info
