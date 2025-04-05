import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, depth=20, channels=64):
        super(DnCNN, self).__init__()

        layers = [
            nn.Conv2d(
                in_channels=3,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        ]

        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels=channels,
                out_channels=3,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        )

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return noise
