import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ch, out_ch=None, k=3, s=1, p=1, groups=32):
        super().__init__()
        out_ch = out_ch or ch
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(groups, ch)
        self.conv = nn.Conv2d(ch, out_ch, k, s, p, padding_mode='replicate')

    def forward(self, x):
        return self.conv(self.norm(self.act(x)))


class ConvTransposeBlock(ConvBlock):
    def __init__(self, ch, out_ch=None, k=3, s=1, p=1, groups=32):
        super().__init__(ch, out_ch, k, s, p, groups)
        self.conv = nn.ConvTranspose2d(ch, out_ch, k, s, p)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block1 = ConvBlock(ch)
        self.block2 = ConvBlock(ch)

    def forward(self, x):
        return x + self.block2(self.block1(x))


class AutoEncoder(nn.Module):
    def __init__(self, ch=3, dim=128, mults=(1, 2, 4), z_dim=4):
        super().__init__()
        self.dim = dim

        encoder = [nn.Conv2d(ch, dim, 3, 1, 1, padding_mode='replicate')]
        for m in mults:
            encoder.append(ConvBlock(dim, self.dim * m, k=4, s=2))
            dim = self.dim * m
            encoder.append(ResBlock(dim))
            encoder.append(ResBlock(dim))
        encoder.append(nn.Conv2d(dim, z_dim, 1))

        decoder = [nn.Conv2d(z_dim, dim, 1)]
        for m in reversed(mults):
            decoder.append(ConvTransposeBlock(dim, self.dim * m, k=4, s=2, p=1))
            dim = self.dim * m
            decoder.append(ResBlock(dim))
            decoder.append(ResBlock(dim))
        decoder.append(nn.Conv2d(dim, ch, 3, 1, 1, padding_mode='replicate'))

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    cls = AutoEncoder()
    img = torch.rand(1, 3, 96, 96)
    print(cls)
    print(cls(img).shape)

