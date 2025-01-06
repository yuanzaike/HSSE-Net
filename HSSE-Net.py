import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from Transformer1 import ViT
from Transformer2 import SpeT

#将序列张量重新排列为图像张量
def seq2img(x):
    [b, c, d] = x.shape
    p = int(d ** .5)
    x = x.reshape((b, c, p, p))
    return x


class SpatialEnhancement(nn.Module):
    def __init__(self, in_channels, inter_channels=None, size=None):
        super(SpatialEnhancement, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # Default to half the in_channels if inter_channels is not specified
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # Convolutional layer and batch normalization
        conv_layer = nn.Conv2d
        batch_norm = nn.BatchNorm2d

        # Spatial transformation network
        self.transform = conv_layer(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.reconstruction = nn.Sequential(
            conv_layer(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            batch_norm(self.in_channels)
        )

        # Transformation 1 and Transformation 2 for feature extraction
        self.transformation1 = nn.Sequential(
            conv_layer(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            batch_norm(self.inter_channels),
            nn.Sigmoid()
        )
        self.transformation2 = nn.Sequential(
            conv_layer(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            batch_norm(self.inter_channels),
            nn.Sigmoid()
        )

        # Dimensionality reduction layer
        self.dim_reduction = nn.Sequential(
            nn.Conv1d(
                in_channels=size * size,
                out_channels=1,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, x1, x2):
        """
        Args:
            x1: (N, C, H, W)
            x2: (N, C, H, W)
        """
        batch_size = x1.size(0)

        # Apply transformations and reshape the outputs
        t1 = self.transformation1(x1).view(batch_size, self.inter_channels, -1)
        t2 = self.transformation2(x2).view(batch_size, self.inter_channels, -1)

        # Compute affinity matrix by performing matrix multiplication
        t1 = t1.permute(0, 2, 1)
        affinity_matrix = torch.matmul(t1, t2)
        affinity_matrix = affinity_matrix.permute(0, 2, 1)  # B * HW * TF --> B * TF * HW

        # Reduce dimensionality of the affinity matrix
        affinity_matrix = self.dim_reduction(affinity_matrix)  # B * 1 * HW
        affinity_matrix = affinity_matrix.view(batch_size, 1, x1.size(2), x1.size(3))  # B * 1 * H * W

        # Apply the affinity matrix to x1
        x1 = x1 * affinity_matrix.expand_as(x1)

        return x1

class CNN_Encoder(nn.Module):
    def __init__(self, l1, l2, patch_size):
        super(CNN_Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(l1, 32, 3, 1, 1),
            #(输入通道，输出通道，卷积核大小，步长，填充)
            nn.BatchNorm2d(32),
            nn.ReLU(),  # No effect on order
            # nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # No effect on order
            # nn.MaxPool2d(2),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
            #表示在每个窗口上执行最大池化操作，这里是一个2x2的窗口
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.SAEM1 = SpatialEnhancement(in_channels=64, inter_channels=32,
                                           size=patch_size//2)
        self.SAEM2 = SpatialEnhancement(in_channels=64, inter_channels=32,
                                            size=(patch_size//2)*2)
        self.SAEM3 = SpatialEnhancement(in_channels=64, inter_channels=32,
                                           size=(patch_size//2)*3)
        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        # #nn.Parameter 创建的参数会在模型的反向传播过程中被更新
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda

    def forward(self, x11, x21, x12, x22, x13, x23):
        x11 = self.conv1(x11)
        x21 = self.conv2(x21)
        x12 = self.conv1(x12)
        x22 = self.conv2(x22)
        x13 = self.conv1(x13)
        x23 = self.conv2(x23)

        x1_1 = self.conv1_1(x11)
        x2_1 = self.conv2_1(x21)
        f1_1 = self.SAEM1(x1_1, x2_1)
        f2_1 = self.SAEM1(x2_1, x1_1)
        x_add1 = f1_1 * self.xishu1 + f2_1 * self.xishu2

        x1_2 = self.conv1_2(x12)
        x2_2 = self.conv2_2(x22)
        f1_2 = self.SAEM2(x1_2, x2_2)
        f2_2 = self.SAEM2(x2_2, x1_2)
        x_add2 = f1_2 * self.xishu1 + f2_2 * self.xishu2

        x1_3 = self.conv1_3(x13)
        x2_3 = self.conv2_3(x23)
        f1_3 = self.SAEM3(x1_3, x2_3)
        f2_3 = self.SAEM3(x2_3, x1_3)
        x_add3 = f1_3 * self.xishu1 + f2_3 * self.xishu2
        return x_add1, x_add2, x_add3


class CNN_Decoder(nn.Module):
    def __init__(self, l1, l2):
        super(CNN_Decoder, self).__init__()

        self.dconv1 = nn.Sequential(
            nn.Conv2d(64, l1, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(64, l2, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(64, l1, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(64, l2, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv5 = nn.Sequential(
            nn.Upsample(scale_factor=3),  # add Upsample
            nn.Conv2d(64, l1, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv6 = nn.Sequential(
            nn.Upsample(scale_factor=3),  # add Upsample
            nn.Conv2d(64, l2, 3, 1, 1),
            nn.Sigmoid(),

        )

    def forward(self, x_con1):
        x1 = self.dconv1(x_con1)
        x2 = self.dconv2(x_con1)

        x3 = self.dconv3(x_con1)
        x4 = self.dconv4(x_con1)

        x5 = self.dconv5(x_con1)
        x6 = self.dconv6(x_con1)
        return x1, x2, x3, x4, x5, x6


class CNN_Classifier(nn.Module):
    def __init__(self, Classes):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x_out = F.softmax(x, dim=1)
        return x_out


class Merge(nn.Module):
    def __init__(self, kernel_size=7):
        super(Merge, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, dim):
        p = int(x1.shape[2] ** .5)
        x1 = x1.reshape((x1.shape[0], x1.shape[1], p, p))
        x2 = x2.reshape((x2.shape[0], x2.shape[1], p, p))
        x3 = x3.reshape((x3.shape[0], x3.shape[1], p, p))

        num1 = x1.shape[1] // dim
        num2 = x2.shape[1] // dim
        num3 = x3.shape[1] // dim
        x_out = torch.empty(x1.shape[0], dim, p, p).cuda()
        for i in range(dim):
            x1_tmp = x1[:, i * num1:(i + 1) * num1, :, :]
            x2_tmp = x2[:, i * num2:(i + 1) * num2, :, :]
            x3_tmp = x3[:, i * num3:(i + 1) * num3, :, :]
            x_tmp1 = torch.cat((x1_tmp, x2_tmp, x3_tmp), dim=1)
            avgout = torch.mean(x_tmp1, dim=1, keepdim=True)
            maxout, _ = torch.max(x_tmp1, dim=1, keepdim=True)
            x_tmp2 = torch.cat([avgout, maxout], dim=1)
            x_tmp3 = self.conv(x_tmp2)
            x_tmp4 = self.sigmoid(x_tmp3)
            x_out[:, i:i+1, :, :] = x_tmp4
        x_out = x_out.reshape(x_out.shape[0], dim, p*p)
        return x_out


class HSSE(nn.Module):
    def __init__(self, l1, l2, patch_size, num_patches, num_classes, encoder_embed_dim, decoder_embed_dim, en_depth, en_heads,
                 de_depth, de_heads, mlp_dim, dim_head=16, dropout=0., emb_dropout=0.):
        super().__init__()
        self.cnn_encoder = CNN_Encoder(l1, l2, patch_size)
        self.cnn_decoder = CNN_Decoder(l1, l2)
        self.cnn_classifier = CNN_Classifier(num_classes)
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1/3]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1/3]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([1/3]))
        self.merge = Merge()
        self.loss_fun2 = nn.MSELoss()
        #计算模型的预测值与实际值之间的均方误差
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1, encoder_embed_dim))
        self.encoder_pos_embed2 = nn.Parameter(torch.randn(1, self.patch_size ** 2, encoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1, decoder_embed_dim))
        self.decoder_pos_embed2 = nn.Parameter(torch.randn(1, self.patch_size ** 2, decoder_embed_dim))
        self.encoder_embedding1 = nn.Linear(((patch_size // 2) * 1) ** 2, self.patch_size ** 2)
        #nn.Linear 是全连接层。((patch_size // 2) * 1) ** 2是输入的特征维度，self.patch_size ** 2是输出的特征维度
        self.encoder_embedding2 = nn.Linear(((patch_size // 2) * 2) ** 2, self.patch_size ** 2)
        self.encoder_embedding3 = nn.Linear(((patch_size // 2) * 3) ** 2, self.patch_size ** 2)
        self.decoder_embedding = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embedding = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.en_transformer = ViT(encoder_embed_dim, en_depth, en_heads, dim_head, mlp_dim, dropout,
                                          num_patches, mode='ViT')
        self.en_transformer2 = SpeT(encoder_embed_dim, en_depth, en_heads, dim_head, mlp_dim, dropout,
                                          num_patches, mode='ViT')
        self.de_transformer = ViT(decoder_embed_dim, de_depth, de_heads, dim_head, mlp_dim, dropout,
                                          num_patches, mode='ViT')
        self.de_transformer2 = SpeT(decoder_embed_dim, de_depth, de_heads, dim_head, mlp_dim, dropout,
                                          num_patches, mode='ViT')
        self.decoder_pred1 = nn.Linear(decoder_embed_dim, 64, bias=True)  # decoder to patch
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(encoder_embed_dim),
            nn.Linear(encoder_embed_dim, num_classes)
        )
        self.spapa = torch.nn.Parameter(torch.Tensor([0.5]))
        self.spepa = torch.nn.Parameter(torch.Tensor([0.5]))
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True)

        )

    def encoder(self, x11, x21, x12, x22, x13, x23):

        x_fuse1, x_fuse2, x_fuse3 = self.cnn_encoder(x11, x21, x12, x22, x13, x23)  # x_fuse1:64*64*8*8, x_fuse2:64*64*4*4, x_fuse2:64*64*2*2
        x_flat1 = x_fuse1.flatten(2)
        #将 x_fuse1 从维度 2 开始将所有的后续维度展平为一维。变为[batch_size, channels, height * width]
        x_flat2 = x_fuse2.flatten(2)
        x_flat3 = x_fuse3.flatten(2)

        x_1 = self.encoder_embedding1(x_flat1)
        x_2 = self.encoder_embedding2(x_flat2)
        x_3 = self.encoder_embedding3(x_flat3)
        #使用全连接层。将输入的特征进行线性变换，映射到指定的输出维度。
        x_cnn = self.merge(x_1, x_2, x_3, self.encoder_embed_dim)  # x_cnn:64*64*(p*p)


        x_spa = torch.einsum('ndl->nld', x_cnn)
        x_spe = torch.einsum('ndl->nld', x_cnn)
        b, n, _ = x_spa.shape
        # add pos embed w/o cls token
        x_spa = x_spa + self.encoder_pos_embed[:, 1:, :]
        x_spe = x_spe + self.encoder_pos_embed2

        # append cls token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  #repeat对self.cls_token进行扩展复制以匹配批次大小(b)
        # cls_tokens2 = repeat(self.cls_token2, '() n d -> b n d', b=b)
        x_spa = torch.cat((cls_tokens, x_spa), dim=1)
        # x_spe = torch.cat((cls_tokens2, x_spe), dim=1)
        # add position embedding
        x_spa += self.encoder_pos_embed[:, :1]
        # x_spe += self.encoder_pos_embed2[:, :1]
        x_spa = self.dropout(x_spa)
        x_spe = self.dropout(x_spe)

        x_spa = self.en_transformer(x_spa, mask=None)
        x_spe = self.en_transformer2(x_spe, mask=None)

        return x_spa, x_spe, x_cnn

    def classifier(self, x_spa, x_spe, x_cnn):
        # classification: using cls_token output
        x_spa = self.to_latent(x_spa[:, 0])
        x_spe = torch.einsum('ndl->nld', x_spe)
        # MLP classification layer
        x_cls1 = self.mlp_head(x_spa)
        x_cls2 = self.cnn_classifier(seq2img(x_spe))
        x_cls3 = self.cnn_classifier(seq2img(x_cnn))
        # x_cls2, x_cls3 = self.cnn_classifier(seq2img(x_spe), seq2img(x_cnn))

        x_cls = x_cls1 * self.coefficient1 + x_cls2 * self.coefficient2 + x_cls3 * self.coefficient3
        return x_cls

    def decoder(self, x_spa, x_spe, imgs11, imgs21, imgs12, imgs22, imgs13, imgs23):
        # embed tokens
        x_spa = self.decoder_embedding(x_spa)
        x_spe = self.decoder_embedding(x_spe)
        """ with or without decoder_pos_embed"""
        # add pos embed
        x_spa += self.decoder_pos_embed
        x_spe += self.decoder_pos_embed2

        x_spa = self.de_transformer(x_spa, mask=None)
        x_spe = self.de_transformer2(x_spe, mask=None)

        # predictor projection
        x_1 = self.decoder_pred1(x_spa)
        x_2 = self.decoder_pred1(x_spe)

        # remove cls token
        x_con1 = x_1[:, 1:, :]
        # x_con2 = x_2[:, 1:, :]

        x_con1 = torch.einsum('nld->ndl', x_con1)
        x_con2 = torch.einsum('nld->ndl', x_2)

        x_con1 = x_con1.reshape((x_con1.shape[0], x_con1.shape[1], self.patch_size, self.patch_size))
        x_con2 = x_con2.reshape((x_con2.shape[0], x_con2.shape[1], self.patch_size, self.patch_size))

        x_con =self.fuse(torch.cat([x_con1, x_con2], dim=1))

        x1, x2, x3, x4, x5, x6 = self.cnn_decoder(x_con)

        # con_loss
        con_loss1 = 0.5 * self.loss_fun2(x1, imgs11) + 0.5 * self.loss_fun2(x2, imgs21)
        con_loss2 = 0.5 * self.loss_fun2(x3, imgs12) + 0.5 * self.loss_fun2(x4, imgs22)
        con_loss3 = 0.5 * self.loss_fun2(x5, imgs13) + 0.5 * self.loss_fun2(x6, imgs23)
        con_loss = 1 / 3 * con_loss1 + 1 / 3 * con_loss2 + 1 / 3 * con_loss3

        return con_loss

    def forward(self, img11, img21, img12, img22, img13, img23):

        x_spa, x_spe, x_cnn = self.encoder(img11, img21, img12, img22, img13, img23)

        con_loss = self.decoder(x_spa, x_spe, img11, img21, img12, img22, img13, img23)
        x_cls = self.classifier(x_spa, x_spe, x_cnn)
        return x_cls, con_loss




