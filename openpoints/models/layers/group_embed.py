# Patch embedding for 2D/3D data
# Reference:
import math
import torch
from torch import nn as nn
import torch.nn.functional as F
from .subsample import furthest_point_sample, random_sample
from .group import KNNGroup, QueryAndGroup, create_grouper, get_aggregation_feautres
from .conv import create_convblock1d, create_convblock2d, create_linearblock, create_norm, create_act
from .local_aggregation import CHANNEL_MAP
from ..build import MODELS


class SubsampleGroup(nn.Module):
    """ Point cloud to subsampled groups
    """

    def __init__(self,
                 num_groups=256, group_size=32,
                 subsample='fps',  # random, FPS
                 group='ballquery',
                 radius=0.1,
                 **kwargs
                 ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size

        self.subsample = subsample
        self.group = group

        if 'ball' in self.group.lower() or 'query' in self.group.lower():
            self.grouper = QueryAndGroup(radius, self.group_size)
        elif 'knn' in self.group.lower():
            self.grouper = KNNGroup(self.group_size)
        else:
            raise NotImplementedError(f'{self.group.lower()} is not implemented. Only support ballquery, knn')

    def forward(self, p, x=None):
        # Ensure input is contiguous
        p = p.contiguous()
        if x is not None:
            x = x.contiguous()
        
        # CUDA synchronization before sampling
        if torch.cuda.is_available() and p.is_cuda:
            torch.cuda.synchronize()
        
        if 'fps' in self.subsample.lower() or 'furthest' in self.subsample.lower() or 'farthest' in self.subsample.lower():
            idx = furthest_point_sample(p, self.num_groups).to(torch.int64)
        elif 'random' in self.subsample.lower() or 'rs' in self.subsample.lower():
            idx = random_sample(p, self.num_groups)
        else:
            raise NotImplementedError(f'{self.subsample.lower()} is not implemented. Only support fps, random')
        
        # Ensure idx is contiguous
        if not idx.is_contiguous():
            idx = idx.contiguous()
        
        center_p = torch.gather(p, 1,
                                  idx.unsqueeze(-1).expand(-1, -1, 3))  # downsampled point cloud, [B, npoint, 3]
        # center_p = torch.gather(p, 1,
        #                     idx.unsqueeze(-1).expand(-1, -1, p.shape[-1]))  # downsampled point cloud, [B, npoint, p.shape[-1]]
        
        # CUDA synchronization after sampling
        if torch.cuda.is_available() and p.is_cuda:
            torch.cuda.synchronize()
            print(f"[SubsampleGroup] 采样完成，准备调用 grouper...")
            print(f"[SubsampleGroup] p shape: {p.shape}, device: {p.device}, is_contiguous: {p.is_contiguous()}")
            print(f"[SubsampleGroup] center_p shape: {center_p.shape}, device: {center_p.device}, is_contiguous: {center_p.is_contiguous()}")
            print(f"[SubsampleGroup] idx shape: {idx.shape}, device: {idx.device}, dtype: {idx.dtype}, is_contiguous: {idx.is_contiguous()}")
            if x is not None:
                print(f"[SubsampleGroup] x shape: {x.shape}, device: {x.device}, is_contiguous: {x.is_contiguous()}")
        
        try:
            if x is not None:
                B, C, N = x.shape[:3]
                center_x = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, C, -1)).unsqueeze(-1)
                print(f"[SubsampleGroup] 调用 grouper(center_p, p, x)...")
                grouped_p, fj = self.grouper(center_p, p, x)
                print(f"[SubsampleGroup] grouper 返回成功")
                print(f"[SubsampleGroup] grouped_p shape: {grouped_p.shape}, device: {grouped_p.device}, is_contiguous: {grouped_p.is_contiguous()}")
                return grouped_p, center_p, fj, center_x
            else:
                print(f"[SubsampleGroup] 调用 grouper(center_p, p)...")
                grouped_p, _ = self.grouper(center_p, p)
                print(f"[SubsampleGroup] grouper 返回成功")
                print(f"[SubsampleGroup] grouped_p shape: {grouped_p.shape}, device: {grouped_p.device}, is_contiguous: {grouped_p.is_contiguous()}")
                return grouped_p, center_p
        except Exception as e:
            import traceback
            print(f"[SubsampleGroup] grouper 调用失败: {e}")
            print(f"[SubsampleGroup] traceback:\n{traceback.format_exc()}")
            if torch.cuda.is_available() and p.is_cuda:
                print(f"[SubsampleGroup] CUDA error: {torch.cuda.get_last_error()}")
            raise


@MODELS.register_module()
class PointPatchEmbed(nn.Module):
    """ Point cloud to Group Embedding using GCN
    Patch Embedding for 3d data (point cloud)
    A convolution based approach to patchifying a point cloud w/ embedding projection.
    """

    def __init__(self,
                 sample_ratio=0.0625, group_size=32,
                 in_channels=3,
                 layers=4,
                 embed_dim=256,
                 channels=None,
                 subsample='fps',  # random, FPS
                 group='ballquery',
                 normalize_dp=False,
                 radius=0.1,
                 feature_type='dp_df',
                 relative_xyz=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args={'order': 'conv-norm-act'},
                 reduction='max',
                 **kwargs
                 ):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.group_size = group_size

        self.feature_type = feature_type
        # subsample layer and group layer
        if subsample.lower() == 'fps':
            self.sample_fn = furthest_point_sample
        elif 'random' in subsample.lower():
            self.sample_fn = random_sample

        # TODO: make this embedding progressively
        self.group = group.lower()
        if 'ball' in self.group or 'query' in self.group:
            self.grouper = QueryAndGroup(nsample=self.group_size,
                                         relative_xyz=relative_xyz, normalize_dp=normalize_dp,
                                         radius=radius)
        elif 'knn' in self.group.lower():
            self.grouper = KNNGroup(self.group_size, relative_xyz=relative_xyz, normalize_dp=normalize_dp)
        else:
            raise NotImplementedError(f'{self.group.lower()} is not implemented. Only support ballquery, knn')

        # # convolutions
        if channels is None:
            channels = [CHANNEL_MAP[feature_type](in_channels)] + [embed_dim] * (layers // 2) + [embed_dim * 2] * (
                    layers // 2 - 1) + [embed_dim]
        else:
            channels = [CHANNEL_MAP[feature_type](in_channels)] + channels + [embed_dim]
            layers = len(channels) -1
        conv1 = []
        for i in range(layers // 2):
            conv1.append(create_convblock2d(channels[i], channels[i + 1],
                                            norm_args=norm_args if i!=(layers//2-1) else None,
                                            act_args=act_args if i!=(layers//2-1) else None,
                                            **conv_args))
        self.conv1 = nn.Sequential(*conv1)

        channels[layers // 2] *= 2
        conv2 = []
        for i in range(layers // 2, layers):
            conv2.append(create_convblock2d(channels[i], channels[i + 1],
                                            norm_args=norm_args if i!=(layers-1) else None,
                                            act_args=act_args if i!=(layers-1) else None,
                                            **conv_args
                                            ))
        self.conv2 = nn.Sequential(*conv2)

        # reduction layer
        if reduction in ['mean', 'avg', 'meanpool', 'avgpool']:
            self.pool = lambda x: torch.mean(x, dim=-1, keepdim=True)
        else:
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=True)[0]
        self.out_channels = channels[-1]
        self.channel_list = [in_channels, embed_dim]

    def forward(self, p, x=None):
        # downsample
        B, N, _ = p.shape[:3]
        idx = self.sample_fn(p, int(N * self.sample_ratio)).long()
        center_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

        # query neighbors.
        dp, fj = self.grouper(center_p, p, x)

        """visualization
        from openpoints.dataset.vis3d import vis_multi_points
        new_p = (dp.permute(0, 2, 3, 1) + center_p.unsqueeze(2)).view(B, -1, 3)
        vis_multi_points([p[0].cpu().numpy(), new_p[0].cpu().numpy(), center_p[0].cpu().numpy()])
        """

        # concat neighborhood x
        # TODO: using a local aggregation layer
        if self.feature_type == 'dp':
            fj = dp
        elif self.feature_type == 'dp_fj':
            fj = torch.cat([dp, fj], dim=1)
        elif self.feature_type == 'dp_df':
            center_x = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))
            fj = torch.cat([dp, fj - center_x.unsqueeze(-1)], dim=1)
        elif self.feature_type == 'df':
            center_x = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))
            fj = fj - center_x.unsqueeze(-1)
        fj = self.conv1(fj)

        fj = torch.cat(
            [self.pool(fj).expand(-1, -1, -1, self.group_size),
             fj],
            dim=1)
        out_f = self.pool(self.conv2(fj)).squeeze(-1)
        return [p, center_p], [x, out_f]


@MODELS.register_module()
class P3Embed(nn.Module):
    """
    Progressive Point Patch Embedding for 3d data (point cloud)
    A convolution based approach to patchifying a point cloud w/ embedding projection.
    """

    def __init__(self,
                 sample_ratio=0.0625,
                 scale=4,
                 group_size=32,
                 in_channels=3,
                 layers=4,
                 embed_dim=256,
                 subsample='fps',  # random, FPS
                 group='ballquery',
                 normalize_dp=False,
                 radius=0.1,
                 feature_type='dp_df',
                 relative_xyz=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args={'order': 'conv-norm-act'},
                 reduction='max',
                 **kwargs
                 ):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.group_size = group_size

        self.feature_type = feature_type
        # subsample layer and group layer
        if subsample.lower() == 'fps':
            self.sample_fn = furthest_point_sample
        elif 'random' in subsample.lower():
            self.sample_fn = random_sample

        self.group = group.lower()
        if 'ball' in self.group or 'query' in self.group:
            self.grouper = QueryAndGroup(nsample=self.group_size,
                                         relative_xyz=relative_xyz, normalize_dp=normalize_dp,
                                         radius=radius)
        elif 'knn' in self.group.lower():
            self.grouper = KNNGroup(self.group_size, relative_xyz=relative_xyz, normalize_dp=normalize_dp)
        else:
            raise NotImplementedError(f'{self.group.lower()} is not implemented. Only support ballquery, knn')

        # stages
        stages = int(math.log(1/sample_ratio, scale))
        embed_dim = int(embed_dim // 2 ** (stages-1))
        self.convs = nn.ModuleList()
        self.channel_list = [in_channels]
        for _ in range(int(stages)):
            # convolutions
            channels = [CHANNEL_MAP[feature_type](in_channels)] + [embed_dim] * (layers // 2) + [embed_dim * 2] * (
                    layers // 2 - 1) + [embed_dim]
            conv1 = []
            for i in range(layers // 2):
                conv1.append(create_convblock2d(channels[i], channels[i + 1],
                                                norm_args=norm_args if i!=(layers//2-1) else None,
                                                act_args=act_args if i!=(layers//2-1) else None,
                                                **conv_args))
            conv1 = nn.Sequential(*conv1)

            channels[layers // 2] *= 2
            conv2 = []
            for i in range(layers // 2, layers):
                conv2.append(create_convblock2d(channels[i], channels[i + 1],
                                                norm_args=norm_args,
                                                act_args=act_args,
                                                **conv_args
                                                ))
            conv2 = nn.Sequential(*conv2)
            self.convs.append(nn.ModuleList([conv1, conv2]))

            self.channel_list.append(embed_dim)
            in_channels = embed_dim
            embed_dim *= 2

        # reduction layer
        if reduction in ['mean', 'avg', 'meanpool', 'avgpool']:
            self.pool = lambda x: torch.mean(x, dim=-1, keepdim=True)
        else:
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=True)[0]
        self.out_channels = channels[-1]

    def forward(self, p, f=None):
        B, N, _ = p.shape[:3]
        out_p, out_f = [p], [f]
        for convs in self.convs:
            # Progressive downsampling
            cur_p, cur_f = out_p[-1], out_f[-1]
            idx = self.sample_fn(cur_p, int(N //4)).long()
            N = N // 4
            center_p = torch.gather(cur_p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            center_f = torch.gather(cur_f, 2, idx.unsqueeze(1).expand(-1, cur_f.shape[1], -1))

            # query neighbors.
            dp, fj = self.grouper(center_p, cur_p, cur_f)
            fj = get_aggregation_feautres(center_p, dp, center_f, fj, self.feature_type)

            # graph convolutions
            fj = convs[0](fj)
            fj = torch.cat(
                [self.pool(fj).expand(-1, -1, -1, self.group_size),
                fj],
                dim=1)

            # output
            out_f.append(self.pool(convs[1](fj)).squeeze(-1))
            out_p.append(center_p)
        return out_p, out_f
