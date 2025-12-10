import torch
import torch.nn as nn
import traceback
import sys

from openpoints.models.backbone.pointmlp import pointMLP
from openpoints.models.backbone.transformer import PointTransformerEncoder
from openpoints.models.Mamba3D.Mamba3D import Mamba3DSeg
from openpoints.models.segmentation import BaseSeg
from openpoints.models.PCM.PCM import PointMambaEncoder

from pointcept.models.sparse_unet.spconv_unet_v1m1_base import SpUNetBase
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)

from typing import Any, List, Optional, Tuple, Union, Dict


def safe_print(msg):
    """安全的打印函数，避免在多进程环境中出现问题"""
    try:
        print(msg, flush=True)
        sys.stdout.flush()
    except:
        pass


class PointFeaturePredictor(nn.Module):
    """Point cloud feature predictor supporting multiple backbone architectures"""

    SUPPORTED_MODELS = {
        "pointmlp": pointMLP,
        "transformer": PointTransformerEncoder,
        "pcm": PointMambaEncoder,
        "mamba3d": Mamba3DSeg,
        "sparseunet": SpUNetBase,
        "ptv3": PointTransformerV3,
    }

    def __init__(self, cfg, out_channels, pretrained_path=None):
        super(PointFeaturePredictor, self).__init__()
        self.cfg = cfg
        self.out_channels = out_channels

        # Initialize backbone
        self.encoder = self._create_encoder()

        # Initialize final layers
        self.final = self._create_final_layers()

        # Print model statistics
        self._print_model_stats()

        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights from a file"""
        state_dict = torch.load(pretrained_path)
        info = self.load_state_dict(state_dict, strict=False)
        print("Loaded pretrained weights from {}".format(pretrained_path))
        print("Missing keys: {}".format(info.missing_keys))
        print("Unexpected keys: {}".format(info.unexpected_keys))

    def _create_encoder(self):
        """Create encoder based on model configuration"""
        model_type = self.cfg.model.backbone_type.lower()
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_type == "transformer":
            return self.SUPPORTED_MODELS[model_type](
                in_channels=3, num_groups=128, encoder_dims=384, depth=16
            )
        elif model_type == "sparseunet":
            return self.SUPPORTED_MODELS[model_type](
                in_channels=6, num_classes=64, cfg=self.cfg
            )
        elif model_type == "ptv3":
            return PointTransformerV3(in_channels=6, cfg=self.cfg)
        elif model_type == "pcm":
            return BaseSeg(**self._get_mamba_config())
        elif model_type == "mamba3d":
            return Mamba3DSeg(self._get_mamba3d_config())
        else:  # pointmlp
            return self.SUPPORTED_MODELS[model_type](cfg=self.cfg)

    def _create_final_layers(self):
        """Create final layers based on model type"""
        if self.cfg.model.backbone_type.lower() in ["transformer", "mamba3d"]:
            return nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 23))
        elif self.cfg.model.backbone_type.lower() in ["ptv3", "sparseunet"]:
            return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 23))
        else:
            return nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 23))

    def _print_model_stats(self):
        """Print model parameter statistics"""
        all_params = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        print(f"Encoder parameters: {all_params}")

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network"""
        try:
            safe_print(f"[PointFeaturePredictor] forward() 开始")
            
            # 检查输入
            if isinstance(x, dict):
                safe_print(f"[PointFeaturePredictor] 输入x是字典，键: {list(x.keys())}")
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        safe_print(f"[PointFeaturePredictor] x['{k}'] 形状: {v.shape}, dtype: {v.dtype}, device: {v.device}, is_contiguous: {v.is_contiguous()}")
            else:
                safe_print(f"[PointFeaturePredictor] 输入x形状: {x.shape}, dtype: {x.dtype}, device: {x.device}, is_contiguous: {x.is_contiguous()}")
            
            # 检查encoder
            safe_print(f"[PointFeaturePredictor] encoder类型: {type(self.encoder)}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                safe_print(f"[PointFeaturePredictor] CUDA同步完成")
            
            # 调用encoder - 这是最可能出问题的地方
            safe_print(f"[PointFeaturePredictor] 准备调用 encoder.forward()...")
            try:
                x, center = self.encoder(x, None, None, None)
                safe_print(f"[PointFeaturePredictor] encoder.forward() 调用成功")
                safe_print(f"[PointFeaturePredictor] encoder输出x形状: {x.shape if isinstance(x, torch.Tensor) else type(x)}, dtype: {x.dtype if isinstance(x, torch.Tensor) else 'N/A'}, device: {x.device if isinstance(x, torch.Tensor) else 'N/A'}")
                safe_print(f"[PointFeaturePredictor] center形状: {center.shape if center is not None else None}, dtype: {center.dtype if center is not None else None}, device: {center.device if center is not None else None}")
            except Exception as e:
                safe_print(f"[PointFeaturePredictor] encoder.forward() 调用失败: {str(e)}")
                safe_print(f"[PointFeaturePredictor] 错误堆栈:\n{traceback.format_exc()}")
                if torch.cuda.is_available():
                    safe_print(f"[PointFeaturePredictor] GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB")
                raise
            
            # 同步CUDA操作
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                safe_print(f"[PointFeaturePredictor] encoder调用后CUDA同步完成")
            
            # 调用final层
            safe_print(f"[PointFeaturePredictor] 准备调用 final层...")
            try:
                output = self.final(x).permute(0, 2, 1)
                safe_print(f"[PointFeaturePredictor] final层调用成功，output形状: {output.shape}")
            except Exception as e:
                safe_print(f"[PointFeaturePredictor] final层调用失败: {str(e)}")
                safe_print(f"[PointFeaturePredictor] 错误堆栈:\n{traceback.format_exc()}")
                raise
            
            safe_print(f"[PointFeaturePredictor] forward() 完成")
            return (output, center)
        except Exception as e:
            safe_print(f"[PointFeaturePredictor] forward() 失败: {str(e)}")
            safe_print(f"[PointFeaturePredictor] 完整错误堆栈:\n{traceback.format_exc()}")
            if torch.cuda.is_available():
                safe_print(f"[PointFeaturePredictor] GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB")
            raise

    def forward_feat_fusion(
        self,
        x: torch.Tensor,
        image_features: torch.Tensor,
        c2w_projection_matrix: torch.Tensor,
        fusion_mlps: nn.ModuleList,
        intrinsic: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with image fusion"""
        try:
            safe_print(f"[PointFeaturePredictor] forward_feat_fusion() 开始")
            
            # 检查所有输入参数
            if isinstance(x, dict):
                safe_print(f"[PointFeaturePredictor] 输入x是字典，键: {list(x.keys())}")
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        safe_print(f"[PointFeaturePredictor] x['{k}']: shape={v.shape}, dtype={v.dtype}, device={v.device}, is_contiguous={v.is_contiguous()}")
            else:
                safe_print(f"[PointFeaturePredictor] 输入x形状: {x.shape}, dtype: {x.dtype}, device: {x.device}, is_contiguous: {x.is_contiguous()}")
            
            safe_print(f"[PointFeaturePredictor] image_features形状: {image_features.shape}, dtype: {image_features.dtype}, device: {image_features.device}")
            safe_print(f"[PointFeaturePredictor] c2w_projection_matrix形状: {c2w_projection_matrix.shape}, dtype: {c2w_projection_matrix.dtype}, device: {c2w_projection_matrix.device}")
            safe_print(f"[PointFeaturePredictor] fusion_mlps类型: {type(fusion_mlps)}")
            if isinstance(intrinsic, torch.Tensor):
                safe_print(f"[PointFeaturePredictor] intrinsic形状: {intrinsic.shape}, dtype: {intrinsic.dtype}, device: {intrinsic.device}")
            else:
                safe_print(f"[PointFeaturePredictor] intrinsic类型: {type(intrinsic)}, 值: {intrinsic}")
            
            # 检查encoder
            safe_print(f"[PointFeaturePredictor] encoder类型: {type(self.encoder)}")
            
            # 同步CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                safe_print(f"[PointFeaturePredictor] CUDA同步完成（调用encoder前）")
            
            # 调用encoder - 这是最可能出问题的地方
            safe_print(f"[PointFeaturePredictor] 准备调用 encoder.forward()...")
            safe_print(f"[PointFeaturePredictor] encoder类型: {type(self.encoder)}")
            safe_print(f"[PointFeaturePredictor] encoder设备: {next(self.encoder.parameters()).device if list(self.encoder.parameters()) else 'N/A'}")
            
            # 检查CUDA上下文
            if torch.cuda.is_available():
                current_device_idx = torch.cuda.current_device()
                current_device = torch.device(f'cuda:{current_device_idx}')
                safe_print(f"[PointFeaturePredictor] CUDA设备索引: {current_device_idx}, 设备对象: {current_device}")
                safe_print(f"[PointFeaturePredictor] CUDA上下文: {torch.cuda.current_stream()}")
                # 确保所有张量在正确的设备上
                if isinstance(x, dict):
                    for k, v in x.items():
                        if isinstance(v, torch.Tensor):
                            # 正确比较：比较设备对象，或者比较设备索引
                            if v.device.type == 'cuda' and v.device.index != current_device_idx:
                                safe_print(f"[PointFeaturePredictor] 警告: x['{k}'] 设备不匹配: {v.device} != {current_device}")
                            elif v.device.type != 'cuda':
                                safe_print(f"[PointFeaturePredictor] 警告: x['{k}'] 不在CUDA设备上: {v.device}")
                if isinstance(image_features, torch.Tensor):
                    if image_features.device.type == 'cuda' and image_features.device.index != current_device_idx:
                        safe_print(f"[PointFeaturePredictor] 警告: image_features 设备不匹配: {image_features.device} != {current_device}")
                    elif image_features.device.type != 'cuda':
                        safe_print(f"[PointFeaturePredictor] 警告: image_features 不在CUDA设备上: {image_features.device}")
            
            try:
                safe_print(f"[PointFeaturePredictor] 开始调用 encoder.forward()...")
                safe_print(f"[PointFeaturePredictor] 检查encoder是否有forward方法: {hasattr(self.encoder, 'forward')}")
                safe_print(f"[PointFeaturePredictor] encoder.forward类型: {type(self.encoder.forward)}")
                
                # 尝试获取forward方法的引用
                forward_method = getattr(self.encoder, 'forward', None)
                safe_print(f"[PointFeaturePredictor] forward_method获取成功: {forward_method is not None}")
                
                # 直接调用
                safe_print(f"[PointFeaturePredictor] 准备执行forward调用...")
                import sys
                sys.stdout.flush()
                
                x, center = forward_method(
                    x, image_features, c2w_projection_matrix, fusion_mlps, intrinsic
                )
                
                safe_print(f"[PointFeaturePredictor] encoder.forward() 调用完成")
                safe_print(f"[PointFeaturePredictor] encoder.forward() 调用成功")
                safe_print(f"[PointFeaturePredictor] encoder输出x形状: {x.shape if isinstance(x, torch.Tensor) else type(x)}, dtype: {x.dtype if isinstance(x, torch.Tensor) else 'N/A'}, device: {x.device if isinstance(x, torch.Tensor) else 'N/A'}")
                safe_print(f"[PointFeaturePredictor] center形状: {center.shape if center is not None else None}, dtype: {center.dtype if center is not None else None}, device: {center.device if center is not None else None}")
            except Exception as e:
                safe_print(f"[PointFeaturePredictor] encoder.forward() 调用失败: {str(e)}")
                safe_print(f"[PointFeaturePredictor] 错误堆栈:\n{traceback.format_exc()}")
                if torch.cuda.is_available():
                    safe_print(f"[PointFeaturePredictor] GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB")
                raise

            # 同步CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                safe_print(f"[PointFeaturePredictor] CUDA同步完成（encoder调用后）")

            # 调用final层
            safe_print(f"[PointFeaturePredictor] 准备调用 final层...")
            try:
                output = self.final(x).permute(0, 2, 1)
                safe_print(f"[PointFeaturePredictor] final层调用成功，output形状: {output.shape}")
            except Exception as e:
                safe_print(f"[PointFeaturePredictor] final层调用失败: {str(e)}")
                safe_print(f"[PointFeaturePredictor] 错误堆栈:\n{traceback.format_exc()}")
                raise

            safe_print(f"[PointFeaturePredictor] forward_feat_fusion() 完成")
            return (output, center)
        except Exception as e:
            safe_print(f"[PointFeaturePredictor] forward_feat_fusion() 失败: {str(e)}")
            safe_print(f"[PointFeaturePredictor] 完整错误堆栈:\n{traceback.format_exc()}")
            if torch.cuda.is_available():
                safe_print(f"[PointFeaturePredictor] GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB")
            raise

    def forward_point_fusion(
        self,
        x: torch.Tensor,
        image_features: torch.Tensor,
        unprojected_coords: torch.Tensor,
        fusion_mlps: nn.ModuleList,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with image fusion"""
        try:
            safe_print(f"[PointFeaturePredictor] forward_point_fusion() 开始")
            safe_print(f"[PointFeaturePredictor] 准备调用 encoder.forward()...")
            output = self.encoder.forward(
                x, image_features, unprojected_coords, fusion_mlps
            )
            safe_print(f"[PointFeaturePredictor] encoder.forward() 调用成功")
            
            if self.cfg.model.backbone_type.lower() == "ptv3":
                return (
                    self.final(output.sparse_conv_feat.features),
                    output.sparse_conv_feat.indices,
                )
            elif self.cfg.model.backbone_type.lower() == "sparseunet":
                return self.final(output.features), output.indices
        except Exception as e:
            safe_print(f"[PointFeaturePredictor] forward_point_fusion() 失败: {str(e)}")
            safe_print(f"[PointFeaturePredictor] 错误堆栈:\n{traceback.format_exc()}")
            raise

    def _get_mamba_config(self):
        """Get configuration for PointMambaEncoder"""
        return {
            "encoder_args": {
                "NAME": "PointMambaEncoder",
                "in_channels": 4,
                "embed_dim": 384,
                "groups": 1,
                "res_expansion": 1,
                "activation": "relu",
                "bias": False,
                "use_xyz": True,
                "normalize": "anchor",
                "dim_expansion": [1, 1, 2, 1],
                "pre_blocks": [1, 1, 1, 1],
                "mamba_blocks": [1, 2, 2, 4],
                "pos_blocks": [0, 0, 0, 0],
                "k_neighbors": [12, 12, 12, 12],
                "reducers": [2, 2, 2, 2],
                "rms_norm": True,
                "residual_in_fp32": True,
                "fused_add_norm": True,
                "bimamba_type": "v2",
                "drop_path_rate": 0.1,
                "mamba_pos": True,
                "mamba_layers_orders": [
                    "xyz",
                    "xzy",
                    "yxz",
                    "yzx",
                    "zxy",
                    "zyx",
                    "hilbert",
                    "z",
                    "z-trans",
                ],
                "use_order_prompt": True,
                "prompt_num_per_order": 6,
            },
            "decoder_args": {
                "NAME": "PointMambaDecoder",
                "encoder_channel_list": [384, 384, 384, 768, 768],
                "decoder_channel_list": [
                    768,
                    384,
                    384,
                    384,
                    # 128
                ],
                "decoder_blocks": [1, 1, 1, 1],
                "mamba_blocks": [0, 0, 0, 0],
                "mamba_layers_orders": [],
            },
            "cls_args": {
                "NAME": "SegHead",
                # "global_feat": "max,avg",
                "num_classes": 128,
                "in_channels": 384,
                # "in_channels": 128,
                "norm_args": {"norm": "bn"},
            },
        }

    def _get_mamba3d_config(self):
        """Get configuration for Mamba3DSeg"""

        class Mamba3DConfig:
            def __init__(self):
                self.NAME = "Mamba3D"
                self.trans_dim = 384
                self.depth = 16
                self.drop_path_rate = 0.1
                self.num_heads = 6
                self.group_size = 32
                self.num_group = 128
                self.encoder_dims = 384
                self.bimamba_type = "v4"
                self.center_local_k = 4
                self.ordering = False
                self.label_smooth = 0.0
                self.lr_ratio_cls = 1.0
                self.lr_ratio_lfa = 1.0
                self.fusion = True

        return Mamba3DConfig()
