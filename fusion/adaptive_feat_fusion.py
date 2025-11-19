"""
Adaptive Feature Fusion Module with Learnable Gating

This module extends the original FeatureFusion with learnable gating mechanisms
for adaptive balancing of 2D and 3D features.
"""

import torch
import torch.nn as nn
from .gated_fusion import GatedFeatureFusion, DropPath, FeatureDropout


class AdaptiveFeatureFusion:
    """
    Enhanced feature fusion with learnable gating and regularization.
    
    This class wraps the original feature fusion logic and adds:
    - Learnable channel attention gates
    - DropPath regularization for 2D features
    - Channel-wise dropout for 2D features
    """
    
    def __init__(
        self, 
        fusion_mlp: nn.Module,
        dim_3d: int = 384,
        dim_2d: int = 128,
        use_gating: bool = True,
        drop_path_rate: float = 0.2,
        feature_dropout_rate: float = 0.2,
    ):
        """
        Initialize the AdaptiveFeatureFusion module.
        
        Args:
            fusion_mlp: MLP network for processing concatenated features
            dim_3d: Dimension of 3D features
            dim_2d: Dimension of 2D features
            use_gating: Whether to use learnable gating
            drop_path_rate: Probability for DropPath
            feature_dropout_rate: Probability for channel dropout
        """
        self.fusion_mlp = fusion_mlp
        self.use_gating = use_gating
        
        # Learnable gating module
        if use_gating:
            self.gating = GatedFeatureFusion(dim_3d=dim_3d, dim_2d=dim_2d)
        
        # Regularization modules for 2D features
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else None
        self.feature_dropout = FeatureDropout(feature_dropout_rate) if feature_dropout_rate > 0 else None
    
    def project_points_to_image(
        self, center: torch.Tensor, c2w_matrix: torch.Tensor, intrinsic: torch.Tensor
    ) -> tuple:
        """
        Project 3D points to 2D image space using camera parameters.
        
        Args:
            center: 3D points in world space [B, N, 3]
            c2w_matrix: Camera-to-world matrix [B, 4, 4]
            intrinsic: Camera intrinsic matrix [3, 4] or [B, 3, 4]
            
        Returns:
            tuple: Projected pixel coordinates and depth values
        """
        # Add homogeneous coordinate
        coords_homogeneous = torch.cat(
            [center, torch.ones([*center.shape[:2], 1], device=center.device)], dim=2
        )
        
        # Transform points from world to camera space
        w2c_matrix = torch.linalg.inv(c2w_matrix.permute(0, 2, 1))
        camera_points = torch.matmul(
            w2c_matrix, coords_homogeneous.transpose(1, 2)
        ).transpose(1, 2)
        
        # Handle intrinsic matrix shape
        if intrinsic.dim() == 2:
            intrinsic_00 = intrinsic[0, 0]
            intrinsic_11 = intrinsic[1, 1]
            intrinsic_02 = intrinsic[0, 2]
            intrinsic_12 = intrinsic[1, 2]
        else:
            intrinsic_00 = intrinsic[0, 0, 0]
            intrinsic_11 = intrinsic[0, 1, 1]
            intrinsic_02 = intrinsic[0, 0, 2]
            intrinsic_12 = intrinsic[0, 1, 2]
        
        # Perspective projection
        pixel_coords = camera_points.clone()
        pixel_coords[..., 0] = (
            camera_points[..., 0] * intrinsic_00
        ) / (camera_points[..., 2] + 1e-6) + intrinsic_02
        pixel_coords[..., 1] = (
            camera_points[..., 1] * intrinsic_11
        ) / (camera_points[..., 2] + 1e-6) + intrinsic_12
        
        return torch.round(pixel_coords[..., :2]).long(), camera_points[..., 2]
    
    def __call__(
        self,
        x: torch.Tensor,
        center: torch.Tensor,
        image_features: torch.Tensor,
        c2w_projection_matrix: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform adaptive feature fusion between 3D points and 2D image features.
        
        Args:
            x: Point features [B, N, C]
            center: 3D points in world space [B, N, 3]
            image_features: Image features from encoder [B, C, H, W]
            c2w_projection_matrix: Camera-to-world matrix [B, 4, 4]
            intrinsic: Camera intrinsic matrix [3, 4]
            
        Returns:
            torch.Tensor: Fused features [B, N, C']
        """
        B, N = center.shape[:2]
        C, H, W = image_features.shape[1:]
        
        if c2w_projection_matrix.dim() == 4:
            c2w_projection_matrix = c2w_projection_matrix[:, 0]
        
        # Project 3D points to 2D image space
        pi_xy, p_depth = self.project_points_to_image(
            center, c2w_projection_matrix, intrinsic
        )
        
        # Check which points project inside the image bounds
        inside_mask = (
            (pi_xy[..., 0] >= 0)
            & (pi_xy[..., 1] >= 0)
            & (pi_xy[..., 0] < H)
            & (pi_xy[..., 1] < W)
            & (p_depth >= 0)
        )
        
        # Get valid point indices
        valid_indices = torch.nonzero(inside_mask)
        
        batch_indices, point_indices = valid_indices[:, 0], valid_indices[:, 1]
        pixel_x = pi_xy[batch_indices, point_indices, 0]
        pixel_y = pi_xy[batch_indices, point_indices, 1]
        valid_depth = p_depth[batch_indices, point_indices]
        
        # Handle occlusions using depth comparison
        unique_ids = batch_indices * (H * W) + pixel_y * H + pixel_x
        max_id = unique_ids.max().item() + 1 if len(unique_ids) > 0 else 1
        min_depths = torch.full(
            (max_id,), float("inf"), device=valid_depth.device, dtype=valid_depth.dtype
        )
        if len(unique_ids) > 0:
            min_depths.scatter_reduce_(
                0, unique_ids, valid_depth, reduce="amin", include_self=False
            )
        
        # Keep only the closest points for each pixel
        min_depth_mask = valid_depth == min_depths[unique_ids]
        
        # Initialize output feature tensor
        mapped_features = torch.zeros((B, N, C), device=center.device)
        
        # Assign features from valid projections
        if min_depth_mask.sum() > 0:
            mapped_features[
                batch_indices[min_depth_mask], point_indices[min_depth_mask]
            ] = image_features[
                batch_indices[min_depth_mask],
                :,
                pixel_x[min_depth_mask],
                pixel_y[min_depth_mask],
            ].permute(0, 1)
        
        # Apply regularization to 2D features
        if self.drop_path is not None:
            mapped_features = self.drop_path(mapped_features)
        
        if self.feature_dropout is not None:
            mapped_features = self.feature_dropout(mapped_features)
        
        # Apply learnable gating if enabled
        if self.use_gating:
            # Ensure dimensions match for gating
            x_num = x.shape[1]
            if x_num > N:
                # Transformer with CLS token
                x_patch = x[:, 1:]  # [B, N, C]
                gated_feat, gate_weights = self.gating(x_patch, mapped_features, center)
                
                # Concatenate gated features with 2D features
                x_patch_concat = torch.cat([gated_feat, mapped_features], dim=-1)
                
                # Handle CLS token
                CLS_token_features = torch.cat(
                    [x[:, 0:1], torch.zeros((B, 1, C), device=center.device)], dim=-1
                )
                x = torch.cat([CLS_token_features, x_patch_concat], dim=1)
            else:
                gated_feat, gate_weights = self.gating(x, mapped_features, center)
                x = torch.cat([gated_feat, mapped_features], dim=-1)
        else:
            # Original concatenation without gating
            x_num = x.shape[1]
            if x_num > N:
                # Transformer CLS Token
                x_patch = torch.cat([x[:, 1:], mapped_features], dim=-1)
                CLS_token_features = torch.cat(
                    [x[:, 0:1], torch.zeros((B, 1, C), device=center.device)], dim=-1
                )
                x = torch.cat([CLS_token_features, x_patch], dim=1)
            else:
                x = torch.cat([x, mapped_features], dim=-1)
        
        # Apply feature MLP and return
        return self.fusion_mlp(x)

