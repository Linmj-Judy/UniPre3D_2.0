"""
Gated Fusion Module with Learnable Channel Attention Gates

This module implements learnable gating mechanisms for adaptive fusion of 2D and 3D features,
as described in the improvement plan. The gate weights are dynamically computed based on
global statistics and scale features.
"""

import torch
import torch.nn as nn
import numpy as np


class GatedFeatureFusion(nn.Module):
    """
    Learnable gated fusion module that dynamically balances 2D and 3D feature contributions.
    
    The module computes gate weights based on:
    - Global average and max pooling of features
    - Scale statistics (point count, density, view angle)
    """
    
    def __init__(self, dim_3d: int, dim_2d: int, hidden_dim: int = 128):
        """
        Args:
            dim_3d: Dimension of 3D features
            dim_2d: Dimension of 2D features  
            hidden_dim: Hidden dimension for gate network
        """
        super().__init__()
        self.dim_3d = dim_3d
        self.dim_2d = dim_2d
        self.hidden_dim = hidden_dim
        
        # Scale statistics dimension: log(N), density_mean, density_var, density_skew
        self.scale_stat_dim = 4
        
        # Gate network input dimension
        gate_input_dim = 2 * dim_3d + 2 * dim_2d + self.scale_stat_dim
        
        # Shared MLP for computing gate weights
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Separate output heads for 3D and 2D gates
        self.gate_3d = nn.Linear(hidden_dim, 1)
        self.gate_2d = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_scale_statistics(self, center: torch.Tensor, point_cloud: torch.Tensor = None) -> torch.Tensor:
        """
        Compute scale-related statistics for gating.
        
        Args:
            center: Point cloud centers [B, N, 3]
            point_cloud: Original point cloud (optional) [B, N, 3]
            
        Returns:
            Scale statistics tensor [B, scale_stat_dim]
        """
        B, N = center.shape[:2]
        device = center.device
        
        # 1. Log-normalized point count
        log_n = torch.log(torch.tensor(N, dtype=torch.float32, device=device))
        log_n_norm = (log_n - 8.0) / 4.0  # Normalize around typical values
        log_n_feat = log_n_norm.unsqueeze(0).expand(B, 1)
        
        # 2. Compute point density statistics via grid histogram
        # Discretize space into 16x16x16 grid
        grid_size = 16
        coords_min = center.min(dim=1, keepdim=True)[0]
        coords_max = center.max(dim=1, keepdim=True)[0]
        coords_range = coords_max - coords_min + 1e-6
        
        # Normalize coordinates to [0, grid_size)
        coords_norm = (center - coords_min) / coords_range * (grid_size - 1)
        coords_idx = coords_norm.long().clamp(0, grid_size - 1)
        
        # Compute density histogram
        density_stats = []
        for b in range(B):
            # Flatten grid indices
            grid_idx = (coords_idx[b, :, 0] * grid_size * grid_size + 
                       coords_idx[b, :, 1] * grid_size + 
                       coords_idx[b, :, 2])
            
            # Count points per grid cell
            hist = torch.bincount(grid_idx, minlength=grid_size**3).float()
            hist = hist / (N + 1e-6)  # Normalize
            
            # Compute moments
            density_mean = hist.mean()
            density_var = hist.var()
            density_skew = ((hist - density_mean) ** 3).mean() / (density_var ** 1.5 + 1e-6)
            
            density_stats.append(torch.stack([density_mean, density_var, density_skew]))
        
        density_feat = torch.stack(density_stats, dim=0)  # [B, 3]
        
        # Concatenate all scale statistics
        scale_stats = torch.cat([log_n_feat, density_feat], dim=1)  # [B, 4]
        
        return scale_stats
    
    def forward(
        self, 
        feat_3d: torch.Tensor,
        feat_2d: torch.Tensor, 
        center: torch.Tensor,
        point_cloud: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply gated fusion.
        
        Args:
            feat_3d: 3D features [B, N, C_3d] 
            feat_2d: 2D features [B, N, C_2d]
            center: Point centers [B, N, 3]
            point_cloud: Original point cloud (optional)
            
        Returns:
            Gated features [B, N, C_3d]
        """
        B, N = feat_3d.shape[:2]
        
        # Compute global descriptors for 3D features
        g_3d_avg = feat_3d.mean(dim=1)  # [B, C_3d]
        g_3d_max = feat_3d.max(dim=1)[0]  # [B, C_3d]
        g_3d = torch.cat([g_3d_avg, g_3d_max], dim=1)  # [B, 2*C_3d]
        
        # Compute global descriptors for 2D features
        g_2d_avg = feat_2d.mean(dim=1)  # [B, C_2d]
        g_2d_max = feat_2d.max(dim=1)[0]  # [B, C_2d]
        g_2d = torch.cat([g_2d_avg, g_2d_max], dim=1)  # [B, 2*C_2d]
        
        # Compute scale statistics
        scale_stats = self.compute_scale_statistics(center, point_cloud)  # [B, scale_stat_dim]
        
        # Concatenate all inputs
        gate_input = torch.cat([g_3d, g_2d, scale_stats], dim=1)  # [B, gate_input_dim]
        
        # Compute gate weights
        h = self.gate_network(gate_input)  # [B, hidden_dim]
        w_3d = torch.sigmoid(self.gate_3d(h))  # [B, 1]
        w_2d = torch.sigmoid(self.gate_2d(h))  # [B, 1]
        
        # Expand gate weights to match feature dimensions
        w_3d = w_3d.unsqueeze(1)  # [B, 1, 1]
        w_2d = w_2d.unsqueeze(1)  # [B, 1, 1]
        
        # Ensure 2D features match 3D feature dimension
        if feat_2d.shape[-1] != feat_3d.shape[-1]:
            # Project 2D features to 3D feature dimension
            if not hasattr(self, 'feat_2d_proj'):
                self.feat_2d_proj = nn.Linear(feat_2d.shape[-1], feat_3d.shape[-1]).to(feat_2d.device)
            feat_2d = self.feat_2d_proj(feat_2d)
        
        # Apply gated fusion: F_fuse = w_3d * F_3d + w_2d * F_2d
        feat_gated = w_3d * feat_3d + w_2d * feat_2d
        
        return feat_gated, {'w_3d': w_3d, 'w_2d': w_2d}


class FusionRouter(nn.Module):
    """
    Learnable routing module that selects between different fusion strategies.
    
    Routes between:
    - Only 3D branch (no fusion)
    - Feature fusion (late fusion at decoder)
    - Point fusion (early fusion at encoder)
    """
    
    def __init__(self, feat_dim: int, hidden_dim: int = 64):
        """
        Args:
            feat_dim: Feature dimension from encoder
            hidden_dim: Hidden dimension for router network
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        
        # Scale statistics dimension
        self.scale_stat_dim = 4
        
        # Router network
        router_input_dim = feat_dim + self.scale_stat_dim
        self.router_network = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3 routes: only_3d, feat_fusion, point_fusion
        )
        
        # Temperature for Gumbel-Softmax (will be annealed during training)
        self.register_buffer('temperature', torch.tensor(1.0))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_temperature(self, temp: float):
        """Set temperature for Gumbel-Softmax"""
        self.temperature.fill_(temp)
    
    def compute_scale_statistics(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute scale statistics (same as in GatedFeatureFusion).
        
        Args:
            coords: Point coordinates [B, N, 3] or [N, 3]
            
        Returns:
            Scale statistics [B, scale_stat_dim] or [1, scale_stat_dim]
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            
        B, N = coords.shape[:2]
        device = coords.device
        
        # Log-normalized point count
        log_n = torch.log(torch.tensor(N, dtype=torch.float32, device=device))
        log_n_norm = (log_n - 8.0) / 4.0
        log_n_feat = log_n_norm.unsqueeze(0).expand(B, 1)
        
        # Compute density statistics
        grid_size = 16
        coords_min = coords.min(dim=1, keepdim=True)[0]
        coords_max = coords.max(dim=1, keepdim=True)[0]
        coords_range = coords_max - coords_min + 1e-6
        
        coords_norm = (coords - coords_min) / coords_range * (grid_size - 1)
        coords_idx = coords_norm.long().clamp(0, grid_size - 1)
        
        density_stats = []
        for b in range(B):
            grid_idx = (coords_idx[b, :, 0] * grid_size * grid_size + 
                       coords_idx[b, :, 1] * grid_size + 
                       coords_idx[b, :, 2])
            
            hist = torch.bincount(grid_idx, minlength=grid_size**3).float()
            hist = hist / (N + 1e-6)
            
            density_mean = hist.mean()
            density_var = hist.var()
            density_skew = ((hist - density_mean) ** 3).mean() / (density_var ** 1.5 + 1e-6)
            
            density_stats.append(torch.stack([density_mean, density_var, density_skew]))
        
        density_feat = torch.stack(density_stats, dim=0)
        scale_stats = torch.cat([log_n_feat, density_feat], dim=1)
        
        return scale_stats
    
    def forward(
        self, 
        feat_3d: torch.Tensor, 
        coords: torch.Tensor,
        training: bool = True
    ) -> tuple:
        """
        Compute routing weights.
        
        Args:
            feat_3d: 3D features from encoder [B, N, C] or [B, C]
            coords: Point coordinates [B, N, 3] or [N, 3]
            training: Whether in training mode
            
        Returns:
            routing_weights: Soft routing weights [B, 3]
            routing_logits: Raw logits before softmax [B, 3]
        """
        # Handle different input shapes
        if feat_3d.dim() == 3:
            # [B, N, C] -> [B, C] via global pooling
            g_feat = feat_3d.mean(dim=1)
        else:
            # Already [B, C]
            g_feat = feat_3d
        
        # Compute scale statistics
        scale_stats = self.compute_scale_statistics(coords)
        
        # Concatenate features
        router_input = torch.cat([g_feat, scale_stats], dim=1)
        
        # Compute routing logits
        logits = self.router_network(router_input)  # [B, 3]
        
        if training:
            # Apply Gumbel-Softmax for differentiable sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            routing_weights = torch.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
        else:
            # Use softmax without Gumbel noise during inference
            routing_weights = torch.softmax(logits / self.temperature, dim=-1)
        
        return routing_weights, logits


class DropPath(nn.Module):
    """
    Drop Path (Stochastic Depth) for regularization.
    Randomly drops entire feature paths during training.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with paths randomly dropped
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        
        # Generate random tensor for path dropping
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        
        # Scale output to maintain expected value
        output = x.div(keep_prob) * random_tensor
        
        return output


class FeatureDropout(nn.Module):
    """
    Channel-wise dropout for 2D features.
    Randomly drops feature channels to prevent over-reliance on 2D features.
    """
    
    def __init__(self, drop_prob: float = 0.2):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., C]
            
        Returns:
            Output with random channels dropped
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        
        # Generate random mask for channels
        mask_shape = (1,) * (x.ndim - 1) + (x.shape[-1],)
        mask = torch.bernoulli(torch.full(mask_shape, keep_prob, device=x.device))
        
        # Apply mask and scale
        output = x * mask / keep_prob
        
        return output

