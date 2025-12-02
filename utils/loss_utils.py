"""
Loss utilities for training Gaussian Splatting models.

Includes reconstruction losses and additional regularization losses for the improved model.
"""

import torch
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    """Generate 1D Gaussian kernel"""
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create 2D Gaussian window for SSIM"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    """
    Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image [C, H, W]
        img2: Second image [C, H, W]
        window_size: Size of Gaussian window
        size_average: Whether to average over spatial dimensions
        
    Returns:
        SSIM value
    """
    channel = img1.size(0)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # Add batch dimension
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def l1_loss(network_output, gt):
    """L1 loss between network output and ground truth"""
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    """L2 (MSE) loss between network output and ground truth"""
    return ((network_output - gt) ** 2).mean()


def focal_l2_loss(
    network_output, gt, background, non_bg_loss_rate=1.0, bg_loss_rate=1.0
):
    """
    Focal L2 loss with separate weighting for foreground and background.
    
    Args:
        network_output: Rendered images
        gt: Ground truth images
        background: Background color [3]
        non_bg_loss_rate: Weight for non-background pixels
        bg_loss_rate: Weight for background pixels
    """
    # Compute pixel-wise L2 loss
    pixel_loss = (network_output - gt) ** 2
    
    # Create background mask
    bg_mask = (gt == background.view(3, 1, 1)).all(dim=0, keepdim=True)
    
    # Apply different weights to foreground and background
    weighted_loss = torch.where(
        bg_mask,
        pixel_loss * bg_loss_rate,
        pixel_loss * non_bg_loss_rate,
    )
    
    return weighted_loss.mean()


def feature_consistency_loss(
    feat_with_2d: torch.Tensor,
    feat_without_2d: torch.Tensor,
    stop_gradient: bool = True,
) -> torch.Tensor:
    """
    Feature consistency loss to prevent over-reliance on 2D features.
    
    Encourages the 3D encoder to produce similar features with or without 2D guidance.
    
    Args:
        feat_with_2d: Features extracted with 2D guidance [B, N, C] or [N, C]
        feat_without_2d: Features extracted without 2D guidance [B, N, C] or [N, C]
        stop_gradient: Whether to stop gradient through feat_without_2d
        
    Returns:
        Consistency loss value
    """
    if stop_gradient:
        feat_without_2d = feat_without_2d.detach()
    
    # Compute L2 distance
    loss = F.mse_loss(feat_with_2d, feat_without_2d)
    
    return loss


def routing_sparsity_loss(routing_weights: torch.Tensor, lambda_sparse: float = 0.01) -> torch.Tensor:
    """
    Sparsity regularization for routing weights.
    
    Encourages the router to make decisive choices by maximizing the maximum weight.
    
    Args:
        routing_weights: Routing weights [B, num_routes]
        lambda_sparse: Regularization strength
        
    Returns:
        Sparsity loss value
    """
    # Encourage max weight to be close to 1
    max_weights = routing_weights.max(dim=-1)[0]  # [B]
    loss = lambda_sparse * (1 - max_weights).mean()
    
    return loss


def compute_total_loss(
    rendered_images: torch.Tensor,
    gt_images: torch.Tensor,
    cfg,
    device: torch.device,
    iteration: int = 0,
    lpips_fn=None,
    routing_weights: torch.Tensor = None,
    feat_with_2d: torch.Tensor = None,
    feat_without_2d: torch.Tensor = None,
) -> dict:
    """
    Compute total training loss including all components.
    
    Args:
        rendered_images: Rendered images from Gaussian splatting
        gt_images: Ground truth images
        cfg: Configuration object
        device: Torch device
        iteration: Current training iteration
        lpips_fn: LPIPS loss function (optional)
        routing_weights: Routing weights for sparsity loss (optional)
        feat_with_2d: Features with 2D guidance for consistency loss (optional)
        feat_without_2d: Features without 2D guidance for consistency loss (optional)
        
    Returns:
        Dictionary of loss components
    """
    losses = {}
    
    # Background color
    background = torch.tensor(
        [1, 1, 1] if cfg.data.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=device,
    )
    
    # Main reconstruction loss
    if cfg.opt.loss == "focal_l2":
        losses["l12_loss"] = focal_l2_loss(
            rendered_images,
            gt_images,
            background,
            cfg.opt.non_bg_color_loss_rate,
            cfg.opt.bg_color_loss_rate,
        )
    elif cfg.opt.loss == "l1":
        losses["l12_loss"] = l1_loss(rendered_images, gt_images)
    else:  # l2
        losses["l12_loss"] = l2_loss(rendered_images, gt_images)
    
    # LPIPS perceptual loss
    if lpips_fn is not None and cfg.opt.lambda_lpips > 0 and iteration > cfg.opt.start_lpips_after:
        losses["lpips_loss"] = torch.mean(
            lpips_fn(rendered_images * 2 - 1, gt_images * 2 - 1)
        )
    else:
        losses["lpips_loss"] = torch.tensor(0.0, device=device)
    
    # Routing sparsity loss
    if routing_weights is not None and hasattr(cfg.opt, 'lambda_sparse') and cfg.opt.lambda_sparse > 0:
        losses["sparse_loss"] = routing_sparsity_loss(routing_weights, cfg.opt.lambda_sparse)
    else:
        losses["sparse_loss"] = torch.tensor(0.0, device=device)
    
    # Feature consistency loss
    if (feat_with_2d is not None and feat_without_2d is not None and 
        hasattr(cfg.opt, 'lambda_consistency') and cfg.opt.lambda_consistency > 0):
        losses["consistency_loss"] = feature_consistency_loss(feat_with_2d, feat_without_2d)
    else:
        losses["consistency_loss"] = torch.tensor(0.0, device=device)
    
    # Total loss
    losses["total_loss"] = (
        losses["l12_loss"]
        + losses["lpips_loss"] * cfg.opt.lambda_lpips
        + losses["sparse_loss"]
        + losses["consistency_loss"] * getattr(cfg.opt, 'lambda_consistency', 0.0)
    )
    
    return losses
