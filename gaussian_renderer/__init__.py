# Adapted from https://github.com/graphdeco-inria/gaussian-splatting/tree/main
# to take in a predicted dictionary with 3D Gaussian parameters.

import math
import torch
import numpy as np

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # type: ignore
    _HAS_DGR = True
except ImportError:  # pragma: no cover - fallback path
    GaussianRasterizationSettings = None  # type: ignore
    GaussianRasterizer = None  # type: ignore
    _HAS_DGR = False


from utils.graphics_utils import focal2fov

def render_predicted(pc : dict,
                     world_view_transform,
                     full_proj_transform,
                     camera_center,
                     bg_color : torch.Tensor,
                     cfg,
                     scaling_modifier = 1.0,
                     override_color = None,
                     focals_pixels = None):
    if _HAS_DGR:
        return _render_with_dgr(
            pc,
            world_view_transform,
            full_proj_transform,
            camera_center,
            bg_color,
            cfg,
            scaling_modifier,
            override_color,
            focals_pixels,
        )

    return _render_fallback(
        pc,
        world_view_transform,
        full_proj_transform,
        bg_color,
        cfg,
        override_color,
    )


def _render_with_dgr(pc,
                     world_view_transform,
                     full_proj_transform,
                     camera_center,
                     bg_color,
                     cfg,
                     scaling_modifier,
                     override_color,
                     focals_pixels):
    """
    Render the scene as specified by pc dictionary. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc["xyz"], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    if focals_pixels == None:
        tanfovx = math.tan(cfg.data.fov * np.pi / 360)
        tanfovy = math.tan(cfg.data.fov * np.pi / 360)
    else:
        tanfovx = math.tan(focal2fov(focals_pixels[0].item(), cfg.data.training_resolution))
        tanfovy = math.tan(focal2fov(focals_pixels[1].item(), cfg.data.training_resolution))
        
    image_height = int(cfg.data.training_resolution) if hasattr(cfg.data, "training_resolution") else cfg.data.training_height
    image_width = int(cfg.data.training_resolution) if hasattr(cfg.data, "training_resolution") else cfg.data.training_width
    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg.model.max_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc["xyz"]
    means2D = screenspace_points
    opacity = pc["opacity"]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc["scaling"]
    rotations = pc["rotation"]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if "features_rest" in pc.keys():
            shs = torch.cat([pc["features_dc"], pc["features_rest"]], dim=1).contiguous()
        else:
            shs = pc["features_dc"]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def _render_fallback(pc,
                     world_view_transform,
                     full_proj_transform,
                     bg_color,
                     cfg,
                     override_color=None):
    """
    Differentiable fallback point splatting implemented purely in PyTorch.
    It does not model true anisotropic 3D Gaussians but provides a reasonable
    approximation for debugging when the CUDA rasterizer is unavailable.
    """

    device = pc["xyz"].device
    dtype = pc["xyz"].dtype
    bg = bg_color.to(device=device, dtype=dtype).view(-1, 1, 1)

    image_height = int(getattr(cfg.data, "training_resolution", cfg.data.training_height))
    image_width = int(getattr(cfg.data, "training_resolution", cfg.data.training_width))

    colors = _extract_colors(pc, override_color).to(device=device, dtype=dtype)
    opacities = torch.sigmoid(pc["opacity"].reshape(-1, 1))

    screenspace_points, visibility = _project_points(
        pc["xyz"],
        world_view_transform,
        full_proj_transform,
        image_width,
        image_height,
    )

    accum_image = torch.ones(3, image_height, image_width, device=device, dtype=dtype) * bg

    valid_coords = screenspace_points[visibility]
    valid_colors = colors[visibility]
    valid_opacity = opacities[visibility]
    valid_scales = pc["scaling"][visibility]

    if valid_coords.shape[0] == 0:
        radii = torch.zeros(pc["xyz"].shape[0], device=device, dtype=dtype)
        return {
            "render": accum_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": visibility,
            "radii": radii,
        }

    radii = torch.zeros(pc["xyz"].shape[0], device=device, dtype=dtype)
    approx_radius = valid_scales.mean(dim=-1) * max(image_height, image_width) * 0.02
    radii[visibility] = approx_radius

    image = torch.ones_like(accum_image) * bg

    flat_image = image.view(3, -1)
    lin_indices, bilinear_weights = _bilinear_indices(valid_coords, image_width, image_height)

    weighted_colors = (valid_colors * valid_opacity)[:, :, None] * bilinear_weights.unsqueeze(1)
    for i in range(4):
        idx = lin_indices[:, i]
        contrib = weighted_colors[:, :, i]
        flat_image.scatter_add_(1, idx.unsqueeze(0).expand(3, -1), contrib.t())

    final_image = torch.clamp(image, 0.0, 1.0)

    return {
        "render": final_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility,
        "radii": radii,
    }


def _extract_colors(pc, override_color):
    if override_color is not None:
        return override_color
    feats = pc.get("features_dc", None)
    if feats is None:
        raise ValueError("features_dc missing in point cloud dictionary.")
    if feats.shape[1] < 3:
        pad = torch.zeros(feats.shape[0], 3 - feats.shape[1], device=feats.device, dtype=feats.dtype)
        feats = torch.cat([feats, pad], dim=1)
    return torch.sigmoid(feats[:, :3])


def _project_points(xyz, world_view_transform, full_proj_transform, width, height):
    device = xyz.device
    dtype = xyz.dtype
    ones = torch.ones((xyz.shape[0], 1), device=device, dtype=dtype)
    homo = torch.cat([xyz, ones], dim=-1)

    world_view = world_view_transform.to(device=device, dtype=dtype)
    full_proj = full_proj_transform.to(device=device, dtype=dtype)

    view = world_view @ homo.t()
    clip = full_proj @ view
    clip = clip.t()

    w = clip[:, 3:4]
    ndc = clip[:, :3] / (w + 1e-9)

    x = (ndc[:, 0] * 0.5 + 0.5) * (width - 1)
    y = (1 - (ndc[:, 1] * 0.5 + 0.5)) * (height - 1)
    depth = ndc[:, 2]
    screenspace = torch.stack([x, y, depth], dim=-1)

    visibility = (
        (w[:, 0] > 0)
        & (ndc[:, 0].abs() <= 1)
        & (ndc[:, 1].abs() <= 1)
        & (ndc[:, 2].abs() <= 1)
    )
    return screenspace, visibility


def _bilinear_indices(coords, width, height):
    x = coords[:, 0].clamp(0, width - 1)
    y = coords[:, 1].clamp(0, height - 1)

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)

    wx = x - x0
    wy = y - y0

    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy

    idx00 = (y0 * width + x0).long()
    idx01 = (y1 * width + x0).long()
    idx10 = (y0 * width + x1).long()
    idx11 = (y1 * width + x1).long()

    lin_indices = torch.stack([idx00, idx10, idx01, idx11], dim=-1)
    weights = torch.stack([w00, w10, w01, w11], dim=-1)
    return lin_indices, weights
