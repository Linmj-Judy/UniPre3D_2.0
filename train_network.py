"""
Training script for 3D Gaussian Splatting model with multi-modal fusion support.
This implementation supports both single-GPU and distributed training.

Key features:
- Multi-modal fusion between 2D and 3D features
- EMA (Exponential Moving Average) support
- Distributed training support
- Flexible validation and checkpointing
"""

import os
import sys
import faulthandler
import signal

# Enable faulthandler to catch segmentation faults
# This will print a Python traceback when a segfault occurs
faulthandler.enable()
# Also dump traceback to file
faulthandler.dump_traceback_later(timeout=1, exit=False)

# Register signal handler for SIGSEGV (segmentation fault)
def segfault_handler(signum, frame):
    print("\n" + "="*80)
    print("SEGMENTATION FAULT DETECTED!")
    print("="*80)
    import traceback
    traceback.print_stack(frame)
    print("\n" + "="*80)
    faulthandler.dump_traceback()
    sys.exit(1)

signal.signal(signal.SIGSEGV, segfault_handler)

import numpy as np
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
from ema_pytorch import EMA
import torch.distributed as dist
import pointcept.utils.comm as comm
from pointcept.engines.launch import launch
from pointcept.engines.defaults import create_ddp_model, worker_init_fn
from pointcept.datasets import point_collate_fn
import traceback
import sys

from model.gaussian_predictor import GaussianSplatPredictor
from dataset.dataset_factory import get_dataset
from gaussian_renderer import render_predicted
from eval import evaluate_dataset
from utils.general_utils import safe_state, to_device,prepare_model_inputs

from utils.loss_utils import (
    l1_loss, 
    l2_loss, 
    focal_l2_loss,
    compute_total_loss,
    feature_consistency_loss,
    routing_sparsity_loss,
)
import lpips as lpips_lib
from typing import Dict, List, Tuple
from functools import partial
import multiprocessing

from logger import Logger


def safe_print(msg):
    """安全的打印函数，避免在多进程环境中出现问题"""
    try:
        print(msg, flush=True)
        sys.stdout.flush()
    except:
        pass


class DataManager:
    """Manages all data loading and processing operations"""

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.setup_dataloaders()

    def setup_dataloaders(self) -> None:
        """Initialize all data loaders"""
        try:
            safe_print("[DataManager] 开始初始化数据集...")
            self.dataset = get_dataset(self.cfg, "train", device=self.device)
            safe_print(f"[DataManager] 训练集初始化完成，大小: {len(self.dataset)}")
            
            self.val_dataset = get_dataset(self.cfg, "val", device=self.device)
            safe_print(f"[DataManager] 验证集初始化完成，大小: {len(self.val_dataset)}")
            
            self.test_dataset = get_dataset(self.cfg, "test", device=self.device)
            safe_print(f"[DataManager] 测试集初始化完成，大小: {len(self.test_dataset)}")
        except Exception as e:
            safe_print(f"[DataManager] 数据集初始化失败: {str(e)}")
            safe_print(f"[DataManager] 错误堆栈:\n{traceback.format_exc()}")
            raise

        # Setup distributed sampling
        self.train_sampler = (
            torch.utils.data.distributed.DistributedSampler(self.dataset)
            if comm.get_world_size() > 1
            else None
        )
        self.val_sampler = (
            torch.utils.data.distributed.DistributedSampler(self.val_dataset)
            if comm.get_world_size() > 1
            else None
        )

        # Calculate batch size per GPU
        self.bs_per_gpu = (
            self.cfg.opt.batch_size
            if not self.cfg.general.multiple_gpu
            else self.cfg.opt.batch_size // len(self.cfg.general.device)
        )

        self.init_fn = self._get_worker_init_fn()
        self._create_dataloaders()

    def _get_worker_init_fn(self):
        """Get worker initialization function for data loading"""
        if self.cfg.general.random_seed is not None and self.cfg.general.multiple_gpu:
            return partial(
                worker_init_fn,
                num_workers=len(self.cfg.general.device) * 4,
                rank=comm.get_rank(),
                seed=self.cfg.general.random_seed,
            )
        return None

    def _create_dataloaders(self) -> None:
        """Create data loaders based on model type"""
        common_loader_params = {
            "num_workers": 0,
            "collate_fn": point_collate_fn if self.cfg.opt.level == "scene" else None,
        }

        if self.cfg.opt.level == "scene":
            self.train_loader = DataLoader(
                self.dataset,
                batch_size=self.bs_per_gpu,
                shuffle=(self.train_sampler is None),
                drop_last=True,
                sampler=self.train_sampler,
                worker_init_fn=self.init_fn,
                **common_loader_params,
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.bs_per_gpu,
                shuffle=False,
                drop_last=False,
                sampler=self.val_sampler,
                worker_init_fn=self.init_fn,
                **common_loader_params,
            )
        else:
            self.train_loader = DataLoader(
                self.dataset,
                batch_size=self.cfg.opt.batch_size,
                shuffle=True,
                drop_last=False,  # Allow incomplete batches for small datasets
                **common_loader_params,
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.opt.batch_size,
                shuffle=True,
                **common_loader_params,
            )

        # Test loader configuration remains same for both cases
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=True, **common_loader_params
        )


class ModelManager:
    """Manages model creation, optimization and checkpointing"""

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model, self.optimizer, self.scheduler = self._create_model_and_optimizer()
        self.model = self.model.to(device)
        self.setup_distributed()
        self.setup_ema()

    def _create_model_and_optimizer(
        self,
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        """Create and initialize model and optimizer"""
        try:
            safe_print("[ModelManager] 开始创建模型...")
            model_class = GaussianSplatPredictor
            model = model_class(self.cfg)
            safe_print("[ModelManager] 模型创建完成")

            safe_print("[ModelManager] 开始创建优化器...")
            optimizer_params = self._get_optimizer_params(model)
            optimizer = torch.optim.AdamW(
                optimizer_params, lr=0.0, eps=1e-15, betas=self.cfg.opt.betas
            )
            safe_print("[ModelManager] 优化器创建完成")
            
            scheduler = None
            if self.cfg.opt.step_lr != -1:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.cfg.opt.step_lr, gamma=self.cfg.opt.lr_gamma
                )
                safe_print("[ModelManager] 学习率调度器创建完成")
            return model, optimizer, scheduler
        except Exception as e:
            safe_print(f"[ModelManager] 模型/优化器创建失败: {str(e)}")
            safe_print(f"[ModelManager] 错误堆栈:\n{traceback.format_exc()}")
            raise

    def _get_optimizer_params(self, model: torch.nn.Module) -> List[Dict]:
        """Get optimizer parameters based on model type"""
        base_lr = self.cfg.opt.base_lr
        params = [{"params": model.point_network.parameters(), "lr": base_lr}]

        if self.cfg.opt.use_fusion:
            fusion_params = [
                {"params": model.fusion_mlps.parameters(), "lr": base_lr},
                {"params": model.image_conv.parameters(), "lr": base_lr},
            ]
            params.extend(fusion_params)

        return params

    def setup_distributed(self) -> None:
        """Setup distributed training if needed"""
        if self.cfg.general.multiple_gpu:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = create_ddp_model(
                self.model.cuda(), broadcast_buffers=False, find_unused_parameters=True
            )

    def setup_ema(self) -> None:
        """Setup EMA if enabled"""
        if self.cfg.opt.ema.use:
            self.ema = EMA(
                self.model,
                beta=self.cfg.opt.ema.beta,
                update_every=self.cfg.opt.ema.update_every,
                update_after_step=self.cfg.opt.ema.update_after_step,
            )
        else:
            self.ema = None

    def save_checkpoint(self, iteration: int, best_psnr: float, save_path: str) -> None:
        """Save model checkpoint"""
        ckpt_save_dict = {
            "iteration": iteration,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_state_dict": (
                self.ema.ema_model.state_dict() if self.ema else self.model.state_dict()
            ),
            "best_PSNR": best_psnr,
        }
        torch.save(ckpt_save_dict, save_path)

    def save_latest_checkpoint(self, iteration: int, best_psnr: float, save_dir: str) -> None:
        """Save latest model checkpoint"""
        save_path = os.path.join(save_dir, "model_latest.pth")
        self.save_checkpoint(iteration, best_psnr, save_path)

    def save_best_checkpoint(self, iteration: int, best_psnr: float, save_dir: str) -> None:
        """Save best model checkpoint"""
        save_path = os.path.join(save_dir, "model_best.pth")
        self.save_checkpoint(iteration, best_psnr, save_path)

class ValidationManager:
    """Manages model validation and evaluation"""

    def __init__(self, cfg: DictConfig, device: torch.device, logger):
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.lpips_fn = (
            lpips_lib.LPIPS(net="vgg").to(device) if cfg.opt.lambda_lpips != 0 else None
        )

    def validate_model(
        self, model: torch.nn.Module, val_loader: DataLoader, iteration: int, lr: float = 0.0
    ) -> float:
        """Validate model performance"""
        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            scores = evaluate_dataset(
                model,
                val_loader,
                device=self.device,
                model_cfg=self.cfg,
            )
            self.logger.log_validation_progress(
                scores,
                iteration,
                lr=lr
            )

            psnr = torch.tensor(scores["PSNR_novel"]).to(self.device)
            if self.cfg.general.multiple_gpu:
                dist.all_reduce(psnr, op=dist.ReduceOp.SUM)
                psnr /= dist.get_world_size()

        return psnr.item()

    def calculate_losses(
        self,
        rendered_images: torch.Tensor,
        gt_images: torch.Tensor,
        iteration: int,
        routing_weights: torch.Tensor = None,
        feat_with_2d: torch.Tensor = None,
        feat_without_2d: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Calculate all training losses including improved regularization terms"""
        losses = {}

        # Calculate reconstruction loss
        background = torch.tensor(
            [1, 1, 1] if self.cfg.data.white_background else [0, 0, 0],
            dtype=torch.float32,
            device=self.device,
        )

        if self.cfg.opt.loss == "focal_l2":
            losses["l12_loss"] = focal_l2_loss(
                rendered_images,
                gt_images,
                background,
                self.cfg.opt.non_bg_color_loss_rate,
                self.cfg.opt.bg_color_loss_rate,
            )
        else:
            loss_func = l1_loss if self.cfg.opt.loss == "l1" else l2_loss
            losses["l12_loss"] = loss_func(rendered_images, gt_images)

        # Add LPIPS loss if enabled
        if (
            self.cfg.opt.lambda_lpips != 0
            and iteration > self.cfg.opt.start_lpips_after
        ):
            losses["lpips_loss"] = torch.mean(
                self.lpips_fn(rendered_images * 2 - 1, gt_images * 2 - 1)
            )
        else:
            losses["lpips_loss"] = torch.tensor(0.0, device=self.device)

        # Add routing sparsity loss if routing is enabled
        if routing_weights is not None and hasattr(self.cfg.opt, 'lambda_sparse'):
            losses["sparse_loss"] = routing_sparsity_loss(
                routing_weights, self.cfg.opt.lambda_sparse
            )
        else:
            losses["sparse_loss"] = torch.tensor(0.0, device=self.device)

        # Add feature consistency loss if dual forward is enabled
        if (feat_with_2d is not None and feat_without_2d is not None and 
            hasattr(self.cfg.opt, 'lambda_consistency')):
            losses["consistency_loss"] = feature_consistency_loss(
                feat_with_2d, feat_without_2d, stop_gradient=True
            )
        else:
            losses["consistency_loss"] = torch.tensor(0.0, device=self.device)

        # Calculate total loss
        losses["total_loss"] = (
            losses["l12_loss"] 
            + losses["lpips_loss"] * self.cfg.opt.lambda_lpips
            + losses["sparse_loss"]
            + losses["consistency_loss"] * getattr(self.cfg.opt, 'lambda_consistency', 0.0)
        )

        return losses


class Trainer:
    """Main trainer class that orchestrates the training process"""

    def __init__(self, cfg: DictConfig):
        try:
            safe_print("[Trainer] 开始初始化Trainer...")
            self.vis_dir = os.getcwd()
            safe_print(f"[Trainer] 工作目录: {self.vis_dir}")
            
            safe_print("[Trainer] 初始化设备状态...")
            self.device = safe_state(cfg)
            safe_print(f"[Trainer] 设备: {self.device}")
            self.cfg = cfg

            # Initialize components
            safe_print("[Trainer] 初始化Logger...")
            self.logger = Logger(cfg, self.vis_dir)
            safe_print("[Trainer] Logger初始化完成")
            
            safe_print("[Trainer] 初始化DataManager...")
            self.data_manager = DataManager(cfg, self.device)
            safe_print("[Trainer] DataManager初始化完成")
            
            safe_print("[Trainer] 初始化ModelManager...")
            self.model_manager = ModelManager(cfg, self.device)
            safe_print("[Trainer] ModelManager初始化完成")
            
            safe_print("[Trainer] 初始化ValidationManager...")
            self.validation_manager = ValidationManager(cfg, self.device, self.logger)
            safe_print("[Trainer] ValidationManager初始化完成")

            self.best_psnr = 0.0
            
            # Initialize router temperature annealing if routing is enabled
            if getattr(cfg.opt, 'use_routing', False):
                self.router_temp_start = cfg.opt.router_temp_start
                self.router_temp_end = cfg.opt.router_temp_end
                self.router_temp_anneal_iters = cfg.opt.router_temp_anneal_iters
            
            safe_print("[Trainer] Trainer初始化完成！")
        except Exception as e:
            safe_print(f"[Trainer] Trainer初始化失败: {str(e)}")
            safe_print(f"[Trainer] 错误堆栈:\n{traceback.format_exc()}")
            raise

    def train(self) -> None:
        """Main training loop"""
        for iteration in range(1, self.cfg.opt.iterations + 1):
            if self.cfg.opt.mode != "test":
                # Set sampler epoch to ensure different samples across GPUs
                if self.cfg.general.multiple_gpu:
                    self.data_manager.train_sampler.set_epoch(iteration)
                
                # Training step
                loss_dict = self.train_iteration(iteration)

                # Optimizer step
                loss_dict["total_loss"].backward()
                
                # Check gradients
                if not self._check_and_clip_gradients():
                    if (not self.cfg.general.multiple_gpu) or (comm.get_rank() == 0 and self.cfg.general.multiple_gpu):
                        print("Warning! Exiting training due to NaN gradients.")
                    self.model_manager.optimizer.zero_grad()
                    continue
                
                # Update model parameters
                self.model_manager.optimizer.step()
                self.model_manager.optimizer.zero_grad()

                # Step scheduler if enabled
                if self.model_manager.scheduler is not None:
                    self.model_manager.scheduler.step()

                # Update EMA if enabled
                if self.model_manager.ema:
                    self.model_manager.ema.update()

                # Logging
                if iteration % self.cfg.logging.loss_log == 0:
                    self.logger.log_training_progress(loss_dict, iteration)

                # Validation
                if iteration % self.cfg.logging.val_log == 0:
                    self.validate(iteration)

                # Generating test examples
                if iteration % self.cfg.logging.loop_log == 0 or iteration == 1:
                    self.generate_test_examples(iteration)

        self.logger.finish()

    def _check_and_clip_gradients(self) -> bool:
        """Check for invalid gradients (NaN) and apply gradient clipping if valid.
        
        Returns:
            bool: True if gradients are valid and clipping was applied, 
                False if NaN gradients were detected.
        """
        # Check for NaN gradients
        has_invalid_gradients = any(
            torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            for param in self.model_manager.model.parameters()
            if param.grad is not None
        )
        
        if has_invalid_gradients:
            return False
        
        # Apply gradient clipping if gradients are valid
        torch.nn.utils.clip_grad_norm_(
            self.model_manager.model.parameters(),
            max_norm=1.0,
        )
        return True
    
    def render_validation_views(
        self, gaussian_splats: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render validation views using the predicted Gaussian splats.

        Args:
            gaussian_splats: Dictionary containing predicted Gaussian splat parameters
            data: Dictionary containing ground truth data and camera parameters

        Returns:
            tuple: (rendered_images, gt_images)
                - rendered_images: Tensor of rendered novel views
                - gt_images: Tensor of corresponding ground truth images
        """
        rendered_images = []
        gt_images = []

        try:
            # Set background color based on configuration
            background = torch.tensor(
                [1, 1, 1] if self.cfg.data.white_background else [0, 0, 0],
                dtype=torch.float32,
                device=self.device,
            )

            batch_size = data["gt_images"].shape[0]
            num_views = data["gt_images"].shape[1]
            safe_print(f"[Trainer] 渲染验证视图: batch_size={batch_size}, num_views={num_views}, input_images={self.cfg.data.input_images}")

            # Process each batch
            for b_idx in range(batch_size):
                safe_print(f"[Trainer] 处理批次 {b_idx}/{batch_size}...")
                try:
                    # Extract gaussian parameters for current batch
                    gaussian_splat_batch = {
                        k: v[b_idx].contiguous()
                        for k, v in gaussian_splats.items()
                        if len(v.shape) > 1
                    }
                    safe_print(f"[Trainer] 批次 {b_idx}: Gaussian splat参数提取完成，键: {list(gaussian_splat_batch.keys())}")
                    
                    # 检查参数形状
                    for k, v in gaussian_splat_batch.items():
                        safe_print(f"[Trainer] 批次 {b_idx}: {k} 形状: {v.shape}, dtype: {v.dtype}, device: {v.device}")

                    # Render each validation view
                    for r_idx in range(self.cfg.data.input_images, num_views):
                        safe_print(f"[Trainer] 批次 {b_idx}, 视图 {r_idx}/{num_views}: 开始渲染...")
                        try:
                            # 检查变换矩阵
                            world_view = data["world_view_transforms"][b_idx, r_idx].to(self.device)
                            full_proj = data["full_proj_transforms"][b_idx, r_idx].to(self.device)
                            camera_center = data["camera_centers"][b_idx, r_idx].to(self.device)
                            
                            safe_print(f"[Trainer] 批次 {b_idx}, 视图 {r_idx}: 变换矩阵形状 - world_view: {world_view.shape}, full_proj: {full_proj.shape}, camera_center: {camera_center.shape}")
                            
                            # 渲染
                            render_result = render_predicted(
                                gaussian_splat_batch,
                                world_view,
                                full_proj,
                                camera_center,
                                background,
                                self.cfg,
                                focals_pixels=None,
                            )
                            
                            image = render_result["render"]
                            safe_print(f"[Trainer] 批次 {b_idx}, 视图 {r_idx}: 渲染完成，图像形状: {image.shape}")

                            gt_image = data["gt_images"][b_idx, r_idx].to(self.device)

                            rendered_images.append(image)
                            gt_images.append(gt_image)
                        except Exception as e:
                            safe_print(f"[Trainer] 批次 {b_idx}, 视图 {r_idx}: 渲染失败: {str(e)}")
                            safe_print(f"[Trainer] 批次 {b_idx}, 视图 {r_idx}: 错误堆栈:\n{traceback.format_exc()}")
                            if torch.cuda.is_available():
                                safe_print(f"[Trainer] GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB")
                            raise
                except Exception as e:
                    safe_print(f"[Trainer] 批次 {b_idx}: 处理失败: {str(e)}")
                    safe_print(f"[Trainer] 批次 {b_idx}: 错误堆栈:\n{traceback.format_exc()}")
                    raise

            # Stack all images into tensors
            safe_print(f"[Trainer] 堆叠图像，共 {len(rendered_images)} 张...")
            rendered_images = torch.stack(rendered_images, dim=0)
            gt_images = torch.stack(gt_images, dim=0)
            safe_print(f"[Trainer] 堆叠完成: rendered_images形状={rendered_images.shape}, gt_images形状={gt_images.shape}")

            return rendered_images, gt_images
        except Exception as e:
            safe_print(f"[Trainer] render_validation_views失败: {str(e)}")
            safe_print(f"[Trainer] 完整错误堆栈:\n{traceback.format_exc()}")
            if torch.cuda.is_available():
                safe_print(f"[Trainer] GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB")
            raise

    def update_router_temperature(self, iteration: int):
        """Update router temperature for Gumbel-Softmax annealing"""
        if getattr(self.cfg.opt, 'use_routing', False):
            progress = min(iteration / self.router_temp_anneal_iters, 1.0)
            current_temp = self.router_temp_start + (self.router_temp_end - self.router_temp_start) * progress
            
            # Update temperature in model's router
            if hasattr(self.model_manager.model, 'module'):
                # DDP model
                if hasattr(self.model_manager.model.module, 'router'):
                    self.model_manager.model.module.router.set_temperature(current_temp)
            else:
                if hasattr(self.model_manager.model, 'router'):
                    self.model_manager.model.router.set_temperature(current_temp)
    
    def train_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
        """Execute one training iteration"""
        try:
            # 步骤1: 数据加载
            safe_print(f"[Trainer] Iteration {iteration}: 开始加载数据...")
            try:
                data = next(iter(self.data_manager.train_loader))
                safe_print(f"[Trainer] Iteration {iteration}: 数据加载完成")
                safe_print(f"[Trainer] Iteration {iteration}: 数据键: {list(data.keys())}")
                if 'gt_images' in data:
                    safe_print(f"[Trainer] Iteration {iteration}: gt_images形状: {data['gt_images'].shape}")
            except Exception as e:
                safe_print(f"[Trainer] Iteration {iteration}: 数据加载失败: {str(e)}")
                safe_print(f"[Trainer] Iteration {iteration}: 错误堆栈:\n{traceback.format_exc()}")
                raise

            # 步骤2: 准备模型输入
            safe_print(f"[Trainer] Iteration {iteration}: 准备模型输入...")
            try:
                model_inputs = prepare_model_inputs(data, self.cfg, self.data_manager.bs_per_gpu, self.device)
                safe_print(f"[Trainer] Iteration {iteration}: 模型输入准备完成")
                safe_print(f"[Trainer] Iteration {iteration}: 模型输入键: {list(model_inputs.keys())}")
            except Exception as e:
                safe_print(f"[Trainer] Iteration {iteration}: 模型输入准备失败: {str(e)}")
                safe_print(f"[Trainer] Iteration {iteration}: 错误堆栈:\n{traceback.format_exc()}")
                raise

            # 步骤3: 更新路由温度
            try:
                self.update_router_temperature(iteration)
            except Exception as e:
                safe_print(f"[Trainer] Iteration {iteration}: 路由温度更新失败: {str(e)}")

            # 步骤4: 模型前向传播
            safe_print(f"[Trainer] Iteration {iteration}: 开始模型前向传播...")
            self.model_manager.model.train()
        
            # Check if dual forward pass is needed for consistency loss
            use_dual_forward = getattr(self.cfg.opt, 'use_dual_forward', False)
            routing_weights = None
            feat_with_2d = None
            feat_without_2d = None
        
            try:
                if use_dual_forward and self.cfg.opt.use_fusion:
                    # Forward pass WITH 2D features
                    safe_print(f"[Trainer] Iteration {iteration}: 执行双前向传播（带2D特征）...")
                    gaussian_splats = self.model_manager.model(**model_inputs)
                    safe_print(f"[Trainer] Iteration {iteration}: 前向传播（带2D）完成")
                
                    # Extract intermediate features if available
                    if hasattr(self.model_manager.model, 'get_intermediate_features'):
                        feat_with_2d = self.model_manager.model.get_intermediate_features()
                
                    # Forward pass WITHOUT 2D features (for consistency loss)
                    model_inputs_no_2d = model_inputs.copy()
                    model_inputs_no_2d['image'] = None
                
                    with torch.no_grad():
                        _ = self.model_manager.model(**model_inputs_no_2d)
                        if hasattr(self.model_manager.model, 'get_intermediate_features'):
                            feat_without_2d = self.model_manager.model.get_intermediate_features()
                else:
                    # Standard forward pass
                    safe_print(f"[Trainer] Iteration {iteration}: 执行标准前向传播...")
                    gaussian_splats = self.model_manager.model(**model_inputs)
                    safe_print(f"[Trainer] Iteration {iteration}: 前向传播完成")
                    safe_print(f"[Trainer] Iteration {iteration}: Gaussian splats键: {list(gaussian_splats.keys())}")
            except Exception as e:
                safe_print(f"[Trainer] Iteration {iteration}: 模型前向传播失败: {str(e)}")
                safe_print(f"[Trainer] Iteration {iteration}: 错误堆栈:\n{traceback.format_exc()}")
                # 检查GPU内存
                if torch.cuda.is_available():
                    safe_print(f"[Trainer] Iteration {iteration}: GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB")
                raise
            
            # 步骤5: 提取路由权重
            try:
                if hasattr(self.model_manager.model, 'get_routing_weights'):
                    routing_weights = self.model_manager.model.get_routing_weights()
            except Exception as e:
                safe_print(f"[Trainer] Iteration {iteration}: 提取路由权重失败: {str(e)}")
        
            # 步骤6: 渲染验证视图
            safe_print(f"[Trainer] Iteration {iteration}: 开始渲染验证视图...")
            try:
                rendered_images, gt_images = self.render_validation_views(gaussian_splats, data)
                safe_print(f"[Trainer] Iteration {iteration}: 渲染完成")
                safe_print(f"[Trainer] Iteration {iteration}: 渲染图像形状: {rendered_images.shape}")
            except Exception as e:
                safe_print(f"[Trainer] Iteration {iteration}: 渲染失败: {str(e)}")
                safe_print(f"[Trainer] Iteration {iteration}: 错误堆栈:\n{traceback.format_exc()}")
                # 检查GPU内存
                if torch.cuda.is_available():
                    safe_print(f"[Trainer] Iteration {iteration}: GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB")
                raise

            # 步骤7: 计算损失
            safe_print(f"[Trainer] Iteration {iteration}: 开始计算损失...")
            try:
                losses = self.validation_manager.calculate_losses(
                    rendered_images, gt_images, iteration,
                    routing_weights=routing_weights,
                    feat_with_2d=feat_with_2d,
                    feat_without_2d=feat_without_2d,
                )
                safe_print(f"[Trainer] Iteration {iteration}: 损失计算完成")
                return losses
            except Exception as e:
                safe_print(f"[Trainer] Iteration {iteration}: 损失计算失败: {str(e)}")
                safe_print(f"[Trainer] Iteration {iteration}: 错误堆栈:\n{traceback.format_exc()}")
                raise
        except Exception as e:
            safe_print(f"[Trainer] Iteration {iteration}: 训练迭代失败: {str(e)}")
            safe_print(f"[Trainer] Iteration {iteration}: 完整错误堆栈:\n{traceback.format_exc()}")
            raise

    def validate(self, iteration: int) -> None:
        """Perform validation and generate test videos"""
        current_psnr = self.validation_manager.validate_model(
            (
                self.model_manager.model
                if not self.model_manager.ema
                else self.model_manager.ema
            ),
            self.data_manager.val_loader,
            iteration,
            lr = (
                    self.model_manager.scheduler.get_last_lr()[0]
                    if self.model_manager.scheduler is not None
                    else self.cfg.opt.base_lr
                )
        )

        # Only save checkpoints on rank 0 to avoid conflicts in distributed training
        if comm.get_rank() == 0:
            # Always save latest checkpoint after each validation
            self.model_manager.save_latest_checkpoint(
                iteration, self.best_psnr, self.vis_dir
            )
            
            # Save best checkpoint if performance improved
            if current_psnr > self.best_psnr:
                self.best_psnr = current_psnr
                self.model_manager.save_best_checkpoint(
                    iteration, self.best_psnr, self.vis_dir
                )

    def generate_test_examples(self, iteration: int) -> None:
        """Generate test videos if needed"""
        # Get test data from test loader
        vis_data = next(iter(self.data_manager.test_loader))

        vis_data = to_device(vis_data, self.device)

        # Generate gaussian splats
        model_inputs = prepare_model_inputs(vis_data, self.cfg, self.data_manager.bs_per_gpu, self.device)

        gaussian_splats = self.model_manager.model(**model_inputs)

        # Generate test videos
        test_loop = []
        test_loop_gt = []
        # Set background color based on configuration
        background = torch.tensor(
            [1, 1, 1] if self.cfg.data.white_background else [0, 0, 0],
            dtype=torch.float32,
            device=self.device,
        )

        # Render each view
        for r_idx in range(vis_data["gt_images"].shape[1]):
            # Render predicted view
            test_image = render_predicted(
                {k: v[0].contiguous() for k, v in gaussian_splats.items()},
                vis_data["world_view_transforms"][:, r_idx],
                vis_data["full_proj_transforms"][:, r_idx],
                vis_data["camera_centers"][:, r_idx],
                background,
                self.cfg,
                focals_pixels=None,
            )["render"]

            test_loop.append(
                (np.clip(test_image.detach().cpu().numpy(), 0, 1) * 255).astype(
                    np.uint8
                )
            )

            # Add ground truth
            test_loop_gt.append(
                (
                    np.clip(
                        vis_data["gt_images"][0, r_idx].detach().cpu().numpy(),
                        0,
                        1,
                    )
                    * 255
                ).astype(np.uint8)
            )

        # Log videos
        self.logger.log_test_videos(
            test_loop,
            test_loop_gt,
            iteration,
            0,
        )


@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(cfg: DictConfig):
    """Main entry point for training"""
    with open_dict(cfg):
        # Handle both ListConfig and other types of device specifications
        if hasattr(cfg.general.device, '__len__') and not isinstance(cfg.general.device, str):
            cfg.general.multiple_gpu = len(cfg.general.device) > 1
        else:
            cfg.general.multiple_gpu = False

    torch_home = OmegaConf.select(cfg, "env.torch_home", default=None)
    if torch_home:
        torch_home = os.path.abspath(os.path.expanduser(str(torch_home)))
        os.makedirs(torch_home, exist_ok=True)
        os.environ["TORCH_HOME"] = torch_home


    # Only set multiprocessing start method if not already set and if needed
    # This avoids conflicts with other libraries that may have already set it
    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method is None:
            # Start method not set, set it to spawn for CUDA compatibility
            multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        # Start method already set by another process, skip
        pass
    
    if cfg.general.multiple_gpu:
        # Check if enough GPUs are available
        requested_gpus = len(cfg.general.device)
        available_gpus = torch.cuda.device_count()
        
        if requested_gpus > available_gpus:
            raise RuntimeError(
                f"配置要求使用 {requested_gpus} 个 GPU，但只有 {available_gpus} 个 GPU 可用。\n"
                f"请检查：\n"
                f"1. SLURM 脚本中的 --gres=gpu: 设置是否正确（当前要求 {requested_gpus} 个）\n"
                f"2. CUDA_VISIBLE_DEVICES 环境变量是否限制了可见的 GPU 数量\n"
                f"3. 或者修改配置中的 general.device 参数以匹配实际可用的 GPU 数量"
            )
        
        if requested_gpus != available_gpus:
            print(f"警告: 配置要求 {requested_gpus} 个 GPU，实际可用 {available_gpus} 个 GPU。")
            print(f"将使用前 {requested_gpus} 个 GPU: {list(cfg.general.device)}")
        
        launch(
            main_worker,
            num_gpus_per_machine=requested_gpus,
            dist_url="auto",
            cfg=(cfg,),
        )
    else:
        launch(main_worker, num_gpus_per_machine=1, dist_url="auto", cfg=(cfg,))


def main_worker(cfg: DictConfig):
    """Main training worker function"""
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
