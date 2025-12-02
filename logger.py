import os
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import glob
from typing import Dict, List
import pointcept.utils.comm as comm


class Logger:
    """Manages all logging operations with automatic fallback for offline mode"""

    def __init__(self, cfg: DictConfig, vis_dir: str):
        self.cfg = cfg
        self.vis_dir = vis_dir
        self.wandb_run = None
        
        # Create directories BEFORE setup_wandb (which may reference them)
        self.videos_dir = os.path.join(self.vis_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        self.logs_dir = os.path.join(self.vis_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Now setup wandb
        self.is_wandb_available = self.setup_wandb()

    def check_network_availability(self) -> bool:
        """
        Quick network check without hanging on retries

        Returns:
            bool: True if network is available, False otherwise
        """
        import socket
        try:
            # Try to connect to Google DNS with short timeout
            socket.setdefaulttimeout(2)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
            return True
        except Exception:
            return False

    def setup_wandb(self) -> bool:
        """
        Initialize W&B logging with automatic offline mode detection

        Returns:
            bool: True if wandb was successfully initialized, False if running in offline mode
        """
        # First check if network is available
        if not self.check_network_availability():
            print("=" * 80)
            print("No network connection detected. Running in OFFLINE mode.")
            print(f"Logs will be saved to: {self.logs_dir}")
            print(f"Videos will be saved to: {self.videos_dir}")
            print("=" * 80)
            return False
        
        # Try to import and use wandb
        try:
            import wandb
            
            # Set wandb to offline mode to prevent hanging
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_SILENT"] = "true"
            
            dict_cfg = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)

            try:
                if os.path.isdir(os.path.join(self.vis_dir, "wandb")):
                    # Resume existing run
                    run_name_path = glob.glob(
                        os.path.join(self.vis_dir, "wandb", "latest-run", "run-*")
                    )[0]
                    run_id = (
                        os.path.basename(run_name_path).split("run-")[1].split(".wandb")[0]
                    )
                    self.wandb_run = wandb.init(
                        project=self.cfg.wandb.project,
                        resume=True,
                        id=run_id,
                        config=dict_cfg,
                        mode="offline",
                    )
                else:
                    # Start new run
                    self.wandb_run = wandb.init(
                        project=self.cfg.wandb.project, 
                        reinit=True, 
                        config=dict_cfg,
                        mode="offline",
                    )
                print("=" * 80)
                print("W&B initialized in OFFLINE mode (logs saved locally)")
                print(f"W&B logs will be saved to: {self.vis_dir}/wandb")
                print("You can sync later with: wandb sync <path>")
                print("=" * 80)
                return True
            except Exception as e:
                print("=" * 80)
                print(f"Warning: Failed to initialize wandb ({str(e)})")
                print("Running in OFFLINE mode without wandb")
                print(f"Logs will be saved to: {self.logs_dir}")
                print("=" * 80)
                self.wandb_run = None
                return False
        except ImportError:
            print("=" * 80)
            print("Warning: wandb not installed. Running in OFFLINE mode")
            print(f"Logs will be saved to: {self.logs_dir}")
            print("=" * 80)
            return False

    def log_validation_progress(
        self, scores: Dict[str, torch.Tensor], iteration: int, lr: float = None
    ) -> None:
        """Log validation progress with fallback for offline mode"""
        if not (
            (comm.get_rank() == 0 and self.cfg.general.multiple_gpu)
            or not self.cfg.general.multiple_gpu
        ):
            return

        if self.is_wandb_available and self.wandb_run is not None:
            import wandb
            wandb.log(scores, step=iteration)
        
        print(f"@ Iteration {iteration} Val:", end="")
        print(scores)
        if lr is not None:
            print(f"Learning rate: {lr:.6f}")
        
        # Save to log file
        log_file = os.path.join(self.logs_dir, "validation_log.txt")
        with open(log_file, "a") as f:
            f.write(f"Iteration {iteration}: {scores}")
            if lr is not None:
                f.write(f" | LR: {lr:.6f}")
            f.write("\n")

    def _check_main_process(self) -> bool:
        """
        Check if the current process is the main process for logging

        Returns:
            bool: True if this is the main process, False otherwise
        """
        return (
            (comm.get_rank() == 0 and self.cfg.general.multiple_gpu)
            or not self.cfg.general.multiple_gpu
        )

    def log_training_progress(
        self, loss_dict: Dict[str, torch.Tensor], iteration: int
    ) -> None:
        """Log training progress with fallback for offline mode"""
        if not self._check_main_process():
            return

        # Print to console
        log_msg = f"@ Iteration {iteration}:"
        log_msg += f"  Training log10 loss: {np.log10(loss_dict['total_loss'].item() + 1e-8):.4f}"
        
        if "l12_loss" in loss_dict:
            log_msg += f"  L12 log10 loss: {np.log10(loss_dict['l12_loss'].item() + 1e-8):.4f}"
        if "lpips_loss" in loss_dict:
            log_msg += f"  LPIPS loss: {np.log10(loss_dict['lpips_loss'].item() + 1e-8):.4f}"
        if "sparse_loss" in loss_dict and loss_dict["sparse_loss"].item() > 0:
            log_msg += f"  Sparse loss: {loss_dict['sparse_loss'].item():.4f}"
        if "consistency_loss" in loss_dict and loss_dict["consistency_loss"].item() > 0:
            log_msg += f"  Consistency loss: {loss_dict['consistency_loss'].item():.4f}"
        
        print(log_msg)
        
        # Save to log file
        log_file = os.path.join(self.logs_dir, "training_log.txt")
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

        if self.is_wandb_available and self.wandb_run is not None:
            # Log to wandb if available
            import wandb
            wandb.log(
                {"training_loss": np.log10(loss_dict["total_loss"].item() + 1e-8)},
                step=iteration,
            )

            if "l12_loss" in loss_dict:
                wandb.log(
                    {"training_l12_loss": np.log10(loss_dict["l12_loss"].item() + 1e-8)},
                    step=iteration,
                )

            if "lpips_loss" in loss_dict:
                wandb.log(
                    {
                        "training_lpips_loss": np.log10(
                            loss_dict["lpips_loss"].item() + 1e-8
                        )
                    },
                    step=iteration,
                )
            
            if "sparse_loss" in loss_dict:
                wandb.log(
                    {"routing_sparse_loss": loss_dict["sparse_loss"].item()},
                    step=iteration,
                )
            
            if "consistency_loss" in loss_dict:
                wandb.log(
                    {"feature_consistency_loss": loss_dict["consistency_loss"].item()},
                    step=iteration,
                )

    def log_test_videos(
        self,
        test_loop: List[np.ndarray],
        test_loop_gt: List[np.ndarray],
        iteration: int,
        test_generate_num: int = None,
    ) -> None:
        """
        Log test videos to wandb or save locally with fallback for offline mode.

        Args:
            test_loop: List of rendered test images for video
            test_loop_gt: List of ground truth test images for video
            iteration: Current training iteration
            test_generate_num: Test generation number for multiple test cases
        """
        if not self._check_main_process():
            return
        
        if self.is_wandb_available and self.wandb_run is not None:
            # Log to wandb
            import wandb
            if test_loop is not None:
                video_name = (
                    f"rot_{test_generate_num}"
                    if test_generate_num is not None
                    else "rot"
                )
                wandb.log(
                    {
                        video_name: wandb.Video(
                            np.asarray(test_loop), fps=10, format="mp4"
                        )
                    },
                    step=iteration,
                )

            if test_loop_gt is not None:
                video_name = (
                    f"rot_gt_{test_generate_num}"
                    if test_generate_num is not None
                    else "rot_gt"
                )
                wandb.log(
                    {
                        video_name: wandb.Video(
                            np.asarray(test_loop_gt), fps=5, format="mp4"
                        )
                    },
                    step=iteration,
                )
        else:
            # Save locally using imageio
            try:
                import imageio.v3 as iio
            except ImportError:
                print("Please install imageio with: pip install imageio imageio-ffmpeg")
                return

            print(
                f"@ Iteration {iteration}: Saving videos locally to {self.videos_dir}"
            )

            def process_frames(frames):
                """Convert frames to correct format for video saving"""
                return [
                    # Convert from (C, H, W) to (H, W, C) and ensure uint8
                    (
                        (frame.transpose(1, 2, 0) if frame.shape[0] == 3 else frame)
                        if frame.dtype == np.uint8
                        else (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    for frame in frames
                ]

            # Save predicted video
            if test_loop is not None:
                video_name = (
                    f"iter_{iteration}_rot_{test_generate_num}.mp4"
                    if test_generate_num is not None
                    else f"iter_{iteration}_rot.mp4"
                )
                video_path = os.path.join(self.videos_dir, video_name)
                iio.imwrite(
                    video_path,
                    process_frames(test_loop),
                    fps=10,
                    codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"],
                )
                print(f"Saved predicted video to {video_path}")

            # Save ground truth video
            if test_loop_gt is not None:
                video_name = (
                    f"iter_{iteration}_rot_gt_{test_generate_num}.mp4"
                    if test_generate_num is not None
                    else f"iter_{iteration}_rot_gt.mp4"
                )
                video_path = os.path.join(self.videos_dir, video_name)
                iio.imwrite(
                    video_path,
                    process_frames(test_loop_gt),
                    fps=5,
                    codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"],
                )
                print(f"Saved ground truth video to {video_path}")

    def finish(self):
        """Cleanup wandb run if it exists"""
        if self.wandb_run is not None:
            self.wandb_run.finish()
