"""Finetuning script for SAM2."""  # noqa: INP001

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from training.loss_fns import iou_loss, sigmoid_focal_loss

import nnx.data
from nnx.data.ctspine1k.dataset import CTSpine1K, CUDALoader, LoadingMode, SAMAdapter


@nnx.configurable
class SAM2FinetuningConfig:
    """Configurable finetuning unit for SAM2."""

    # Model configuration
    checkpoint_dir: str | None = None
    checkpoint: str = "../checkpoints/sam2.1_hiera_large.pt"
    model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    image_size: int = 1024

    # Training parameters
    amp_enabled: bool = False
    learning_rate: float = 1e-6
    weight_decay: float = 1e-3
    epochs: int = 3
    clip_norm: float = 1.0

    # Scheduler configuration
    step_size: float = 2
    lr_gamma: float = 0.1

    # Dataset configuration
    batch_size: int = 64
    validation_size: int = 64
    train_workers: int = 3
    validation_workers: int = 3
    prefetch_factor: int = 25
    shuffle: bool = True

    # Checkpoint and evaluation
    validation_steps: int = 1


class SAM2Finetuning:
    """Finetuning class to tune the parameters of SAM2."""

    def __init__(
        self,
        config: SAM2FinetuningConfig,
        t_dataloader: DataLoader | CUDALoader,
        v_dataloader: DataLoader | CUDALoader,
        *,
        from_checkpoint: bool = False,
        with_wandb: bool = False,
    ) -> None:
        """C'tor of SAM2Finetuning.

        Args:
            config: the finetuning config associated with this project.
            t_dataloader: dataloader for the training phase
            v_dataloader: dataloader for the training phase
            from_checkpoint: whether the parameters should be initialised
                from a checkpoint.
            with_wandb: if weights and biases should be used to log metrics

        Raises:
            ValueError: for the case when finetuning is ran without CUDA support or
                invalid config.

        """
        self._config = config
        self._t_dataloader = t_dataloader
        self._v_dataloader = v_dataloader

        if not torch.cuda.is_available():
            msg = "For finetuning please ensure to have a CUDA capable GPU."
            raise ValueError(msg)

        self._device = torch.device("cuda")
        sam2_model = build_sam2(
            self._config.model_config,
            self._config.checkpoint,
            self._device,
        )

        self._predictor = SAM2ImagePredictor(sam2_model)
        self._predictor.model.image_encoder.eval()
        self._predictor.model.sam_mask_decoder.train()
        self._predictor.model.sam_prompt_encoder.train()

        self._optimizer = torch.optim.AdamW(
            params=self._predictor.model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        self._scheduler = StepLR(
            self._optimizer,
            step_size=self._config.step_size,
            gamma=self._config.lr_gamma,
        )

        self._grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self._config.amp_enabled,
        )

        if from_checkpoint:
            self._load_checkpoint()

        self._wandb = with_wandb

        if self._wandb:
            wandb.init(project="SAM2 Finetuning")

    def tune(self) -> None:  # noqa: PLR0914
        """Tune the SAM2 model with the given data."""
        data_iter = tqdm(range(self._config.epochs), desc="Starting finetuning...")
        for epoch in data_iter:
            torch.cuda.empty_cache()

            val_str = ""
            for batch in self._t_dataloader:
                # performance suicide, but having a dedicated Trainer as in
                # sam2/training/trainer.py and sam2/training/utils/data_utils.py
                # would be overkill for finetuning on comparably small dataset
                accumulated_iou_losses = []
                accumulated_focal_losses = []

                batch_size = len(batch["masks"])
                print(batch_size, len(batch["masks"]))
                for batch_idx in range(batch_size):
                    if not len(batch["masks"][batch_idx]):
                        data_iter.set_description_str(
                            "Finetuning. Skipped batch with no masks.",
                        )
                        continue  # batch with no masks

                    masks = batch["masks"][batch_idx]
                    point_coords = batch["point_coords"][batch_idx]
                    point_labels = batch["point_labels"][batch_idx]
                    image = batch["images"][batch_idx]

                    iou_losses, focal_losses = self._training_step(
                        image=image,
                        masks=masks,
                        point_coords=point_coords,
                        point_labels=point_labels,
                    )

                    accumulated_iou_losses.extend(iou_losses)
                    accumulated_focal_losses.extend(focal_losses)

                if not accumulated_iou_losses:
                    print("No data in batch, continuing...")
                    continue

                iou_loss = sum(accumulated_iou_losses)
                focal_loss = sum(accumulated_focal_losses)
                loss = (iou_loss + focal_loss) / len(accumulated_iou_losses)

                desc_str = f"Finetuning. Current loss: {loss}" + val_str

                data_iter.set_description_str(desc_str)

                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._predictor.model.parameters(),
                    max_norm=self._config.clip_norm,
                )
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()

                self._predictor.model.zero_grad()

                if self._wandb:
                    wandb.log(
                        {
                            "Loss/IoU Loss": iou_loss.item(),
                            "Loss/Focal Loss": focal_loss.item(),
                            "Loss/Total Loss": loss.item(),
                        },
                    )

            if epoch and self._config.checkpoint_dir:
                self._save_checkpoint(epoch)

            if epoch and ((epoch + 1) % self._config.validation_steps == 0):
                score = -1  # self._validate()
                val_str = f" || Current validation score {score}."

            self._scheduler.step()

    @staticmethod
    def _model_forward(
        predictor: SAM2ImagePredictor,
        points,
        labels,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(points, labels),
            boxes=None,
            masks=None,
        )

        low_res_masks, prediction_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor.get_image_embedding(),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=predictor.get_high_resolution_features(),
        )
        pred_masks = predictor.mask_postprocessing(low_res_masks)

        return pred_masks, prediction_scores

    def _training_step(
        self,
        image: torch.Tensor,
        masks: list[torch.Tensor],
        point_coords: list[torch.Tensor],
        point_labels: list[torch.Tensor],
    ) -> float:
        """Compute a training step for the given batch.

        For details see Table 12 in https://arxiv.org/pdf/2408.00714.


        Args:
            image: the image for which to do the step.
            masks: list containing the masks for the image.
            point_coords: list containing the points for the image.
            point_labels: labels for each point, whether it is foreground or background.

        Returns:
            Losses for the given sample.

        """
        image_set = False
        iou_losses = []
        focal_losses = []

        for mask, points, labels in zip(masks, point_coords, point_labels, strict=True):
            with torch.amp.autocast(
                device_type="cuda",
                enabled=self._config.amp_enabled,
            ):
                mask_ = mask.to(self._device)

                if not image_set:
                    self._predictor.set_image(image.to(self._device))
                    image_set = True

                pred_masks, prediction_scores = self._model_forward(
                    self._predictor,
                    points.to(self._device),
                    labels.to(self._device),
                )

                iou_loss_: torch.Tensor = iou_loss(
                    pred_masks,
                    mask_[None],
                    prediction_scores,
                    1,
                )
                focal_loss: torch.Tensor = sigmoid_focal_loss(
                    pred_masks,
                    mask_[None],
                    1,
                    alpha=0.1,
                )

            iou_losses.append(iou_loss_)
            focal_losses.append(focal_loss)

        return iou_losses, focal_losses

    def _validation_step(
        self,
        image: torch.Tensor,
        masks: list[torch.Tensor],
        point_coords: list[torch.Tensor],
        point_labels: list[torch.Tensor],
    ) -> float:
        """Execute a validation step.

        Args:
            image: the image for which to do the step.
            masks: list containing the masks for the image.
            point_coords: list containing the points for the image.
            point_labels: labels for each point, whether it is foreground or background.

        Returns:
            Value associated with given metrics in this function. Here it is the
            validation accuracy (IoU).

        """
        image_set = False
        scores = []
        for mask, points, labels in zip(masks, point_coords, point_labels, strict=True):
            with torch.no_grad():
                mask_ = mask.to(self._device)

                if not image_set:
                    self._predictor.set_image(image.to(self._device))
                    image_set = True

                pred_masks, prediction_scores = self._model_forward(
                    self._predictor,
                    points.to(self._device),
                    labels.to(self._device),
                )

                iou_score: torch.Tensor = 1 - iou_loss(
                    pred_masks,
                    mask_[None],
                    prediction_scores,
                    1,
                )

                scores.append(iou_score.item())

        return np.mean(scores)

    def _validate(self) -> float:
        scores: list = []

        self._predictor.model.eval()
        for batch in tqdm(self._v_dataloader, desc="Running validation."):
            batch_size = len(batch["masks"])
            for batch_idx in range(batch_size):
                if not len(batch["masks"][batch_idx]):
                    continue  # batch with no masks

                masks = batch["masks"][batch_idx]
                point_coords = batch["point_coords"][batch_idx]
                point_labels = batch["point_labels"][batch_idx]
                image = batch["images"][batch_idx]

                score = self._validation_step(
                    image=image,
                    masks=masks,
                    point_coords=point_coords,
                    point_labels=point_labels,
                )
                scores.append(score)

        # restore regime as used during trainin
        self._predictor.model.image_encoder.eval()
        self._predictor.model.sam_mask_decoder.train()
        self._predictor.model.sam_prompt_encoder.train()

        if self._wandb:
            wandb.log({"Validation/IoU Score": np.mean(scores)})

        return np.mean(scores)

    def _save_checkpoint(self, step: int) -> None:
        checkpoint = {
            "step": step,
            "model_state_dict": self._predictor.model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scaler_state_dict": self._grad_scaler.state_dict(),
            "scheduler_state_dict": self._scheduler.state_dict(),
            "config": self._config,
        }

        checkpoint_path = Path(self._config.checkpoint_dir) / f"sam2_ft_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self._config.checkpoint_dir)
        self._predictor.model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._grad_scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetuning SAM2 for spine segmentation.",
    )

    parser.add_argument(
        "--training_dir",
        type=str,
        required=True,
        help="Directory containing the CTSpine1K training dataset",
    )
    parser.add_argument(
        "--validation_dir",
        type=str,
        required=True,
        help="Directory containing the CTSpine1K validation dataset",
    )
    parser.add_argument(
        "--with_wandb",
        type=bool,
        required=False,
        default=False,
        help="Whether we want to use wandb to log the data.",
    )
    parser.add_argument(
        "--from_checkpoint",
        type=bool,
        required=False,
        default=False,
        help="Whether the model should be initialised from a checkpoint.",
    )
    parser.add_argument(
        "--gin_config",
        type=str,
        required=False,
        default=None,
        help="Configurable parameters which can be parsed by gin.",
    )
    args = parser.parse_args()

    if args.gin_config:
        nnx.parse_gin_config(args.gin_config)

    config = SAM2FinetuningConfig()

    training_dir = Path(args.training_dir)
    validation_dir = Path(args.validation_dir)

    t_dataset = SAMAdapter(
        dataset=CTSpine1K(cache_dir=training_dir, loading_mode=LoadingMode.PRELOAD_RAM),
        image_size=config.image_size,
    )
    v_dataset = SAMAdapter(
        dataset=CTSpine1K(
            cache_dir=validation_dir,
            loading_mode=LoadingMode.PRELOAD_RAM,
        ),
        image_size=config.image_size,
    )

    t_dataloader = CUDALoader(
        dataset=t_dataset,
        batch_size=config.batch_size,
        collate_fn=t_dataset.collate_fn,
        drop_last=True,
    )

    v_dataloader = CUDALoader(
        dataset=v_dataset,
        batch_size=config.validation_size,
        collate_fn=v_dataset.collate_fn,
        drop_last=True,
    )

    sam2_finetuning = SAM2Finetuning(
        config=config,
        t_dataloader=t_dataloader,
        v_dataloader=v_dataloader,
        with_wandb=False,  # args.with_wandb,
        from_checkpoint=args.from_checkpoint,
    )

    sam2_finetuning.tune()
