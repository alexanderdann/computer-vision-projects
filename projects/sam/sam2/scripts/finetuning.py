"""Finetuning script for SAM2."""  # noqa: INP001

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from training.loss_fns import iou_loss, sigmoid_focal_loss

import nnx.data
from nnx.data.ctspine1k.dataset import CTSpine1K, SAMAdapter


@nnx.configurable
class SAM2FinetuningConfig:
    """Configurable finetuning unit for SAM2."""

    # Model configuration
    checkpoint_dir: str = "/home/ubuntu/data/checkpoints"
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
    scheduler_type: str = "cosine"
    min_lr: float = 1e-8
    start_factor: float = 0.1
    end_factor: float = 0.8
    warmup_epochs: int = 2

    # Dataset configuration
    batch_size: int = 64
    validation_size: int = 64
    num_workers: int = 10
    prefetch_factor: int = 25
    shuffle: bool = True

    # Checkpoint and evaluation
    save_checkpoint: bool = True
    validation_steps: int = 1
    log_steps: int = 20


class SAM2Finetuning:
    """Finetuning class to tune the parameters of SAM2."""

    def __init__(
        self,
        config: SAM2FinetuningConfig,
        t_dataloader: DataLoader,
        v_dataloader: DataLoader,
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

        if self._config.warmup_epochs >= self._config.epochs:
            msg = "Invalid combination of warmup and total epochs."
            raise ValueError(msg)

        warmup_scheduler = LinearLR(
            self._optimizer,
            start_factor=self._config.start_factor,
            end_factor=self._config.end_factor,
            total_iters=self._config.warmup_epochs,
        )

        cosine_scheduler = CosineAnnealingLR(
            self._optimizer,
            T_max=self._config.epochs - self._config.warmup_epochs,
            eta_min=self._config.min_lr,
        )

        self._scheduler = SequentialLR(
            self._optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self._config.warmup_epochs],
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

    def tune(self) -> None:
        """Tune the SAM2 model with the given data."""
        for epoch in range(self._config.epochs):
            data_iter = tqdm(self._t_dataloader, desc="Starting finetuning...")
            val_str = ""

            if epoch and self._config.save_checkpoint:
                self._save_checkpoint(epoch)

            if epoch and (epoch % self._config.validation_steps):
                score = self._validate()
                val_str = f" || Current validation score {score}."

            for batch in data_iter:
                for batch_idx in range(self._config.batch_size):
                    if not len(batch["masks"][batch_idx]):
                        data_iter.set_description_str(
                            "Finetuning. Skipped batch with no masks.",
                        )
                        continue  # batch with no masks

                    masks = self.to_device(batch["masks"][batch_idx], self._device)
                    point_coords = self.to_device(
                        batch["point_coords"][batch_idx],
                        self._device,
                    )
                    point_labels = self.to_device(
                        batch["point_labels"][batch_idx],
                        self._device,
                    )
                    image = self.to_device(batch["images"][batch_idx], self._device)

                    loss = self._training_step(
                        image=image,
                        masks=masks,
                        point_coords=point_coords,
                        point_labels=point_labels,
                    )

                    desc_str = f"Finetuning. Current loss: {loss}" + val_str

                    data_iter.set_description_str(desc_str)

                self._grad_scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._predictor.model.parameters(),
                    max_norm=self._config.clip_norm,
                )
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()

                self._predictor.model.zero_grad()

            self._scheduler.step()

    @classmethod
    def to_device(cls, data, device):  # noqa: ANN206
        """Recursively move data to device."""  # noqa: DOC201
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, (list, tuple)):
            return [cls.to_device(item, device) for item in data]
        if isinstance(data, dict):
            return {k: cls.to_device(v, device) for k, v in data.items()}
        if isinstance(data, torch.nested.NestedTensor):
            return data.to(device)
        # If it's not a tensor or container of tensors, return as is
        return data

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
            Average loss over the masks.

        """
        image_set = False
        losses = []
        for idx, [mask, points, labels] in enumerate(
            zip(masks, point_coords, point_labels, strict=True),
        ):
            with torch.amp.autocast(
                device_type="cuda",
                enabled=self._config.amp_enabled,
            ):
                if not image_set:
                    self._predictor.set_image(image)
                    image_set = True

                pred_masks, prediction_scores = self._model_forward(
                    self._predictor,
                    points,
                    labels,
                )

                iou_loss_: torch.Tensor = iou_loss(
                    pred_masks,
                    mask[None],
                    prediction_scores,
                    1,
                )
                focal_loss: torch.Tensor = sigmoid_focal_loss(
                    pred_masks,
                    mask[None],
                    1,
                    alpha=0.1,
                )

            loss: torch.Tensor = self._grad_scaler.scale(iou_loss_ + focal_loss)
            loss.backward()

            losses.append(loss.item())

            if self._wandb and (idx % self._config.log_steps == 0):
                wandb.log(
                    {
                        "Loss/IoU Loss": iou_loss_.item(),
                        "Loss/Focal Loss": focal_loss.item(),
                        "Loss/Total Loss": loss.item(),
                    },
                )

        return np.mean(losses)

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
        for _, [mask, points, labels] in enumerate(
            zip(masks, point_coords, point_labels, strict=True),
        ):
            with torch.no_grad():
                if not image_set:
                    self._predictor.set_image(image)
                    image_set = True

                pred_masks, prediction_scores = self._model_forward(
                    self._predictor,
                    points,
                    labels,
                )

                iou_score: torch.Tensor = 1 - iou_loss(
                    pred_masks,
                    mask[None],
                    prediction_scores,
                    1,
                )

                if self._wandb:
                    wandb.log({"Validation/IoU Score": iou_score.item()})

                scores.append(iou_score.item())

        return np.mean(scores)

    def _validate(self) -> float:
        scores: list = []

        self._predictor.model.eval()
        for batch in self._v_dataloader:
            for batch_idx in range(self._config.batch_size):
                if not len(batch["masks"][batch_idx]):
                    continue  # batch with no masks

                masks = self.to_device(batch["masks"][batch_idx], self._device)
                point_coords = self.to_device(
                    batch["point_coords"][batch_idx],
                    self._device,
                )
                point_labels = self.to_device(
                    batch["point_labels"][batch_idx],
                    self._device,
                )
                image = self.to_device(batch["images"][batch_idx], self._device)

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

        return np.mean(scores)

    def _save_checkpoint(self, step: int) -> None:
        checkpoint = {
            "step": step,
            "model_state_dict": self._predictor.model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scaler_state_dict": self._grad_scaler.state_dict(),
            "config": self._config,
        }

        checkpoint_path = Path(self._config.checkpoint_dir) / f"sam2_ft_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self._config.checkpoint_dir)
        self._predictor.model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._grad_scaler.load_state_dict(checkpoint["scaler_state_dict"])


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
    args = parser.parse_args()

    config = SAM2FinetuningConfig()

    training_dir = Path(args.training_dir)
    validation_dir = Path(args.validation_dir)

    t_dataset = SAMAdapter(
        dataset=CTSpine1K(cache_dir=training_dir),
        image_size=config.image_size,
    )
    v_dataset = SAMAdapter(
        dataset=CTSpine1K(cache_dir=validation_dir),
        image_size=config.image_size,
    )

    t_dataloader = DataLoader(
        t_dataset,
        batch_size=config.batch_size,
        collate_fn=t_dataset.collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,  # should always be True unless there is a good reason
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=config.num_workers,
        shuffle=config.shuffle,
    )

    v_dataloader = DataLoader(
        t_dataset,
        batch_size=config.validation_size,
        collate_fn=t_dataset.collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,  # should always be True unless there is a good reason
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=config.num_workers,
        shuffle=config.shuffle,
    )

    sam2_finetuning = SAM2Finetuning(
        config=config,
        t_dataloader=t_dataloader,
        v_dataloader=v_dataloader,
        with_wandb=args.with_wandb,
        from_checkpoint=args.from_checkpoint,
    )

    sam2_finetuning.tune()
