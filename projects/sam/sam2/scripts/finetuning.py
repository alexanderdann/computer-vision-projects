"""Finetuning script for SAM2."""  # noqa: INP001

from pathlib import Path

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from training.loss_fns import iou_loss, sigmoid_focal_loss

import nnx.data
from nnx.data.ctspine1k.dataset import CTSpine1K, SAMAdapter

nnx.data.rng = np.random.default_rng(seed=1)

SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_large.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
BATCH_SIZE = 3


def to_device(data, device):  # noqa: ANN201
    """Recursively move data to device."""  # noqa: DOC201
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, (list, tuple)):
        return [to_device(item, device) for item in data]
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, torch.nested.NestedTensor):
        return data.to(device)
    # If it's not a tensor or container of tensors, return as is
    return data


def training_step(  # noqa: PLR0913, PLR0917
    predictor: SAM2ImagePredictor,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler,
    image: torch.Tensor,
    masks: list[torch.Tensor],
    point_coords: list[torch.Tensor],
    point_labels: list[torch.Tensor],
) -> float:
    """Compute a training step for the given batch.

    For details see Table 12 in https://arxiv.org/pdf/2408.00714.

    Returns:
        Average loss over the masks.

    """
    image_set = False
    losses = []
    for _, [mask, points, labels] in enumerate(
        zip(masks, point_coords, point_labels, strict=True),
    ):
        with torch.amp.autocast(device_type="cuda"):
            if not image_set:
                predictor.set_image(image)
                image_set = True

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
            )

            loss: torch.Tensor = grad_scaler.scale(iou_loss_ + focal_loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1)

            losses.append(loss.item())

            grad_scaler.step(optimizer)
            grad_scaler.update()
            predictor.model.zero_grad()

    return np.mean(losses)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cache_dir = Path("/home/ubuntu/data/CTSpine1K")
    files = [file.parent for file in cache_dir.rglob("*") if file.suffix == ".gz"]

    sam2_model = build_sam2(MODEL_CONFIG, SAM2_CHECKPOINT, device)

    predictor = SAM2ImagePredictor(sam2_model)

    ct_dataset = CTSpine1K(cache_dir=cache_dir)
    dataset = SAMAdapter(dataset=ct_dataset, image_size=predictor.model.image_size)

    predictor.model.sam_mask_decoder.train(mode=True)
    predictor.model.sam_prompt_encoder.train(mode=True)

    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(),
        lr=1e-6,
        weight_decay=1e-3,
    )

    grad_scaler = torch.amp.GradScaler("cuda")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=dataset.collate_fn,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=10,
    )

    data_iter = tqdm(dataloader, desc="Starting finetuning...")
    for sample in data_iter:
        for batch_idx in range(BATCH_SIZE):
            if not len(sample["masks"][batch_idx]):
                data_iter.set_description_str(
                    "Finetuning. Skipped batch with no masks.",
                )
                continue  # batch with no masks

            masks = to_device(sample["masks"][batch_idx], device)
            point_coords = to_device(sample["point_coords"][batch_idx], device)
            point_labels = to_device(sample["point_labels"][batch_idx], device)
            image = to_device(sample["images"][batch_idx], device)

            loss = training_step(
                predictor=predictor,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                image=image,
                masks=masks,
                point_coords=point_coords,
                point_labels=point_labels,
            )

            data_iter.set_description_str(f"Finetuning. Current loss: {loss}")
