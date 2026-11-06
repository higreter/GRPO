"""Frozen-reference placement for a single-process, two-device GRPO run.

TRL's :class:`GRPOTrainer` supports a frozen full-model reference when ``beta`` is
non-zero, but its standard single-process setup places that model on the accelerator
device.  This adapter keeps the trainable policy on logical ``cuda:0`` and moves the
frozen reference to an explicitly selected logical ``cuda:1``.  It is deliberately
*not* data parallelism: Trainer is forced to one policy GPU and no gradient can cross
to the reference.

The adapter is inactive unless both ``GRPO_POLICY_DEVICE`` and
``GRPO_REFERENCE_DEVICE`` are set.  Launches should expose exactly the desired physical
cards through ``CUDA_VISIBLE_DEVICES`` and then use ``cuda:0`` / ``cuda:1`` here.

Reference forwards are chunked independently with
``GRPO_REFERENCE_LOGPS_BATCH_SIZE`` (default 1).  This changes only the memory schedule
of an inference-only calculation; it leaves every token's reference log-probability and
the configured KL objective unchanged.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Optional


class DualDeviceReferenceError(RuntimeError):
    """The requested policy/reference topology cannot be safely constructed."""


@dataclass(frozen=True)
class DualDeviceReferenceTopology:
    """Logical-device layout and reference-forward chunk size for one run."""

    policy_device: str
    reference_device: str
    reference_logps_batch_size: int

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = "single-process policy + frozen external reference"
        payload["data_parallel"] = False
        return payload


def topology_from_environ(environ: Optional[dict[str, str]] = None) -> Optional[DualDeviceReferenceTopology]:
    """Parse explicit dual-device settings without importing torch.

    Keeping this parser torch-free lets CPU-only planning and unit tests remain so.  The
    actual CUDA and model assertions happen immediately before the trainer is built.
    """
    env = os.environ if environ is None else environ
    policy = env.get("GRPO_POLICY_DEVICE", "").strip()
    reference = env.get("GRPO_REFERENCE_DEVICE", "").strip()
    chunk = env.get("GRPO_REFERENCE_LOGPS_BATCH_SIZE", "1").strip()
    if not policy and not reference:
        return None
    if not policy or not reference:
        raise DualDeviceReferenceError(
            "set both GRPO_POLICY_DEVICE and GRPO_REFERENCE_DEVICE, or neither"
        )
    if policy != "cuda:0":
        raise DualDeviceReferenceError(
            "GRPO_POLICY_DEVICE must be logical cuda:0; select the physical policy GPU "
            "through CUDA_VISIBLE_DEVICES"
        )
    if reference == policy:
        raise DualDeviceReferenceError(
            "policy and reference must be distinct logical CUDA devices"
        )
    if not reference.startswith("cuda:"):
        raise DualDeviceReferenceError(
            "GRPO_REFERENCE_DEVICE must be a CUDA device such as cuda:1"
        )
    try:
        reference_index = int(reference.split(":", 1)[1])
        chunk_size = int(chunk)
    except ValueError as error:
        raise DualDeviceReferenceError(
            "GRPO_REFERENCE_DEVICE and GRPO_REFERENCE_LOGPS_BATCH_SIZE must use integer indices"
        ) from error
    if reference_index < 1:
        raise DualDeviceReferenceError(
            "GRPO_REFERENCE_DEVICE must be a non-policy logical GPU (cuda:1 or later)"
        )
    if chunk_size < 1:
        raise DualDeviceReferenceError("GRPO_REFERENCE_LOGPS_BATCH_SIZE must be >= 1")
    return DualDeviceReferenceTopology(
        policy_device=policy,
        reference_device=reference,
        reference_logps_batch_size=chunk_size,
    )


class DualDeviceReferenceMixin(object):
    """Move TRL's frozen reference model and its inputs to a separate CUDA device."""

    def __init__(self, *args: Any, dual_device_topology: Optional[DualDeviceReferenceTopology] = None,
                 **kwargs: Any) -> None:
        self._dual_device_topology = dual_device_topology
        if dual_device_topology is not None:
            # Transformers.Trainer otherwise sees both visible devices and silently
            # chooses DataParallel.  The policy remains a single-process, single-GPU
            # model; only the explicitly managed, frozen reference uses cuda:1.
            training_args = kwargs.get("args")
            if training_args is None:
                raise DualDeviceReferenceError("GRPO trainer arguments were not provided")
            training_args._n_gpu = 1
        super().__init__(*args, **kwargs)
        if dual_device_topology is not None:
            self._place_frozen_reference(dual_device_topology)

    def _place_frozen_reference(self, topology: DualDeviceReferenceTopology) -> None:
        import torch

        if not torch.cuda.is_available():
            raise DualDeviceReferenceError("CUDA is unavailable for the requested dual-device run")
        if torch.cuda.device_count() <= int(topology.reference_device.split(":", 1)[1]):
            raise DualDeviceReferenceError(
                "requested {0}, but only {1} logical CUDA device(s) are visible".format(
                    topology.reference_device, torch.cuda.device_count())
            )
        if self.beta == 0.0 or self.ref_model is None:
            raise DualDeviceReferenceError(
                "a separate reference was requested but TRL did not construct one; "
                "check beta, PEFT mode, and reference policy"
            )
        if isinstance(self.model, torch.nn.DataParallel):
            raise DualDeviceReferenceError("DataParallel is forbidden for dual-device reference runs")

        policy_device = torch.device(topology.policy_device)
        reference_device = torch.device(topology.reference_device)
        model_device = next(self.model.parameters()).device
        if model_device != policy_device:
            raise DualDeviceReferenceError(
                "policy model is on {0}, expected {1}".format(model_device, policy_device)
            )

        self.ref_model.to(reference_device)
        self.ref_model.eval()
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)
        reference_parameter_count = sum(1 for _ in self.ref_model.parameters())
        if any(parameter.requires_grad for parameter in self.ref_model.parameters()):
            raise DualDeviceReferenceError("reference parameters are not fully frozen")
        reference_model_device = next(self.ref_model.parameters()).device
        if reference_model_device != reference_device:
            raise DualDeviceReferenceError(
                "reference model is on {0}, expected {1}".format(
                    reference_model_device, reference_device)
            )
        self.dual_device_reference_status = {
            **topology.manifest(),
            "policy_model_device": str(model_device),
            "reference_model_device": str(reference_model_device),
            "reference_parameters": reference_parameter_count,
            "reference_all_frozen": True,
        }

    def create_optimizer(self):
        """Assert that no frozen-reference parameter reaches the policy optimizer."""
        result = super().create_optimizer()
        topology = getattr(self, "_dual_device_topology", None)
        if topology is not None:
            reference_ids = {id(parameter) for parameter in self.ref_model.parameters()}
            optimizer_ids = {
                id(parameter)
                for group in self.optimizer.param_groups
                for parameter in group["params"]
            }
            overlap = reference_ids.intersection(optimizer_ids)
            if overlap:
                raise DualDeviceReferenceError(
                    "the policy optimizer contains {0} frozen reference parameter(s)".format(len(overlap))
                )
            self.dual_device_reference_status["optimizer_reference_parameter_overlap"] = 0
        return result

    @staticmethod
    def _move_reference_tensor(value: Any, device: Any) -> Any:
        """Move tensor inputs only; counters/lists used by VLM paths stay Python objects."""
        try:
            import torch
        except ImportError:  # pragma: no cover - only reached in a malformed training environment
            return value
        return value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value

    def _get_per_token_logps_and_entropies(
        self,
        model: Any,
        input_ids: Any,
        attention_mask: Any,
        logits_to_keep: int,
        batch_size: Optional[int] = None,
        compute_entropy: bool = False,
        compute_aux_loss: bool = False,
        pixel_values: Any = None,
        image_grid_thw: Any = None,
        num_images: Any = None,
        pixel_attention_mask: Any = None,
        spatial_shapes: Any = None,
        num_tiles: Any = None,
        image_sizes: Any = None,
        token_type_ids: Any = None,
        mm_token_type_ids: Any = None,
        image_position_ids: Any = None,
        sampling_mask: Any = None,
    ):
        """Run only the frozen-reference forward on its device, then return logps to policy.

        TRL calls this method under ``torch.no_grad`` during generation/scoring.  The
        returned log-probabilities are therefore detached values, exactly as in TRL's
        single-device reference path; moving them back cannot create a reference gradient.
        """
        topology = getattr(self, "_dual_device_topology", None)
        if topology is None or model is not getattr(self, "ref_model", None):
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep,
                batch_size=batch_size, compute_entropy=compute_entropy,
                compute_aux_loss=compute_aux_loss, pixel_values=pixel_values,
                image_grid_thw=image_grid_thw, num_images=num_images,
                pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes,
                num_tiles=num_tiles, image_sizes=image_sizes, token_type_ids=token_type_ids,
                mm_token_type_ids=mm_token_type_ids, image_position_ids=image_position_ids,
                sampling_mask=sampling_mask,
            )

        import torch

        source_device = input_ids.device
        reference_device = torch.device(topology.reference_device)
        reference_batch_size = min(
            int(batch_size or input_ids.size(0)), topology.reference_logps_batch_size
        )
        with torch.no_grad():
            logps, entropies, aux_loss = super()._get_per_token_logps_and_entropies(
                model,
                self._move_reference_tensor(input_ids, reference_device),
                self._move_reference_tensor(attention_mask, reference_device),
                logits_to_keep,
                batch_size=reference_batch_size,
                compute_entropy=compute_entropy,
                compute_aux_loss=compute_aux_loss,
                pixel_values=self._move_reference_tensor(pixel_values, reference_device),
                image_grid_thw=self._move_reference_tensor(image_grid_thw, reference_device),
                num_images=num_images,
                pixel_attention_mask=self._move_reference_tensor(pixel_attention_mask, reference_device),
                spatial_shapes=self._move_reference_tensor(spatial_shapes, reference_device),
                num_tiles=num_tiles,
                image_sizes=self._move_reference_tensor(image_sizes, reference_device),
                token_type_ids=self._move_reference_tensor(token_type_ids, reference_device),
                mm_token_type_ids=self._move_reference_tensor(mm_token_type_ids, reference_device),
                image_position_ids=self._move_reference_tensor(image_position_ids, reference_device),
                sampling_mask=self._move_reference_tensor(sampling_mask, reference_device),
            )
        def back(value: Any) -> Any:
            return value.to(source_device, non_blocking=True) if value is not None else None
        return back(logps), back(entropies), back(aux_loss)


def wrap_trainer_class(base: type) -> type:
    """Return a named class preserving method-specific mixins underneath the topology mixin."""
    class DualDeviceReferenceTrainer(DualDeviceReferenceMixin, base):
        pass

    DualDeviceReferenceTrainer.__name__ = "DualDeviceReference{0}".format(base.__name__)
    return DualDeviceReferenceTrainer


__all__ = [
    "DualDeviceReferenceError",
    "DualDeviceReferenceMixin",
    "DualDeviceReferenceTopology",
    "topology_from_environ",
    "wrap_trainer_class",
]
