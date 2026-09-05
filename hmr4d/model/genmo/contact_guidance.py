"""Differentiable contact guidance for GENMO DDIM sampling.

The callback in this module operates on the predicted normalized x0 motion.  It
does not update model weights and it never edits a completed SMPL-X sequence.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch

from hmr4d.utils.body_model.utils import (
    SMPLH_JOINT_NAMES,
    SMPLH_LEFT_ARM,
    SMPLH_RIGHT_ARM,
)


MOTION_DIM = 151
BODY_POSE_SLICE = slice(0, 126)
ROOT_VELOCITY_SLICE = slice(148, 151)
WRIST_JOINT_INDEX = {"left": 20, "right": 21}

# SMPL-X wrist-local offsets, in metres.  The signs follow the neutral SMPL-X
# T-pose, whose left/right hands extend along positive/negative local X.
WRIST_TO_PALM_OFFSET_M = {
    "left": (0.08, 0.0, 0.0),
    "right": (-0.08, 0.0, 0.0),
}


def _arm_joint_names(hand: str) -> Iterable[str]:
    hand = str(hand).lower()
    if hand == "left":
        return SMPLH_LEFT_ARM
    if hand == "right":
        return SMPLH_RIGHT_ARM
    raise ValueError(f"selected hand must be left or right, got {hand!r}")


def arm_channel_slices(hand: str) -> Dict[str, slice]:
    """Derive the four arm-joint 6D ranges from the active SMPL-X ordering."""
    body_names = SMPLH_JOINT_NAMES[1:22]
    ranges = {}
    for name in _arm_joint_names(hand):
        joint_index = body_names.index(name)
        ranges[name] = slice(joint_index * 6, (joint_index + 1) * 6)
    return ranges


def allowed_channel_mask(
    hand: str,
    *,
    include_root: bool,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return the verified 151-D arm-only or arm+root channel mask."""
    mask = torch.zeros(MOTION_DIM, device=device, dtype=dtype)
    for channel_slice in arm_channel_slices(hand).values():
        mask[channel_slice] = 1
    if include_root:
        mask[ROOT_VELOCITY_SLICE] = 1
    return mask


def generate_sampling_noise(shape, *, device, seed, dtype=torch.float32) -> torch.Tensor:
    """Create explicit sampling noise without mutating PyTorch's global RNG."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def should_use_root_fallback(
    contact_error_m: float,
    max_arm_rotation_change_deg: float,
    *,
    reach_error_threshold: float,
    max_arm_rotation_change_limit_deg: float,
) -> bool:
    """Fallback is needed unless arm-only satisfies both acceptance gates."""
    return not (
        contact_error_m <= reach_error_threshold
        and max_arm_rotation_change_deg <= max_arm_rotation_change_limit_deg
    )


def root_candidate_improves_hold(
    arm_contact_error_m: float,
    arm_post_contact_mean_error_m: float,
    arm_post_contact_max_error_m: float,
    root_contact_error_m: float,
    root_post_contact_mean_error_m: float,
    root_post_contact_max_error_m: float,
) -> bool:
    """Prefer root fallback only when its strict post-contact hold is better."""
    arm_score = (
        float(arm_post_contact_max_error_m),
        float(arm_post_contact_mean_error_m),
        float(arm_contact_error_m),
    )
    root_score = (
        float(root_post_contact_max_error_m),
        float(root_post_contact_mean_error_m),
        float(root_contact_error_m),
    )
    if not all(torch.isfinite(torch.tensor(arm_score + root_score))):
        raise ValueError("Guidance candidate scores contain NaN/Inf")
    return root_score < arm_score


@dataclass(frozen=True)
class ContactGuidanceConfig:
    contact_frame: int
    selected_hand: str
    object_surface_target: torch.Tensor
    reference_motion: torch.Tensor
    object_surface_targets: Optional[torch.Tensor] = None
    # Explicit two-segment position targets.  The first is static before
    # contact; the second starts at contact and follows the frozen object.
    pre_contact_surface_target: Optional[torch.Tensor] = None
    post_contact_surface_targets: Optional[torch.Tensor] = None
    include_root: bool = False
    guidance_strength: float = 0.7
    contact_window_radius: int = 2
    contact_transition_frames: int = 8
    w_contact: float = 4.0
    w_reference: float = 0.05
    w_temporal: float = 0.10
    post_contact_relative_velocity_weight: float = 8.0
    post_contact_worst_frame_weight: float = 0.0
    post_contact_worst_frame_fraction: float = 0.125
    post_contact_terminal_position_weight: float = 1.0
    post_contact_terminal_frames: int = 16
    contact_frame_position_weight: float = 2.0
    contact_frame_weight_radius: int = 2
    grad_clip_norm: float = 1.0
    root_guidance_multiplier: float = 1.0
    diffusion_steps: int = 50
    human_depth_scale: float = 1.0
    camera_origin_global: Optional[torch.Tensor] = None


class ContactGuidance:
    """A ``denoised_fn`` callback for ``ddim_sample_loop_with_aux``."""

    def __init__(
        self,
        config: ContactGuidanceConfig,
        palm_position_fn: Callable[[torch.Tensor, str], torch.Tensor],
    ):
        self.config = config
        self.palm_position_fn = palm_position_fn
        self.step_diagnostics = []

    def _time_scale(self, timestep: torch.Tensor, *, root: bool) -> torch.Tensor:
        # DDIM visits large t first. Arm guidance remains active throughout;
        # root guidance fades to zero over the final 35% of denoising.
        t = timestep.float() / max(float(self.config.diffusion_steps - 1), 1.0)
        if root:
            t = torch.clamp(t / 0.35, min=0.0, max=1.0)
        else:
            t = torch.ones_like(t)
        return t.reshape(-1, 1, 1)

    def __call__(self, x0_pred: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = x0_pred.dtype
        with torch.enable_grad():
            base_x0 = x0_pred.detach().float()
            reference = self.config.reference_motion.to(base_x0).detach()
            target = self.config.object_surface_target.to(base_x0).reshape(1, 1, 3)
            mask = allowed_channel_mask(
                self.config.selected_hand,
                include_root=self.config.include_root,
                device=base_x0.device,
                dtype=base_x0.dtype,
            )
            active_indices = torch.nonzero(mask, as_tuple=False).flatten()
            active_x0 = (
                base_x0.index_select(-1, active_indices).clone().requires_grad_(True)
            )
            x0 = torch.index_copy(base_x0, -1, active_indices, active_x0)
            mask = mask.reshape(1, 1, -1)

            palm = self.palm_position_fn(x0, self.config.selected_hand)
            if palm.ndim != 3 or palm.shape[-1] != 3:
                raise ValueError(f"palm_position_fn must return [B,T,3], got {palm.shape}")
            depth_scale = float(self.config.human_depth_scale)
            if not torch.isfinite(torch.tensor(depth_scale)) or depth_scale <= 0.0:
                raise ValueError(f"human_depth_scale must be finite and positive, got {depth_scale}")
            if self.config.camera_origin_global is not None:
                camera_origin = self.config.camera_origin_global.to(base_x0).reshape(1, 1, 3)
                palm = camera_origin + depth_scale * (palm - camera_origin)
            elif depth_scale != 1.0:
                raise ValueError("camera_origin_global is required when human_depth_scale != 1")
            frame = int(self.config.contact_frame)
            start = max(0, frame - int(self.config.contact_window_radius))
            end = min(palm.shape[1], frame + int(self.config.contact_window_radius) + 1)
            if not 0 <= frame < palm.shape[1] or start >= end:
                raise ValueError(f"contact frame {frame} is outside a {palm.shape[1]}-frame motion")
            # Once contact begins, constrain every subsequent frame to the
            # corresponding point on the frozen object trajectory.  A short
            # smooth ramp avoids an impulse at the onset while preserving the
            # original local-window behavior for callers without a trajectory.
            trajectory = self.config.object_surface_targets
            explicit_pre_target = self.config.pre_contact_surface_target
            explicit_post_targets = self.config.post_contact_surface_targets
            contact_weight = float(self.config.contact_frame_position_weight)
            relative_velocity_loss = base_x0.new_zeros(())
            pre_contact_position_loss = base_x0.new_zeros(())
            post_contact_position_loss = base_x0.new_zeros(())
            if trajectory is not None or explicit_pre_target is not None or explicit_post_targets is not None:
                # Prefer the explicit two-segment representation when
                # supplied, while retaining the legacy full trajectory API.
                pre_target = explicit_pre_target
                post_targets = explicit_post_targets
                if pre_target is not None or post_targets is not None:
                    if pre_target is None or post_targets is None:
                        raise ValueError(
                            "pre_contact_surface_target and "
                            "post_contact_surface_targets must be supplied together"
                        )
                    pre_target = pre_target.to(base_x0).reshape(1, 1, 3)
                    post_targets = post_targets.to(base_x0)
                    expected_post_shape = (palm.shape[1] - frame, 3)
                    if tuple(post_targets.shape) != expected_post_shape:
                        raise ValueError(
                            "post_contact_surface_targets must have shape "
                            f"{expected_post_shape}, got {tuple(post_targets.shape)}"
                        )
                    # Keep the contact frame in the post-contact segment; the
                    # pre-contact target remains the static point up to frame-1.
                    target_seq = torch.cat(
                        [
                            pre_target.expand(1, max(frame, 0), 3),
                            post_targets.reshape(1, -1, 3),
                        ],
                        dim=1,
                    )
                else:
                    trajectory = trajectory.to(base_x0)
                    if trajectory.shape != (palm.shape[1], 3):
                        raise ValueError(
                            "object_surface_targets must have shape [T,3], "
                            f"got {tuple(trajectory.shape)} for T={palm.shape[1]}"
                        )
                    target_seq = trajectory.reshape(1, palm.shape[1], 3)
                transition = max(0, int(self.config.contact_transition_frames))
                weights = torch.zeros(palm.shape[1], device=base_x0.device, dtype=base_x0.dtype)
                if transition:
                    ramp_start = max(0, frame - transition)
                    ramp = torch.linspace(
                        0.0,
                        1.0,
                        frame - ramp_start + 1,
                        device=base_x0.device,
                        dtype=base_x0.dtype,
                    )
                    weights[ramp_start : frame + 1] = ramp
                else:
                    ramp_start = frame
                weights[frame:] = 1.0
                # Give the detected contact frame a modest, smooth emphasis.
                # The ramp avoids a one-frame impulse while ensuring the hand
                # reaches the target at the actual onset frame.
                contact_radius = max(0, int(self.config.contact_frame_weight_radius))
                if (
                    not torch.isfinite(torch.tensor(contact_weight))
                    or contact_weight < 1.0
                ):
                    raise ValueError(
                        "contact_frame_position_weight must be finite and >= 1"
                    )
                for offset in range(-contact_radius, contact_radius + 1):
                    index = frame + offset
                    if 0 <= index < palm.shape[1]:
                        factor = 1.0 + (contact_weight - 1.0) * (
                            1.0 - abs(offset) / max(contact_radius, 1)
                        )
                        weights[index] *= factor
                terminal_weight = float(
                    self.config.post_contact_terminal_position_weight
                )
                terminal_frames = min(
                    max(1, int(self.config.post_contact_terminal_frames)),
                    palm.shape[1] - frame,
                )
                if not torch.isfinite(torch.tensor(terminal_weight)) or terminal_weight < 1.0:
                    raise ValueError(
                        "post_contact_terminal_position_weight must be finite and >= 1"
                    )
                terminal_start = palm.shape[1] - terminal_frames
                weights[terminal_start:] *= torch.linspace(
                    1.0,
                    terminal_weight,
                    terminal_frames,
                    device=base_x0.device,
                    dtype=base_x0.dtype,
                )
                relative_position = palm - target_seq
                residual = relative_position.square().sum(dim=-1)
                position_loss = (
                    residual * weights.reshape(1, -1)
                ).sum() / weights.sum().clamp_min(1.0)
                if pre_target is not None and post_targets is not None:
                    pre_weights = weights[:frame]
                    post_weights = weights[frame:]
                    if pre_weights.numel() and pre_weights.sum() > 0:
                        pre_contact_position_loss = (
                            residual[:, :frame] * pre_weights.reshape(1, -1)
                        ).sum() / pre_weights.sum().clamp_min(1.0)
                    if post_weights.numel() and post_weights.sum() > 0:
                        post_contact_position_loss = (
                            residual[:, frame:] * post_weights.reshape(1, -1)
                        ).sum() / post_weights.sum().clamp_min(1.0)
                post_contact_residual = relative_position[:, frame:].square().sum(dim=-1)
                worst_fraction = float(self.config.post_contact_worst_frame_fraction)
                worst_weight = float(self.config.post_contact_worst_frame_weight)
                if not 0.0 < worst_fraction <= 1.0 or worst_weight < 0.0:
                    raise ValueError(
                        "post-contact worst-frame fraction/weight must be positive"
                    )
                worst_count = max(
                    1, int(torch.ceil(torch.tensor(post_contact_residual.shape[1] * worst_fraction)))
                )
                worst_position_loss = torch.topk(
                    post_contact_residual, k=worst_count, dim=1
                ).values.mean()
                if palm.shape[1] > frame + 1:
                    relative_velocity = (
                        relative_position[:, frame + 1 :] - relative_position[:, frame:-1]
                    )
                    relative_velocity_loss = relative_velocity.square().sum(dim=-1).mean()
                relative_velocity_weight = float(
                    self.config.post_contact_relative_velocity_weight
                )
                if not torch.isfinite(torch.tensor(relative_velocity_weight)) or relative_velocity_weight < 0:
                    raise ValueError(
                        "post_contact_relative_velocity_weight must be finite and non-negative"
                    )
                contact_loss = (
                    position_loss
                    + worst_weight * worst_position_loss
                    + relative_velocity_weight * relative_velocity_loss
                )
                # Keep the per-frame gradient comparable to the legacy
                # contact window when the persistent trajectory contains many
                # frames.  Without this compensation, averaging 50+ frames
                # weakens the contact signal by an order of magnitude.
                legacy_count = max(1, end - start)
                contact_loss = contact_loss * (
                    weights.sum() / float(legacy_count)
                ).clamp_min(1.0)
            else:
                contact_loss = ((palm[:, start:end] - target) ** 2).sum(dim=-1).mean()
                worst_position_loss = base_x0.new_zeros(())
            delta = (x0 - reference) * mask
            reference_loss = delta.square().sum() / mask.sum().clamp_min(1.0) / x0.shape[1]
            if delta.shape[1] > 1:
                temporal_delta = delta[:, 1:] - delta[:, :-1]
                temporal_loss = temporal_delta.square().mean()
            else:
                temporal_loss = delta.new_zeros(())
            total_loss = (
                self.config.w_contact * contact_loss
                + self.config.w_reference * reference_loss
                + self.config.w_temporal * temporal_loss
            )
            if not torch.isfinite(total_loss):
                values = {
                    "contact": float(contact_loss.detach().cpu()),
                    "reference": float(reference_loss.detach().cpu()),
                    "temporal": float(temporal_loss.detach().cpu()),
                    "post_contact_relative_velocity": float(
                        relative_velocity_loss.detach().cpu()
                    ),
                    "palm_finite": bool(torch.isfinite(palm).all()),
                    "target_finite": bool(torch.isfinite(target).all()),
                }
                raise FloatingPointError(
                    f"GENMO contact guidance loss contains NaN/Inf: {values}"
                )

            grad = torch.autograd.grad(total_loss, active_x0, retain_graph=False)[0]
            arm_in_active = active_indices < ROOT_VELOCITY_SLICE.start
            arm_grad_norm = torch.linalg.vector_norm(
                grad[..., arm_in_active].reshape(grad.shape[0], -1), dim=-1
            )
            root_grad_norm = torch.zeros_like(arm_grad_norm)
            if self.config.include_root:
                root_scale = self._time_scale(timestep, root=True)
                root_in_active = active_indices >= ROOT_VELOCITY_SLICE.start
                root_grad = grad[..., root_in_active]
                root_grad_norm = torch.linalg.vector_norm(
                    root_grad.reshape(grad.shape[0], -1), dim=-1
                )
                root_multiplier = float(self.config.root_guidance_multiplier)
                if not torch.isfinite(torch.tensor(root_multiplier)) or root_multiplier < 0.0:
                    raise ValueError(
                        "root_guidance_multiplier must be finite and non-negative, "
                        f"got {root_multiplier}"
                    )
                grad[..., root_in_active] = root_grad * root_scale * root_multiplier
            grad_norm = torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), dim=-1)
            clip = torch.clamp(
                float(self.config.grad_clip_norm) / grad_norm.clamp_min(1e-8), max=1.0
            )
            grad = grad * clip.reshape(-1, 1, 1)
            if not torch.isfinite(grad).all():
                raise FloatingPointError("GENMO contact guidance gradient contains NaN/Inf")

            scale = float(self.config.guidance_strength) * self._time_scale(
                timestep, root=False
            )
            guided_active = active_x0 - scale * grad
            guided = torch.index_copy(base_x0, -1, active_indices, guided_active)
            if not torch.isfinite(guided).all():
                raise FloatingPointError("GENMO guided x0 contains NaN/Inf")
            self.step_diagnostics.append(
                {
                    "timestep": int(timestep[0].detach().cpu()),
                    "contact_loss": float(contact_loss.detach().cpu()),
                    "reference_loss": float(reference_loss.detach().cpu()),
                    "temporal_loss": float(temporal_loss.detach().cpu()),
                    "post_contact_relative_velocity_loss": float(
                        relative_velocity_loss.detach().cpu()
                    ),
                    "pre_contact_position_loss": float(
                        pre_contact_position_loss.detach().cpu()
                    ),
                    "post_contact_position_loss": float(
                        post_contact_position_loss.detach().cpu()
                    ),
                    "contact_frame_position_weight": contact_weight,
                    "post_contact_worst_position_loss": float(
                        worst_position_loss.detach().cpu()
                    ),
                    "gradient_norm": float(grad_norm.max().detach().cpu()),
                    "arm_gradient_norm": float(arm_grad_norm.max().detach().cpu()),
                    "root_gradient_norm": float(root_grad_norm.max().detach().cpu()),
                }
            )
        return guided.detach().to(original_dtype)


def make_contact_guidance(
    spec: Optional[dict],
    palm_position_fn: Callable[[torch.Tensor, str], torch.Tensor],
) -> Optional[ContactGuidance]:
    if spec is None:
        return None
    return ContactGuidance(ContactGuidanceConfig(**spec), palm_position_fn)
