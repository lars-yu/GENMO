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
    SMPLH_LEFT_LEG,
    SMPLH_RIGHT_LEG,
    SMPLH_SPINE,
)


MOTION_DIM = 151
BODY_POSE_SLICE = slice(0, 126)
ROOT_VELOCITY_SLICE = slice(148, 151)
WRIST_JOINT_INDEX = {"left": 20, "right": 21}
# GENMO global is Y-up (gravity = -Y); the ground is the X-Z plane.
GROUND_NORMAL = (0.0, 1.0, 0.0)

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


def _body_channel_slice(name: str) -> slice:
    """6D channel range for a named body joint (verified, never hardcoded)."""
    body_names = SMPLH_JOINT_NAMES[1:22]
    k = body_names.index(name)
    return slice(k * 6, (k + 1) * 6)


def joint_group_names(hand: str) -> Dict[str, list]:
    """Map v23 optimization groups to their SMPL-X joint names."""
    hand = str(hand).lower()
    if hand not in ("left", "right"):
        raise ValueError(f"selected hand must be left or right, got {hand!r}")
    arm = SMPLH_LEFT_ARM if hand == "left" else SMPLH_RIGHT_ARM
    return {
        "arm": list(arm),
        "torso": list(SMPLH_SPINE),
        "legs": list(SMPLH_LEFT_LEG) + list(SMPLH_RIGHT_LEG),
    }


def group_channel_slices(hand: str) -> Dict[str, list]:
    """Per-group 6D channel slices; root is the 148:151 velocity slice."""
    names = joint_group_names(hand)
    slices = {group: [_body_channel_slice(n) for n in ns] for group, ns in names.items()}
    slices["root"] = [ROOT_VELOCITY_SLICE]
    return slices


def leg_channel_slices(hand: str) -> Dict[str, slice]:
    """Left/right leg 6D ranges (hip/knee/ankle/foot), by name."""
    ranges = {}
    for name in list(SMPLH_LEFT_LEG) + list(SMPLH_RIGHT_LEG):
        ranges[name] = _body_channel_slice(name)
    return ranges


def allowed_channel_mask(
    hand: str,
    *,
    include_root: bool,
    include_torso: bool = False,
    include_legs: bool = False,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return the verified 151-D active-channel mask.

    Arm channels are always active.  ``include_torso`` adds the spine joints,
    ``include_legs`` adds both legs, and ``include_root`` adds the root velocity
    slice.  The two new flags default to False so arm-only / arm+root callers
    (and the v22 tests) keep the original mask.
    """
    mask = torch.zeros(MOTION_DIM, device=device, dtype=dtype)
    slices = group_channel_slices(hand)
    for channel_slice in slices["arm"]:
        mask[channel_slice] = 1
    if include_torso:
        for channel_slice in slices["torso"]:
            mask[channel_slice] = 1
    if include_legs:
        for channel_slice in slices["legs"]:
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
    # Post-contact contact is a soft 2 cm hold region rather than a rigid
    # point-to-point attachment. Small palm motion inside the region is left
    # to the GENMO prior.
    post_contact_hold_radius: float = 0.02
    post_contact_relative_velocity_weight: float = 8.0
    post_contact_relative_step_tolerance: float = 0.01
    # Penetration penalty: keep the palm at least ``contact_min_clearance`` m
    # OUTSIDE the object surface along the per-frame outward normal (which
    # rotates with the object), so the hand mesh rests on the surface instead
    # of clipping through it as the object is lifted/rotated.
    object_surface_normals: Optional[torch.Tensor] = None  # [T,3] outward, guidance-global
    # v24: the palm-point guidance target is the TRUE surface point (no standoff),
    # and full-hand-mesh penetration is left to the separate finger stage, so the
    # GENMO-stage clearance/penetration terms are off by default.  A positive
    # clearance must never exceed the post-contact hold radius (checked below).
    contact_min_clearance: float = 0.0
    penetration_weight: float = 0.0
    # Give frames outside the hold region an additional max-error signal so
    # the average loss cannot hide a brief visible release.
    post_contact_worst_frame_weight: float = 0.5
    post_contact_worst_frame_fraction: float = 0.125
    post_contact_terminal_position_weight: float = 1.0
    post_contact_terminal_frames: int = 16
    contact_frame_position_weight: float = 2.0
    contact_frame_weight_radius: int = 2
    # v25 contact-closure: an INDEPENDENT contact-frame position term (not folded
    # into the ~30-frame weighted average) so the single contact frame is not
    # diluted; and a longer t=0 inner loop that runs the arm+torso to real
    # convergence (2 cm, or 3 consecutive improvements < 0.2 mm) instead of a
    # fixed 10 steps, keeping the minimum-contact-error candidate.
    contact_frame_direct_weight: float = 4.0
    t0_max_inner_steps: int = 40
    inner_convergence_delta_m: float = 0.0002
    inner_convergence_patience: int = 3
    # v25c: the t=0 inner loop keeps optimizing until the post-contact follow is
    # also within this target (not just the contact frame), so the hand tracks
    # the lifted object instead of the loop quitting once the contact frame hits
    # 2 cm.  <=0 keeps the legacy contact-frame-only stop.
    post_contact_reach_threshold: float = 0.03
    # v25b: penalize the guided palm moving FASTER frame-to-frame than the
    # natural (reference) motion over the approach/contact window, beyond a small
    # slack.  This removes the "sudden velocity increase" (lunge) that a strong
    # contact pull would otherwise back-load into the last few frames, without
    # capping the reach itself.  <=0 disables.
    palm_velocity_weight: float = 0.0
    palm_velocity_slack_m: float = 0.005
    # v25b: spread the reach across the approach window.  Instead of pulling
    # every pre-contact frame toward the SAME static contact point (which
    # back-loads the motion into a lunge 1-2 frames before contact), give each
    # approach frame a target that ramps (smoothstep) from the natural reference
    # palm to the contact point, so the palm moves a little every frame.
    interpolate_pre_contact_target: bool = False
    # v25b: minimum weight applied to pre-contact approach frames (0 = legacy
    # 0->1 ramp).  A floor makes guidance move the palm throughout the approach
    # instead of only in the last few frames.  Paired with the interpolated
    # pre-contact target so the early pull is gentle (target ~= natural palm).
    approach_weight_floor: float = 0.0
    # v25b: hard line-search cap (metres/frame) on the max frame-to-frame palm
    # step within the approach window.  Rejects any inner update that would make
    # the hand lunge; the reach is forced to spread across the approach.  <=0
    # disables (no cap).
    approach_max_step_m: float = 0.0
    grad_clip_norm: float = 1.0
    # Legacy single-group caps.  Retained for backward compatibility with old
    # specs/tests; the v22 inner-loop path below no longer reads them.  The
    # authoritative caps are the separate arm_*/root_* fields.
    max_guidance_update_norm: float = 0.25
    final_guidance_update_norm: float = 0.05
    arm_guidance_fade_fraction: float = 0.20
    root_guidance_multiplier: float = 1.0
    root_guidance_final_scale: float = 0.10
    # v22: multi-step ("inner loop") guidance near the final DDIM steps, with
    # arm and root updates capped and clipped completely independently.  Each
    # inner iteration re-decodes the palm and recomputes the gradient rather
    # than walking along a single stale gradient.
    late_inner_steps: int = 2
    final_inner_steps: int = 5
    inner_step_late_threshold: int = 4
    arm_max_update_cap: float = 0.60
    arm_final_update_cap: float = 0.25
    root_max_update_cap: float = 0.05
    root_final_update_cap: float = 0.02
    # Temporal Gaussian smoothing of the root-velocity gradient (odd kernel;
    # <= 1 disables).  Root velocity integrates into every later frame, so its
    # gradient must be smoothed instead of updated per-frame independently.
    root_gradient_smooth_kernel: int = 9
    # Temporal Gaussian smoothing of the arm/torso/leg gradient (odd kernel;
    # <=1 disables).  Prevents the hand from jumping frame-to-frame when the
    # per-frame update cap is large (post-contact jitter).
    arm_gradient_smooth_kernel: int = 9
    # Backtracking accept: only keep an inner update when it reduces the
    # contact loss and does not exceed the palm-jump / root-step safety limits.
    inner_line_search: bool = True
    # Safety backstops on the change a single accepted update may introduce
    # (palm displacement in metres vs the pre-step palm; root-velocity channel
    # change in normalized units).  The loss-decrease test is the primary gate;
    # these only reject pathological teleports.
    inner_palm_jump_limit: float = 0.50
    inner_root_step_limit: float = 0.10
    reach_error_threshold: float = 0.02
    # ---- v23 whole-body guidance (all default-off so arm-only/v22 unchanged) ----
    # Group membership beyond the always-on arm.  torso = small spine help;
    # legs+root together form the "whole-body" (walking) candidate.
    include_torso: bool = False
    include_legs: bool = False
    # Distance-dependent soft activation of root/legs (metres, palm->object).
    root_activation_distance_min: float = 0.06
    root_activation_distance_max: float = 0.15
    # root/legs fade to zero at t=0 (arm stays 1.0) so the denoiser can still
    # reconcile the legs; fraction of the final DDIM steps over which they fade.
    root_leg_fade_fraction: float = 0.30
    # Independent per-frame update caps for the torso and leg groups.
    torso_max_update_cap: float = 0.05
    torso_final_update_cap: float = 0.01
    leg_max_update_cap: float = 0.05
    leg_final_update_cap: float = 0.02
    # Ground-plane root target (horizontal residual the root should absorb),
    # in GENMO global metres, supplied by the driver from the arm-only residual.
    root_target_delta_global: Optional[torch.Tensor] = None
    max_root_correction: float = 0.25
    w_root_target: float = 1.0
    w_root_vertical_lock: float = 5.0
    # Support-foot / ground / leg-reference losses (whole-body only).
    foot_contact_probs: Optional[torch.Tensor] = None  # [T,4] L_ankle,L_foot,R_ankle,R_foot
    ground_height: Optional[float] = None
    support_foot_weight: float = 2.0
    ground_contact_weight: float = 1.0
    leg_reference_weight: float = 1.0
    # Arm-reference / smoothness (encourage a natural reach, not a teleport).
    arm_reference_weight: float = 0.0
    arm_smoothness_weight: float = 0.0
    # Palm-position acceleration penalty over the pre-contact approach window
    # [contact_frame - transition, contact_frame].  Every approach frame targets
    # the same contact point, so without this the hand lunges in the final few
    # frames; penalizing acceleration spreads the reach into a smooth, roughly
    # constant-velocity approach.  <=0 disables.
    approach_smoothness_weight: float = 0.0
    # v24 whole-body action terms.  Root velocity residual (candidate vs the
    # desired reference+ramp trajectory) so gradients shape 148:151 correctly and
    # the post-contact residual returns to zero.  Torso reference/temporal keep
    # the chest from twisting.  Elbow-direction keeps the natural bend plane.
    w_root_velocity: float = 1.0
    torso_reference_weight: float = 0.0
    torso_smoothness_weight: float = 0.0
    elbow_direction_weight: float = 0.0
    # Line-search safety: reject a candidate whose support-foot slide (candidate
    # foot velocity relative to the reference motion, metres/frame) exceeds this.
    foot_slide_limit: float = 0.05
    diffusion_steps: int = 50
    human_depth_scale: float = 1.0
    camera_origin_global: Optional[torch.Tensor] = None
    # v23 rigid frame-0 depth placement: a pure translation (GENMO global
    # metres) added to every decoded human point so the frame-0 human depth
    # matches GT.  No scaling, no betas change -> body size is preserved.
    human_translation_global: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Regression guard: if clearance is ever re-enabled it must not exceed
        # the post-contact hold radius, or the palm target becomes unsatisfiable
        # (must be >= clearance off the surface AND <= hold_radius from it).
        if float(self.contact_min_clearance) > float(self.post_contact_hold_radius) + 1e-9:
            raise ValueError(
                "contact_min_clearance must be <= post_contact_hold_radius "
                f"(got {self.contact_min_clearance} > {self.post_contact_hold_radius})"
            )


class ContactGuidance:
    """A ``denoised_fn`` callback for ``ddim_sample_loop_with_aux``."""

    def __init__(
        self,
        config: ContactGuidanceConfig,
        kinematics_fn: Callable[[torch.Tensor, str], object],
    ):
        self.config = config
        # ``kinematics_fn`` may return either a [B,T,3] palm tensor (v22 / test
        # fakes -> palm-only, whole-body losses disabled) or a dict with keys
        # palm_position/root_position/foot_positions (v23 -> whole body).
        self.kinematics_fn = kinematics_fn
        self.palm_position_fn = kinematics_fn  # backward-compatible alias
        self.step_diagnostics = []

    def _decode_kinematics(self, motion: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Normalize the kinematics callback output to a dict."""
        result = self.kinematics_fn(motion, self.config.selected_hand)
        if isinstance(result, dict):
            if "palm_position" not in result:
                raise ValueError("kinematics_fn dict must contain 'palm_position'")
            return result
        return {"palm_position": result}

    def _active_mask(self, device, dtype) -> torch.Tensor:
        return allowed_channel_mask(
            self.config.selected_hand,
            include_root=self.config.include_root,
            include_torso=self.config.include_torso,
            include_legs=self.config.include_legs,
            device=device,
            dtype=dtype,
        )

    def _whole_body_active(self, kin: Dict[str, torch.Tensor]) -> bool:
        """Whole-body losses need the leg/root group and real foot kinematics."""
        return (
            self.config.include_legs
            and self.config.include_root
            and "foot_positions" in kin
            and "root_position" in kin
        )

    def _time_scale(self, timestep: torch.Tensor, *, root: bool) -> torch.Tensor:
        # DDIM visits large t first. Arm guidance must remain active at t=0:
        # the last deterministic DDIM sample is the x0 that is returned. Root
        # guidance is weakened near t=0, but is not completely disabled.
        t = timestep.float() / max(float(self.config.diffusion_steps - 1), 1.0)
        if root:
            final_scale = float(self.config.root_guidance_final_scale)
            if not 0.0 <= final_scale <= 1.0:
                raise ValueError(
                    "root_guidance_final_scale must lie in [0, 1], "
                    f"got {final_scale}"
                )
            progress = torch.clamp(t / 0.35, min=0.0, max=1.0)
            t = final_scale + (1.0 - final_scale) * progress
        else:
            t = torch.ones_like(t)
        return t.reshape(-1, 1, 1)

    def _scheduled_group_cap(
        self, timestep: torch.Tensor, final_cap: float, max_cap: float
    ) -> torch.Tensor:
        """Per-frame update cap that ramps from ``max_cap`` (early DDIM steps)
        down to a non-zero ``final_cap`` at ``t=0``.  Applied independently to
        the arm and root groups so each keeps its own update budget."""
        maximum = float(max_cap)
        final = float(final_cap)
        fade = float(self.config.arm_guidance_fade_fraction)
        if not torch.isfinite(torch.tensor(maximum)) or maximum <= 0.0:
            raise ValueError("update cap maximum must be finite and positive")
        if (
            not torch.isfinite(torch.tensor(final))
            or final <= 0.0
            or final > maximum
        ):
            raise ValueError(
                "final update cap must be finite and lie in "
                f"(0, max], got final={final}, max={maximum}"
            )
        if not 0.0 < fade <= 1.0:
            raise ValueError(
                "arm_guidance_fade_fraction must lie in (0, 1], "
                f"got {fade}"
            )
        t = timestep.float() / max(float(self.config.diffusion_steps - 1), 1.0)
        progress = torch.clamp(t / fade, min=0.0, max=1.0)
        cap = final + (maximum - final) * progress
        return cap.reshape(-1, 1, 1)

    def _scheduled_update_cap(self, timestep: torch.Tensor) -> torch.Tensor:
        """Legacy single-group cap (kept for backward-compatible callers)."""
        return self._scheduled_group_cap(
            timestep,
            self.config.final_guidance_update_norm,
            self.config.max_guidance_update_norm,
        )

    def _root_leg_time_scale(self, timestep: torch.Tensor) -> torch.Tensor:
        """Root/leg time scale: 1.0 early, ramping to 0.0 at t=0 so the final
        DDIM step never edits root velocity / legs directly (avoids foot skate;
        the denoiser gets no chance to re-coordinate the legs at t=0)."""
        fade = float(self.config.root_leg_fade_fraction)
        if not 0.0 < fade <= 1.0:
            raise ValueError(f"root_leg_fade_fraction must lie in (0, 1], got {fade}")
        t = timestep.float() / max(float(self.config.diffusion_steps - 1), 1.0)
        return torch.clamp(t / fade, min=0.0, max=1.0).reshape(-1, 1, 1)

    def _group_time_scale(self, group: str, timestep: torch.Tensor) -> torch.Tensor:
        """Per-group DDIM time schedule (§六).  arm/torso stay on to t=0;
        root/legs fade to zero at t=0."""
        if group in ("root", "legs"):
            return self._root_leg_time_scale(timestep)
        return torch.ones_like(timestep.float()).reshape(-1, 1, 1)

    def _distance_group_weights(self, palm_object_distance: float) -> Dict[str, float]:
        """Soft distance weighting (§四): arm always 1; root/legs ramp in with
        distance via smoothstep; torso sits between."""
        lo = float(self.config.root_activation_distance_min)
        hi = float(self.config.root_activation_distance_max)
        if not 0.0 <= lo < hi:
            raise ValueError(
                "root_activation_distance_min/max must satisfy 0<=min<max, "
                f"got min={lo}, max={hi}"
            )
        u = min(max((float(palm_object_distance) - lo) / (hi - lo), 0.0), 1.0)
        root_w = u * u * (3.0 - 2.0 * u)  # smoothstep
        return {
            "arm": 1.0,
            "torso": 0.5 + 0.5 * root_w,
            "legs": root_w,
            "root": root_w,
        }

    def _root_offset_ramp(self, frames: int, device, dtype) -> torch.Tensor:
        """Smooth 0->1 ramp distributing the root correction from the approach
        start to the contact frame, then held at 1 afterwards (§五)."""
        frame = int(self.config.contact_frame)
        transition = max(0, int(self.config.contact_transition_frames))
        approach_start = max(0, frame - transition) if transition else frame
        ramp = torch.zeros(frames, device=device, dtype=dtype)
        if frame > approach_start:
            steps = frame - approach_start + 1
            u = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype)
            ramp[approach_start : frame + 1] = u * u * (3.0 - 2.0 * u)
        ramp[frame:] = 1.0
        return ramp

    def _inner_step_schedule(self, timestep: torch.Tensor) -> int:
        """Number of inner guidance iterations for this DDIM step.

        DDIM visits ``t`` from large to zero.  The final clean x0 (``t=0``) is
        what gets returned, so it receives the most refinement; a short band of
        late steps gets a moderate amount; every earlier step keeps the single
        cheap update it had before.
        """
        t = int(timestep.reshape(-1)[0].detach().cpu())
        late = max(1, int(self.config.late_inner_steps))
        final = max(1, int(self.config.final_inner_steps))
        threshold = max(0, int(self.config.inner_step_late_threshold))
        if t == 0:
            # §3: run the final clean-x0 step to convergence (up to this cap; the
            # loop below early-stops at 2 cm or on the small-improvement patience).
            return max(final, int(self.config.t0_max_inner_steps))
        if t <= threshold:
            return late
        return 1

    def _assemble(
        self,
        base_x0: torch.Tensor,
        active_indices: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Rebuild the full [B,T,151] motion from the guided active channels."""
        return torch.index_copy(base_x0, -1, active_indices, active)

    def _compute_guidance_loss(self, x0: torch.Tensor, timestep: torch.Tensor):
        """Decode the palm from ``x0`` and build the guidance loss.

        ``x0`` must already be the full [B,T,151] motion (with the guided
        channels attached to the autograd graph); the gradient is taken w.r.t.
        those channels by the caller.  Returns ``(total_loss, comps)`` where
        ``comps`` carries the individual loss terms plus the decoded ``palm``
        and per-frame ``target_seq`` used by the inner-loop metrics.
        """
        base_x0 = x0.detach()
        reference = self.config.reference_motion.to(base_x0).detach()
        target = self.config.object_surface_target.to(base_x0).reshape(1, 1, 3)
        mask = allowed_channel_mask(
            self.config.selected_hand,
            include_root=self.config.include_root,
            device=base_x0.device,
            dtype=base_x0.dtype,
        ).reshape(1, 1, -1)

        kin = self._decode_kinematics(x0)
        palm = kin["palm_position"]
        if palm.ndim != 3 or palm.shape[-1] != 3:
            raise ValueError(f"kinematics_fn palm_position must be [B,T,3], got {palm.shape}")
        depth_scale = float(self.config.human_depth_scale)
        if not torch.isfinite(torch.tensor(depth_scale)) or depth_scale <= 0.0:
            raise ValueError(f"human_depth_scale must be finite and positive, got {depth_scale}")
        camera_origin = None
        if self.config.camera_origin_global is not None:
            camera_origin = self.config.camera_origin_global.to(base_x0)
        elif depth_scale != 1.0:
            raise ValueError("camera_origin_global is required when human_depth_scale != 1")
        t0 = None
        if self.config.human_translation_global is not None:
            t0 = self.config.human_translation_global.to(base_x0)

        def _place_human(p):
            # Apply the same frame-0 placement to every decoded human point:
            # (C) a GT-depth similarity about the camera origin (resolves GENMO's
            # monocular scale/depth ambiguity), and/or (A) a rigid translation.
            shape = [1] * (p.ndim - 1) + [3]
            if camera_origin is not None:
                cam = camera_origin.reshape(shape)
                p = cam + depth_scale * (p - cam)
            if t0 is not None:
                p = p + t0.reshape(shape)
            return p

        palm = _place_human(palm)
        # v24: decode the frozen baseline (reference) kinematics ONCE and place
        # them in the same frame-0 aligned coordinates as the candidate.  The
        # root/foot/arm terms below are residuals on THIS natural motion, so the
        # original walk/bend/gait is preserved instead of collapsing to frame 0.
        if getattr(self, "_ref_kin_raw", None) is None:
            self._ref_kin_raw = {
                k: v.detach() for k, v in self._decode_kinematics(reference).items()
            }
        ref_kin = self._ref_kin_raw
        ref_palm = _place_human(ref_kin["palm_position"])
        ref_root = (
            _place_human(ref_kin["root_position"]) if "root_position" in ref_kin else None
        )
        ref_feet = (
            _place_human(ref_kin["foot_positions"]) if "foot_positions" in ref_kin else None
        )
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
        post_contact_max_distance = base_x0.new_zeros(())
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
                # Keep the contact frame in the post-contact segment.  The
                # pre-contact segment is either the static contact point (legacy)
                # or a smoothstep interpolation from the natural reference palm to
                # the contact point (v25b) that spreads the reach and removes the
                # last-frame lunge.
                if (
                    bool(self.config.interpolate_pre_contact_target)
                    and frame > 0
                    and ref_palm.shape[1] >= frame + 1
                ):
                    transition = max(0, int(self.config.contact_transition_frames))
                    a_start = max(0, frame - transition) if transition else 0
                    idx = torch.arange(frame, device=base_x0.device, dtype=base_x0.dtype)
                    denom = float(max(frame - a_start, 1))
                    u = ((idx - a_start) / denom).clamp(0.0, 1.0)
                    s = (u * u * (3.0 - 2.0 * u)).reshape(1, frame, 1)  # smoothstep
                    natural = ref_palm[:, :frame, :]
                    reach = pre_target.reshape(1, 1, 3) - ref_palm[:, frame:frame + 1, :]
                    pre_seq = natural + s * reach
                    target_seq = torch.cat(
                        [pre_seq, post_targets.reshape(1, -1, 3)], dim=1
                    )
                else:
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
                # v25b: lift the pre-contact weights to a floor so the WHOLE
                # approach window is guided (not just the last few frames).  With
                # a near-zero ramp the reach collapses into a 1-frame lunge; a
                # floor lets each approach frame track its interpolated target so
                # the reach spreads into a smooth, roughly constant-velocity move.
                floor = float(self.config.approach_weight_floor)
                if floor > 0.0:
                    ramp = floor + (1.0 - floor) * ramp
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
            hold_radius = float(self.config.post_contact_hold_radius)
            if not torch.isfinite(torch.tensor(hold_radius)) or hold_radius < 0.0:
                raise ValueError(
                    "post_contact_hold_radius must be finite and non-negative"
                )
            # Before contact, reach the scheduled target normally. From
            # the contact frame onward, only correct separation exceeding
            # the 2 cm hold region. This prevents guidance from fighting
            # GENMO over harmless sliding within the hand.
            post_contact_distance = torch.linalg.vector_norm(
                relative_position[:, frame:], dim=-1
            )
            post_contact_max_distance = post_contact_distance.max()
            post_contact_excess = torch.relu(post_contact_distance - hold_radius)
            post_contact_residual = post_contact_excess.square()
            effective_residual = torch.cat(
                (residual[:, :frame], post_contact_residual), dim=1
            )
            position_loss = (
                effective_residual * weights.reshape(1, -1)
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
                        post_contact_residual * post_weights.reshape(1, -1)
                    ).sum() / post_weights.sum().clamp_min(1.0)
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
                relative_step_tolerance = float(
                    self.config.post_contact_relative_step_tolerance
                )
                if (
                    not torch.isfinite(torch.tensor(relative_step_tolerance))
                    or relative_step_tolerance < 0.0
                ):
                    raise ValueError(
                        "post_contact_relative_step_tolerance must be finite "
                        "and non-negative"
                    )
                relative_step = torch.linalg.vector_norm(relative_velocity, dim=-1)
                relative_velocity_loss = torch.relu(
                    relative_step - relative_step_tolerance
                ).square().mean()
            relative_velocity_weight = float(
                self.config.post_contact_relative_velocity_weight
            )
            if not torch.isfinite(torch.tensor(relative_velocity_weight)) or relative_velocity_weight < 0:
                raise ValueError(
                    "post_contact_relative_velocity_weight must be finite and non-negative"
                )
            # Penetration penalty: signed clearance = how far the palm is
            # OUTSIDE the surface along the per-frame outward normal (rotates
            # with the object).  Penalize (steeply) any frame where the palm is
            # closer than contact_min_clearance / inside the object.
            penetration_loss = base_x0.new_zeros(())
            if self.config.object_surface_normals is not None:
                normals = self.config.object_surface_normals.to(base_x0)
                if normals.shape[0] != palm.shape[1]:
                    raise ValueError(
                        "object_surface_normals must have shape [T,3], got "
                        f"{tuple(normals.shape)} for T={palm.shape[1]}"
                    )
                normals = normals.reshape(1, palm.shape[1], 3)
                min_clearance = float(self.config.contact_min_clearance)
                signed_clearance = ((palm - target_seq) * normals).sum(dim=-1)  # [B,T]
                penetration = torch.relu(min_clearance - signed_clearance)  # >0 if too close/inside
                penetration_loss = (
                    penetration.square() * weights.reshape(1, -1)
                ).sum() / weights.sum().clamp_min(1.0)
            # Smooth the pre-contact approach: penalize palm-position
            # acceleration across the approach window so the reach is spread
            # evenly instead of lunging in the last few frames.
            approach_smoothness_loss = base_x0.new_zeros(())
            approach_weight = float(self.config.approach_smoothness_weight)
            if approach_weight < 0.0 or not torch.isfinite(torch.tensor(approach_weight)):
                raise ValueError("approach_smoothness_weight must be finite and >= 0")
            if approach_weight > 0.0:
                approach_lo = int(ramp_start)
                approach_hi = min(frame + 1, palm.shape[1])
                if approach_hi - approach_lo >= 3:
                    seg = palm[:, approach_lo:approach_hi, :]
                    approach_accel = seg[:, 2:] - 2.0 * seg[:, 1:-1] + seg[:, :-2]
                    approach_smoothness_loss = approach_accel.square().sum(dim=-1).mean()
            # v25b: velocity-vs-reference penalty over the approach + a short
            # post-contact window.  Penalize only the palm SPEED that exceeds the
            # natural reference speed plus a small slack, so the hand cannot
            # suddenly accelerate toward the object.
            palm_velocity_loss = base_x0.new_zeros(())
            palm_velocity_weight = float(self.config.palm_velocity_weight)
            if palm_velocity_weight < 0.0 or not torch.isfinite(torch.tensor(palm_velocity_weight)):
                raise ValueError("palm_velocity_weight must be finite and >= 0")
            if palm_velocity_weight > 0.0:
                cw_radius = max(0, int(self.config.contact_frame_weight_radius))
                v_lo = int(ramp_start)
                v_hi = min(frame + cw_radius + 1, palm.shape[1])
                if v_hi - v_lo >= 2:
                    cand_step = torch.linalg.vector_norm(
                        palm[:, v_lo + 1:v_hi] - palm[:, v_lo:v_hi - 1], dim=-1
                    )
                    ref_step = torch.linalg.vector_norm(
                        ref_palm[:, v_lo + 1:v_hi] - ref_palm[:, v_lo:v_hi - 1], dim=-1
                    )
                    slack = float(self.config.palm_velocity_slack_m)
                    palm_velocity_loss = torch.relu(
                        cand_step - ref_step - slack
                    ).square().mean()
            # §1: independent contact-frame position term.  The weighted mean
            # above spreads one frame's error across ~30 frames; this term keeps
            # the contact frame (and a tiny post-contact window) as a first-class
            # objective so closing to the 2 cm hold target is not averaged away.
            # It uses the SAME hold-aware residual as position_loss (zero gradient
            # once within the hold radius) and never touches pre-contact frames.
            cw_radius = max(0, int(self.config.contact_frame_weight_radius))
            cw_hi = min(palm.shape[1], frame + cw_radius + 1)
            contact_frame_position_loss = effective_residual[:, frame:cw_hi].mean()
            contact_loss = (
                position_loss
                + float(self.config.contact_frame_direct_weight) * contact_frame_position_loss
                + worst_weight * worst_position_loss
                + relative_velocity_weight * relative_velocity_loss
                + float(self.config.penetration_weight) * penetration_loss
                + approach_weight * approach_smoothness_loss
                + palm_velocity_weight * palm_velocity_loss
            )
        else:
            contact_loss = ((palm[:, start:end] - target) ** 2).sum(dim=-1).mean()
            worst_position_loss = base_x0.new_zeros(())
            target_seq = target.expand(1, palm.shape[1], 3)
            penetration_loss = base_x0.new_zeros(())
            contact_frame_position_loss = base_x0.new_zeros(())
            palm_velocity_loss = base_x0.new_zeros(())
        delta = (x0 - reference) * mask
        reference_loss = delta.square().sum() / mask.sum().clamp_min(1.0) / x0.shape[1]
        if delta.shape[1] > 1:
            temporal_delta = delta[:, 1:] - delta[:, :-1]
            temporal_loss = temporal_delta.square().mean()
        else:
            temporal_loss = delta.new_zeros(())

        # ---- Arm reference / smoothness (§八): a natural reach, not a teleport ----
        arm_reference_loss = base_x0.new_zeros(())
        arm_smoothness_loss = base_x0.new_zeros(())
        arm_slices = group_channel_slices(self.config.selected_hand)["arm"]
        arm_idx = torch.cat([
            torch.arange(s.start, s.stop, device=base_x0.device) for s in arm_slices
        ])
        arm_delta = (x0 - reference).index_select(-1, arm_idx)
        arm_reference_loss = arm_delta.square().mean()
        if arm_delta.shape[1] > 2:
            arm_accel = arm_delta[:, 2:] - 2.0 * arm_delta[:, 1:-1] + arm_delta[:, :-2]
            arm_smoothness_loss = arm_accel.square().mean()

        # §8.1 SO(3) geodesic arm reference (preferred over raw 6D-channel L2)
        # and §8.2 elbow-direction, only when the kinematics callback exposes
        # per-joint rotations / joint positions (the real GENMO decode does).
        arm_geodesic_loss = base_x0.new_zeros(())
        arm_geodesic_per_joint = None
        elbow_direction_loss = base_x0.new_zeros(())
        if "arm_rotations" in kin and "arm_rotations" in ref_kin:
            Rc = kin["arm_rotations"]  # [B,T,4,3,3] collar/shoulder/elbow/wrist
            Rr = ref_kin["arm_rotations"].to(Rc)
            R_delta = Rc @ Rr.transpose(-1, -2)
            trace = R_delta[..., 0, 0] + R_delta[..., 1, 1] + R_delta[..., 2, 2]
            cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            angle = torch.acos(cos)  # [B,T,4], radians
            arm_geodesic_loss = angle.square().mean()
            arm_geodesic_per_joint = angle.square().mean(dim=(0, 1))  # [4]
        pos_keys = ("shoulder_position", "elbow_position", "wrist_position")
        if all(k in kin for k in pos_keys) and all(k in ref_kin for k in pos_keys):
            def _elbow_plane_normal(kd):
                sh = _place_human(kd["shoulder_position"])
                el = _place_human(kd["elbow_position"])
                wr = _place_human(kd["wrist_position"])
                ua = torch.nn.functional.normalize(el - sh, dim=-1)
                fa = torch.nn.functional.normalize(wr - el, dim=-1)
                return torch.nn.functional.normalize(torch.cross(ua, fa, dim=-1), dim=-1)
            n_cand = _elbow_plane_normal(kin)
            n_ref = _elbow_plane_normal(ref_kin)
            dot = (n_cand * n_ref).sum(dim=-1)  # [B,T] cos angle between planes
            # Only penalize large deviations (>~60deg, dot<0.5); smooth change is free.
            elbow_direction_loss = torch.relu(0.5 - dot).square().mean()
        # Prefer the geodesic arm reference when rotations are available.
        arm_reference_term = (
            arm_geodesic_loss if arm_geodesic_per_joint is not None else arm_reference_loss
        )

        # §8.3 Torso reference + temporal (only meaningful when torso is active).
        torso_reference_loss = base_x0.new_zeros(())
        torso_smoothness_loss = base_x0.new_zeros(())
        if self.config.include_torso:
            torso_slices = group_channel_slices(self.config.selected_hand)["torso"]
            torso_idx = torch.cat([
                torch.arange(s.start, s.stop, device=base_x0.device) for s in torso_slices
            ])
            torso_delta = (x0 - reference).index_select(-1, torso_idx)
            torso_reference_loss = torso_delta.square().mean()
            if torso_delta.shape[1] > 2:
                torso_accel = (
                    torso_delta[:, 2:] - 2.0 * torso_delta[:, 1:-1] + torso_delta[:, :-2]
                )
                torso_smoothness_loss = torso_accel.square().mean()

        # ---- Whole-body root / foot terms (§三,§四,§五,§九), whole-body only ----
        root_velocity_loss = base_x0.new_zeros(())
        foot_slide_m = 0.0
        root_target_loss = base_x0.new_zeros(())
        root_vertical_loss = base_x0.new_zeros(())
        support_foot_loss = base_x0.new_zeros(())
        ground_contact_loss = base_x0.new_zeros(())
        leg_reference_loss = base_x0.new_zeros(())
        if self._whole_body_active(kin):
            up = torch.as_tensor(GROUND_NORMAL, device=base_x0.device, dtype=base_x0.dtype)
            # Same frame-0 placement (similarity and/or translation) as the palm.
            root_pos = _place_human(kin["root_position"])  # [B,T,3]
            feet = _place_human(kin["foot_positions"])  # [B,T,4,3]
            # v24: residuals are measured against the frozen REFERENCE motion,
            # not frame 0, so the original walk/bend/gait survives.
            ref_root_b = ref_root if ref_root is not None else root_pos.detach()
            ref_feet_b = ref_feet if ref_feet is not None else feet.detach()
            # §4 vertical: guidance may not change root height RELATIVE to the
            # reference (the reference's own vertical motion is fully preserved).
            root_vertical_loss = (
                ((root_pos - ref_root_b) * up).sum(dim=-1).square().mean()
            )
            # §3 root target: desired_root = reference_root + residual ramp.
            # The horizontal ground-plane delta is distributed by a smoothstep
            # ramp (0 before approach_start, 1 at contact, held after).
            if self.config.root_target_delta_global is not None:
                delta_g = self.config.root_target_delta_global.to(base_x0).reshape(1, 1, 3)
                delta_g = delta_g - (delta_g * up).sum(dim=-1, keepdim=True) * up  # horizontal
            else:
                delta_g = base_x0.new_zeros((1, 1, 3))
            ramp = self._root_offset_ramp(
                root_pos.shape[1], base_x0.device, base_x0.dtype
            ).reshape(1, -1, 1)
            desired_root = ref_root_b + ramp * delta_g
            horiz = (root_pos - desired_root)
            horiz = horiz - (horiz * up).sum(dim=-1, keepdim=True) * up
            root_target_loss = horiz.square().sum(dim=-1).mean()
            # §5 root velocity: match the desired-root velocity so gradients flow
            # into 148:151 correctly; post-contact the residual is constant, so
            # the residual velocity (candidate - reference) returns to zero.
            if root_pos.shape[1] > 1:
                candidate_vel = root_pos[:, 1:] - root_pos[:, :-1]
                desired_vel = desired_root[:, 1:] - desired_root[:, :-1]
                root_velocity_loss = (candidate_vel - desired_vel).square().sum(-1).mean()
            # §9 feet: constrain RELATIVE to the reference motion (no absolute
            # lock).  A swing foot that moves in the reference is free to move;
            # only deviations from the reference are penalized, weighted by the
            # contact probability, so both feet are never locked together.
            if self.config.foot_contact_probs is not None and feet.shape[1] > 1:
                probs = self.config.foot_contact_probs.to(base_x0).reshape(1, -1, 4)
                probs = probs[:, : feet.shape[1]]
                cand_foot_vel = feet[:, 1:] - feet[:, :-1]
                ref_foot_vel = ref_feet_b[:, 1:] - ref_feet_b[:, :-1]
                vel_resid = (cand_foot_vel - ref_foot_vel).square().sum(dim=-1)  # [B,T-1,4]
                support_foot_loss = (
                    (probs[:, 1:] * vel_resid).sum() / probs[:, 1:].sum().clamp_min(1.0)
                )
                foot_slide_m = float(
                    torch.sqrt(vel_resid.clamp_min(0.0)).max().detach().cpu()
                )
                # Ground contact relative to the reference foot height (never the
                # global minimum, which would drag ankles down to the toes).
                cand_h = (feet * up).sum(dim=-1)  # [B,T,4]
                ref_h = (ref_feet_b * up).sum(dim=-1)
                ground_contact_loss = (
                    probs * (cand_h - ref_h).square()
                ).sum() / probs.sum().clamp_min(1.0)
            # Legs stay close to the GENMO reference pose.
            leg_idx = torch.cat([
                torch.arange(s.start, s.stop, device=base_x0.device)
                for s in leg_channel_slices(self.config.selected_hand).values()
            ])
            leg_delta = (x0 - reference).index_select(-1, leg_idx)
            leg_reference_loss = leg_delta.square().mean()

        total_loss = (
            self.config.w_contact * contact_loss
            + self.config.w_reference * reference_loss
            + self.config.w_temporal * temporal_loss
            + float(self.config.arm_reference_weight) * arm_reference_term
            + float(self.config.arm_smoothness_weight) * arm_smoothness_loss
            + float(self.config.elbow_direction_weight) * elbow_direction_loss
            + float(self.config.torso_reference_weight) * torso_reference_loss
            + float(self.config.torso_smoothness_weight) * torso_smoothness_loss
            + float(self.config.w_root_target) * root_target_loss
            + float(self.config.w_root_vertical_lock) * root_vertical_loss
            + float(self.config.w_root_velocity) * root_velocity_loss
            + float(self.config.support_foot_weight) * support_foot_loss
            + float(self.config.ground_contact_weight) * ground_contact_loss
            + float(self.config.leg_reference_weight) * leg_reference_loss
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
        comps = {
            "contact_loss": contact_loss,
            "reference_loss": reference_loss,
            "temporal_loss": temporal_loss,
            "relative_velocity_loss": relative_velocity_loss,
            "pre_contact_position_loss": pre_contact_position_loss,
            "post_contact_position_loss": post_contact_position_loss,
            "post_contact_max_distance": post_contact_max_distance,
            "worst_position_loss": worst_position_loss,
            "contact_frame_position_loss": contact_frame_position_loss,
            "palm_velocity_loss": palm_velocity_loss,
            "penetration_loss": penetration_loss,
            "contact_weight": contact_weight,
            "palm": palm,
            "target_seq": target_seq,
            "arm_reference_loss": arm_reference_loss,
            "arm_geodesic_loss": arm_geodesic_loss,
            "arm_geodesic_per_joint": arm_geodesic_per_joint,
            "arm_smoothness_loss": arm_smoothness_loss,
            "elbow_direction_loss": elbow_direction_loss,
            "torso_reference_loss": torso_reference_loss,
            "torso_smoothness_loss": torso_smoothness_loss,
            "root_target_loss": root_target_loss,
            "root_vertical_loss": root_vertical_loss,
            "root_velocity_loss": root_velocity_loss,
            "support_foot_loss": support_foot_loss,
            "ground_contact_loss": ground_contact_loss,
            "leg_reference_loss": leg_reference_loss,
            "foot_slide_m": foot_slide_m,
            "kin": kin,
        }
        return total_loss, comps

    def _split_guidance_gradients(self, grad: torch.Tensor, active_indices: torch.Tensor):
        """Split an active-channel gradient into the arm and root groups."""
        arm_in_active = active_indices < ROOT_VELOCITY_SLICE.start
        root_in_active = active_indices >= ROOT_VELOCITY_SLICE.start
        return grad[..., arm_in_active], grad[..., root_in_active], arm_in_active, root_in_active

    @staticmethod
    def _gaussian_time_smooth(g: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Depthwise Gaussian smoothing of a [B,T,C] signal along time.

        Couples adjacent frames so a large per-frame guidance update cannot make
        the joint jump frame-to-frame (the post-contact hand jitter).  Kernel
        <= 1 is a no-op.
        """
        k = int(kernel_size)
        if k <= 1 or g.shape[-1] == 0 or g.shape[1] < 2:
            return g
        if k % 2 == 0:
            k += 1
        sigma = max(k / 6.0, 1e-3)
        xs = torch.arange(k, device=g.device, dtype=g.dtype) - (k - 1) / 2.0
        kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
        kernel = kernel / kernel.sum()
        channels = g.shape[-1]
        weight = kernel.reshape(1, 1, k).repeat(channels, 1, 1)
        smoothed = torch.nn.functional.conv1d(
            g.transpose(1, 2), weight, padding=k // 2, groups=channels
        )
        return smoothed.transpose(1, 2)

    def _smooth_root_gradient(
        self, root_grad: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Temporally smooth and reweight the root-velocity gradient.

        Root velocity integrates into every later frame, so an independent
        per-frame update produces an abrupt body translation at contact.  A
        Gaussian smooth plus a smoothstep approach ramp turns a needed root
        correction into a gradual walk toward the object before contact.
        """
        if root_grad.shape[-1] == 0:
            return root_grad
        # v24: only Gaussian-smooth + reweight.  The old position ramp (0 before
        # approach_start, 1 after contact) was multiplied onto the root-VELOCITY
        # gradient, which pulled the whole approach trajectory back to frame 0 and
        # kept editing the root after contact.  The approach shape now lives in
        # the loss (desired_root = reference_root + ramp*delta, root_velocity_loss
        # → 0 post-contact); the per-group DDIM schedule (0 at t=0) still applies.
        g = self._gaussian_time_smooth(root_grad, self.config.root_gradient_smooth_kernel)
        root_multiplier = float(self.config.root_guidance_multiplier)
        if not torch.isfinite(torch.tensor(root_multiplier)) or root_multiplier < 0.0:
            raise ValueError(
                "root_guidance_multiplier must be finite and non-negative, "
                f"got {root_multiplier}"
            )
        # NB: the DDIM time schedule for root/legs is applied per-group in
        # _apply_group_updates (fades to 0 at t=0); here we only smooth/reweight.
        return g * root_multiplier

    def _group_membership(self, active_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Boolean masks (over the active channels) for arm/torso/legs/root."""
        slices = group_channel_slices(self.config.selected_hand)

        def in_slices(sl_list):
            m = torch.zeros_like(active_indices, dtype=torch.bool)
            for sl in sl_list:
                m |= (active_indices >= sl.start) & (active_indices < sl.stop)
            return m

        return {
            "arm": in_slices(slices["arm"]),
            "torso": in_slices(slices["torso"]) if self.config.include_torso
            else torch.zeros_like(active_indices, dtype=torch.bool),
            "legs": in_slices(slices["legs"]) if self.config.include_legs
            else torch.zeros_like(active_indices, dtype=torch.bool),
            "root": (active_indices >= ROOT_VELOCITY_SLICE.start) if self.config.include_root
            else torch.zeros_like(active_indices, dtype=torch.bool),
        }

    _GROUP_CAPS = {
        "arm": ("arm_final_update_cap", "arm_max_update_cap"),
        "torso": ("torso_final_update_cap", "torso_max_update_cap"),
        "legs": ("leg_final_update_cap", "leg_max_update_cap"),
        "root": ("root_final_update_cap", "root_max_update_cap"),
    }

    def _apply_group_updates(
        self,
        grad: torch.Tensor,
        active_indices: torch.Tensor,
        n_active: int,
        timestep: torch.Tensor,
        distance_weights: Dict[str, float],
    ):
        """Build the active-channel update with arm/torso/legs/root fully
        independent: each group is grad-clipped, scaled by guidance strength,
        its DDIM group-time-scale and distance weight, then per-frame capped by
        its own cap.  Root gradient is temporally smoothed first."""
        strength = float(self.config.guidance_strength)
        grad_clip = float(self.config.grad_clip_norm)
        membership = self._group_membership(active_indices)

        def clip_grad(g):
            if g.shape[-1] == 0:
                return g
            norm = torch.linalg.vector_norm(g, dim=-1, keepdim=True)
            return g * torch.clamp(grad_clip / norm.clamp_min(1e-8), max=1.0)

        def per_frame_max(x):
            if x.numel() == 0 or x.shape[-1] == 0:
                return 0.0
            return float(torch.linalg.vector_norm(x, dim=-1).max().detach().cpu())

        update = grad.new_zeros(grad.shape[0], grad.shape[1], n_active)
        norms = {}
        for group, sel in membership.items():
            group_norms = {f"{group}_raw_update_norm": 0.0, f"{group}_applied_update_norm": 0.0}
            if bool(sel.any()):
                g = grad[..., sel]
                if group == "root":
                    g = self._smooth_root_gradient(g, timestep)
                else:
                    # Temporally smooth the arm/torso/leg gradient so a large
                    # per-frame cap cannot make the joint jump frame-to-frame
                    # (post-contact hand jitter).
                    g = self._gaussian_time_smooth(
                        g, self.config.arm_gradient_smooth_kernel
                    )
                g = clip_grad(g)
                time_scale = self._group_time_scale(group, timestep)
                weight = float(distance_weights.get(group, 1.0))
                grp_update = strength * time_scale * weight * g
                group_norms[f"{group}_raw_update_norm"] = per_frame_max(grp_update)
                final_attr, max_attr = self._GROUP_CAPS[group]
                cap = self._scheduled_group_cap(
                    timestep, getattr(self.config, final_attr), getattr(self.config, max_attr)
                )
                gnorm = torch.linalg.vector_norm(grp_update, dim=-1, keepdim=True)
                grp_update = grp_update * torch.clamp(cap / gnorm.clamp_min(1e-8), max=1.0)
                group_norms[f"{group}_applied_update_norm"] = per_frame_max(grp_update)
                update[..., sel] = grp_update
            norms.update(group_norms)
        norms["raw_update_norm"] = max(
            norms.get(f"{g}_raw_update_norm", 0.0) for g in membership
        )
        norms["applied_update_norm"] = max(
            norms.get(f"{g}_applied_update_norm", 0.0) for g in membership
        )
        norms["scheduled_update_cap"] = float(
            self._scheduled_group_cap(
                timestep, self.config.arm_final_update_cap, self.config.arm_max_update_cap
            ).max().detach().cpu()
        )
        return update, norms

    def _apply_group_update(
        self,
        arm_grad: torch.Tensor,
        root_grad: torch.Tensor,
        arm_in_active: torch.Tensor,
        root_in_active: torch.Tensor,
        n_active: int,
        timestep: torch.Tensor,
    ):
        """Build a per-frame-capped active-channel update, arm and root fully
        independent.  Returns ``(update, norms)``."""
        strength = float(self.config.guidance_strength)
        grad_clip = float(self.config.grad_clip_norm)

        def clip_grad(g):
            if g.shape[-1] == 0:
                return g
            norm = torch.linalg.vector_norm(g, dim=-1, keepdim=True)
            return g * torch.clamp(grad_clip / norm.clamp_min(1e-8), max=1.0)

        def per_frame_max(x):
            if x.shape[-1] == 0:
                return 0.0
            return float(torch.linalg.vector_norm(x, dim=-1).max().detach().cpu())

        # Arm keeps a flat time scale (no fade); the smaller final cap keeps the
        # t=0 correction stable while the inner loop supplies displacement.
        arm_update = (strength * self._time_scale(timestep, root=False)) * clip_grad(arm_grad)
        root_update = strength * clip_grad(root_grad)  # root already time-scaled/smoothed
        arm_raw = per_frame_max(arm_update)
        root_raw = per_frame_max(root_update)
        if arm_update.shape[-1] > 0:
            arm_cap = self._scheduled_group_cap(
                timestep,
                self.config.arm_final_update_cap,
                self.config.arm_max_update_cap,
            )
            arm_norm = torch.linalg.vector_norm(arm_update, dim=-1, keepdim=True)
            arm_update = arm_update * torch.clamp(
                arm_cap / arm_norm.clamp_min(1e-8), max=1.0
            )
        if root_update.shape[-1] > 0:
            root_cap = self._scheduled_group_cap(
                timestep,
                self.config.root_final_update_cap,
                self.config.root_max_update_cap,
            )
            root_norm = torch.linalg.vector_norm(root_update, dim=-1, keepdim=True)
            root_update = root_update * torch.clamp(
                root_cap / root_norm.clamp_min(1e-8), max=1.0
            )
        arm_applied = per_frame_max(arm_update)
        root_applied = per_frame_max(root_update)
        batch, frames = arm_update.shape[0], arm_update.shape[1]
        update = arm_update.new_zeros(batch, frames, n_active)
        if arm_update.shape[-1] > 0:
            update[..., arm_in_active] = arm_update
        if root_update.shape[-1] > 0:
            update[..., root_in_active] = root_update
        scheduled_cap = float(
            self._scheduled_group_cap(
                timestep,
                self.config.arm_final_update_cap,
                self.config.arm_max_update_cap,
            ).max().detach().cpu()
        )
        norms = {
            "arm_raw_update_norm": arm_raw,
            "arm_applied_update_norm": arm_applied,
            "root_raw_update_norm": root_raw,
            "root_applied_update_norm": root_applied,
            "raw_update_norm": max(arm_raw, root_raw),
            "applied_update_norm": max(arm_applied, root_applied),
            "scheduled_update_cap": scheduled_cap,
        }
        return update, norms

    def _candidate_metrics(
        self,
        x0: torch.Tensor,
        base_x0: torch.Tensor,
        timestep: torch.Tensor,
        base_palm: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Re-decode ``x0`` (no grad) and report the accept-test metrics.

        ``max_palm_adjacent_jump_m`` is the per-frame palm displacement this
        candidate introduces relative to ``base_palm`` (the pre-step palm), not
        the motion's inherent frame-to-frame velocity, so a naturally fast
        reference motion is not rejected.
        """
        with torch.no_grad():
            total_loss_t, comps = self._compute_guidance_loss(x0, timestep)
            palm = comps["palm"]
            target_seq = comps["target_seq"]
            frame = int(self.config.contact_frame)
            per_frame = torch.linalg.vector_norm(palm - target_seq, dim=-1)
            contact_error_m = float(per_frame[:, frame].max().detach().cpu())
            post_contact_max_m = float(per_frame[:, frame:].max().detach().cpu())
            if base_palm is not None:
                palm_move = float(
                    torch.linalg.vector_norm(palm - base_palm, dim=-1)
                    .max()
                    .detach()
                    .cpu()
                )
            else:
                palm_move = 0.0
            root_delta = (x0 - base_x0)[..., ROOT_VELOCITY_SLICE]
            if root_delta.shape[-1] > 0 and root_delta.shape[1] > 0:
                root_step = float(
                    torch.linalg.vector_norm(root_delta, dim=-1).max().detach().cpu()
                )
            else:
                root_step = 0.0
            # v25b: max frame-to-frame palm step within the APPROACH window (up to
            # the contact frame).  Post-contact steps are excluded — there the hand
            # legitimately tracks the moving object.  Used as a hard line-search
            # limit so the reach is spread instead of lunging in one frame.
            transition = max(0, int(self.config.contact_transition_frames))
            a_start = max(0, frame - transition)
            a_hi = min(frame + 1, palm.shape[1])
            if a_hi - a_start >= 2:
                approach_step = torch.linalg.vector_norm(
                    palm[:, a_start + 1:a_hi] - palm[:, a_start:a_hi - 1], dim=-1
                )
                approach_max_step = float(approach_step.max().detach().cpu())
            else:
                approach_max_step = 0.0
        def _f(key):
            v = comps.get(key)
            return float(v.detach().cpu()) if torch.is_tensor(v) else float(v or 0.0)

        return {
            "total_loss": float(total_loss_t.detach().cpu()),
            "contact_loss": float(comps["contact_loss"].detach().cpu()),
            "contact_error_m": contact_error_m,
            "post_contact_max_m": post_contact_max_m,
            "post_contact_max_error_m": post_contact_max_m,
            "max_palm_adjacent_jump_m": palm_move,
            "max_palm_step_m": palm_move,
            "approach_max_step_m": approach_max_step,
            "root_step_m": root_step,
            "foot_slide_m": _f("foot_slide_m"),
            "root_target_loss": _f("root_target_loss"),
            "root_velocity_loss": _f("root_velocity_loss"),
            "arm_reference_loss": _f("arm_reference_loss"),
            "torso_reference_loss": _f("torso_reference_loss"),
            "support_foot_loss": _f("support_foot_loss"),
            "ground_contact_loss": _f("ground_contact_loss"),
            "temporal_loss": _f("temporal_loss"),
        }

    def _run_inner_guidance(self, x0_pred: torch.Tensor, timestep: torch.Tensor):
        """Multi-step guidance for one DDIM step: each inner iteration
        re-decodes the palm, recomputes the gradient, and (optionally) accepts
        the update only when it reduces the contact loss within safety limits.
        """
        original_dtype = x0_pred.dtype
        with torch.enable_grad():
            base_x0 = x0_pred.detach().float()
            mask = self._active_mask(base_x0.device, base_x0.dtype)
            active_indices = torch.nonzero(mask, as_tuple=False).flatten()
            n_active = int(active_indices.numel())
            current = base_x0.index_select(-1, active_indices).clone()

            line_search = bool(self.config.inner_line_search)
            palm_jump_limit = float(self.config.inner_palm_jump_limit)
            root_step_limit = float(self.config.inner_root_step_limit)
            foot_slide_limit = float(self.config.foot_slide_limit)
            reach_threshold = float(self.config.reach_error_threshold)
            n_inner = self._inner_step_schedule(timestep)

            with torch.no_grad():
                _, base_comps = self._compute_guidance_loss(
                    self._assemble(base_x0, active_indices, current), timestep
                )
                base_palm = base_comps["palm"].detach()
            before = self._candidate_metrics(
                self._assemble(base_x0, active_indices, current),
                base_x0,
                timestep,
                base_palm=base_palm,
            )
            last_metrics = before
            last_comps = None
            accepted_count = 0
            last_scale = 0.0
            inner_iterations = 0
            # §4: keep the candidate with the MINIMUM combined (contact +
            # post-contact) error seen across the inner loop, not the last one.
            best_current = current.clone()
            best_error = max(
                float(before["contact_error_m"]),
                float(before.get("post_contact_max_error_m", 0.0)),
            )
            best_contact_error = float(before["contact_error_m"])
            # §3: small-improvement patience for the convergence early-stop.
            conv_delta = float(self.config.inner_convergence_delta_m)
            conv_patience = max(1, int(self.config.inner_convergence_patience))
            small_improve_streak = 0
            grad_diag = {"gradient_norm": 0.0, "sequence_gradient_norm": 0.0,
                         "arm_gradient_norm": 0.0, "torso_gradient_norm": 0.0,
                         "root_gradient_norm": 0.0, "leg_gradient_norm": 0.0}
            norms_acc = {
                "arm_raw_update_norm": 0.0, "arm_applied_update_norm": 0.0,
                "torso_raw_update_norm": 0.0, "torso_applied_update_norm": 0.0,
                "root_raw_update_norm": 0.0, "root_applied_update_norm": 0.0,
                "legs_raw_update_norm": 0.0, "legs_applied_update_norm": 0.0,
                "raw_update_norm": 0.0, "applied_update_norm": 0.0,
                "scheduled_update_cap": 0.0,
            }
            membership = self._group_membership(active_indices)

            for _ in range(n_inner):
                inner_iterations += 1
                leaf = current.detach().requires_grad_(True)
                x0_full = self._assemble(base_x0, active_indices, leaf)
                total_loss, comps = self._compute_guidance_loss(x0_full, timestep)
                last_comps = comps
                grad = torch.autograd.grad(total_loss, leaf, retain_graph=False)[0]
                if not torch.isfinite(grad).all():
                    raise FloatingPointError(
                        "GENMO contact guidance gradient contains NaN/Inf"
                    )
                # Distance-dependent soft weights from the current contact error.
                distance_weights = self._distance_group_weights(
                    last_metrics["contact_error_m"]
                )
                update, norms = self._apply_group_updates(
                    grad, active_indices, n_active, timestep, distance_weights
                )
                # gradient diagnostics, per group
                grad_diag["sequence_gradient_norm"] = max(
                    grad_diag["sequence_gradient_norm"],
                    float(torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), dim=-1).max().detach().cpu()),
                )
                grad_diag["gradient_norm"] = max(
                    grad_diag["gradient_norm"],
                    float(torch.linalg.vector_norm(grad, dim=-1).max().detach().cpu()),
                )
                for group, sel in membership.items():
                    if bool(sel.any()):
                        key = "leg_gradient_norm" if group == "legs" else f"{group}_gradient_norm"
                        grad_diag[key] = max(
                            grad_diag[key],
                            float(torch.linalg.vector_norm(grad[..., sel], dim=-1).max().detach().cpu()),
                        )
                for key, value in norms.items():
                    norms_acc[key] = max(norms_acc.get(key, 0.0), value)

                # §10 accept on the FULL loss + safety limits, and §5: never let a
                # reference-loss reduction buy a contact-error increase — the
                # contact error may not grow beyond floating-point noise, so it
                # cannot accumulate across iterations.
                cur_total = float(total_loss.detach().cpu())
                cur_contact_error = float(last_metrics["contact_error_m"])
                contact_tol = 2e-4  # §5: ~0.2 mm, effectively non-increasing
                approach_step_limit = float(self.config.approach_max_step_m)
                cur_approach_step = float(last_metrics.get("approach_max_step_m", 0.0))
                accepted = False
                for scale in ([1.0, 0.5, 0.25, 0.125] if line_search else [1.0]):
                    candidate = current - scale * update
                    metrics = self._candidate_metrics(
                        self._assemble(base_x0, active_indices, candidate),
                        base_x0,
                        timestep,
                        base_palm=base_palm,
                    )
                    # v25b: reject a candidate that would lunge — the approach
                    # palm step may not exceed the cap, unless it does not make an
                    # already-over-cap step any worse (so a converged state is not
                    # frozen out).
                    approach_ok = (
                        approach_step_limit <= 0.0
                        or metrics["approach_max_step_m"] <= approach_step_limit
                        or metrics["approach_max_step_m"] <= cur_approach_step + 1e-6
                    )
                    if (not line_search) or (
                        metrics["total_loss"] < cur_total
                        and metrics["contact_error_m"] <= cur_contact_error + contact_tol
                        and metrics["max_palm_step_m"] <= palm_jump_limit
                        and metrics["root_step_m"] <= root_step_limit
                        and metrics["foot_slide_m"] <= foot_slide_limit
                        and approach_ok
                    ):
                        current = candidate
                        last_metrics = metrics
                        last_scale = scale
                        accepted_count += 1
                        accepted = True
                        break
                if not accepted:
                    break
                # §4/v25c: track the best candidate on a COMBINED error (contact
                # frame + post-contact tracking).  The line search already forbids
                # contact error from rising, so optimizing longer can only improve
                # the post-contact follow — the hand keeps up with the lifted
                # object instead of the loop quitting the moment the contact frame
                # alone hits 2 cm and abandoning the follow.
                cur_contact = float(last_metrics["contact_error_m"])
                cur_post = float(last_metrics.get("post_contact_max_error_m", 0.0))
                cur_err = max(cur_contact, cur_post)
                improvement = best_error - cur_err
                if cur_err <= best_error + 1e-9:
                    best_error = min(best_error, cur_err)
                    best_contact_error = cur_contact
                    best_current = current.clone()
                # §3: stop only when BOTH the contact frame and the post-contact
                # follow are within their reach targets, or the improvement stalls.
                post_reach = float(self.config.post_contact_reach_threshold)
                if cur_contact <= reach_threshold and (
                    post_reach <= 0.0 or cur_post <= post_reach
                ):
                    break
                small_improve_streak = (
                    small_improve_streak + 1 if improvement < conv_delta else 0
                )
                if small_improve_streak >= conv_patience:
                    break

            # §4: return the minimum-contact-error candidate, not the last one.
            current = best_current
            guided = self._assemble(base_x0, active_indices, current)
            if not torch.isfinite(guided).all():
                raise FloatingPointError("GENMO guided x0 contains NaN/Inf")

            comps = last_comps
            step_diag = {
                "timestep": int(timestep.reshape(-1)[0].detach().cpu()),
                "contact_loss": float(last_metrics.get("contact_loss", before["contact_loss"])),
                "reference_loss": float(comps["reference_loss"].detach().cpu()) if comps else 0.0,
                "temporal_loss": float(comps["temporal_loss"].detach().cpu()) if comps else 0.0,
                "post_contact_relative_velocity_loss": (
                    float(comps["relative_velocity_loss"].detach().cpu()) if comps else 0.0
                ),
                "pre_contact_position_loss": (
                    float(comps["pre_contact_position_loss"].detach().cpu()) if comps else 0.0
                ),
                "post_contact_position_loss": (
                    float(comps["post_contact_position_loss"].detach().cpu()) if comps else 0.0
                ),
                "post_contact_max_distance_m": (
                    float(comps["post_contact_max_distance"].detach().cpu()) if comps else 0.0
                ),
                "post_contact_hold_radius_m": float(self.config.post_contact_hold_radius),
                "contact_frame_position_weight": (
                    comps["contact_weight"] if comps else float(self.config.contact_frame_position_weight)
                ),
                "post_contact_worst_position_loss": (
                    float(comps["worst_position_loss"].detach().cpu()) if comps else 0.0
                ),
                "gradient_norm": grad_diag["gradient_norm"],
                "sequence_gradient_norm": grad_diag["sequence_gradient_norm"],
                "arm_gradient_norm": grad_diag["arm_gradient_norm"],
                "torso_gradient_norm": grad_diag["torso_gradient_norm"],
                "root_gradient_norm": grad_diag["root_gradient_norm"],
                "leg_gradient_norm": grad_diag["leg_gradient_norm"],
                "raw_update_norm": norms_acc["raw_update_norm"],
                "applied_update_norm": norms_acc["applied_update_norm"],
                "scheduled_update_cap": norms_acc["scheduled_update_cap"],
                "arm_raw_update_norm": norms_acc["arm_raw_update_norm"],
                "arm_applied_update_norm": norms_acc["arm_applied_update_norm"],
                "torso_raw_update_norm": norms_acc["torso_raw_update_norm"],
                "torso_applied_update_norm": norms_acc["torso_applied_update_norm"],
                "root_raw_update_norm": norms_acc["root_raw_update_norm"],
                "root_applied_update_norm": norms_acc["root_applied_update_norm"],
                "leg_raw_update_norm": norms_acc["legs_raw_update_norm"],
                "leg_applied_update_norm": norms_acc["legs_applied_update_norm"],
                "root_time_scale_last": float(
                    self._root_leg_time_scale(timestep).max().detach().cpu()
                ),
                "arm_reference_loss": float(comps["arm_reference_loss"].detach().cpu()) if comps else 0.0,
                "arm_smoothness_loss": float(comps["arm_smoothness_loss"].detach().cpu()) if comps else 0.0,
                "root_target_loss": float(comps["root_target_loss"].detach().cpu()) if comps else 0.0,
                "root_vertical_loss": float(comps["root_vertical_loss"].detach().cpu()) if comps else 0.0,
                "support_foot_loss": float(comps["support_foot_loss"].detach().cpu()) if comps else 0.0,
                "ground_contact_loss": float(comps["ground_contact_loss"].detach().cpu()) if comps else 0.0,
                "leg_reference_loss": float(comps["leg_reference_loss"].detach().cpu()) if comps else 0.0,
                "inner_iterations": inner_iterations,
                "inner_candidate_accepted": accepted_count,
                "inner_line_search_scale": float(last_scale),
                "inner_contact_error_before_m": float(before["contact_error_m"]),
                # §4: the returned candidate is the best (min combined-error) one.
                "inner_contact_error_after_m": float(best_contact_error),
                "inner_contact_error_last_m": float(last_metrics["contact_error_m"]),
            }
        return guided.detach().to(original_dtype), step_diag

    def __call__(self, x0_pred: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        guided, step_diag = self._run_inner_guidance(x0_pred, timestep)
        self.step_diagnostics.append(step_diag)
        return guided


def make_contact_guidance(
    spec: Optional[dict],
    kinematics_fn: Callable[[torch.Tensor, str], object],
) -> Optional[ContactGuidance]:
    if spec is None:
        return None
    return ContactGuidance(ContactGuidanceConfig(**spec), kinematics_fn)
