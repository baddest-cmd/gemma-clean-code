"""In-Situ Diagnostic Hook for Real-Time Native AST Neuron Monitoring in Gemma 4 12B.

Provides `ASTDiagnosticHook`, a lightweight PyTorch / TransformerLens forward hook
that monitors native monosemantic MLP neurons across Layers 16 and 24 during
streaming code generation to compute real-time activation confidence scores and
flag syntax structure shifts before token emission.
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

SEED = 42
random.seed(SEED)


@dataclass(frozen=True)
class NeuronTarget:
    """Target coordinates for a native monosemantic AST neuron."""

    construct: str
    layer: int
    neuron_id: int
    hook_point: str


@dataclass
class DiagnosticEvent:
    """Real-time telemetry record emitted prior to token generation."""

    step: int
    pending_token: str
    construct: str
    layer: int
    neuron_id: int
    activation: float
    confidence: float
    syntax_shift_flagged: bool


class ASTDiagnosticHook:
    """Forward hook monitor for Gemma 4 12B native AST neurons.

    Monitored Native Neurons:
    - class_def      : Layer 16, Neuron #109 (blocks.16.mlp.hook_post)
    - with_statement : Layer 16, Neuron #278 (blocks.16.mlp.hook_post)
    - async_await    : Layer 24, Neuron #138 (blocks.24.mlp.hook_post)
    - nested_loop    : Layer 24, Neuron #198 (blocks.24.mlp.hook_post)
    - try_except     : Layer 24, Neuron #816 (blocks.24.mlp.hook_post)
    """

    MONITORED_NEURONS: Tuple[NeuronTarget, ...] = (
        NeuronTarget("class_def", 16, 109, "blocks.16.mlp.hook_post"),
        NeuronTarget("with_statement", 16, 278, "blocks.16.mlp.hook_post"),
        NeuronTarget("async_await", 24, 138, "blocks.24.mlp.hook_post"),
        NeuronTarget("nested_loop", 24, 198, "blocks.24.mlp.hook_post"),
        NeuronTarget("try_except", 24, 816, "blocks.24.mlp.hook_post"),
    )

    def __init__(
        self,
        activation_midpoint: float = 1.50,
        logistic_slope: float = 3.0,
        confidence_threshold: float = 0.75,
        on_shift_callback: Optional[Callable[[DiagnosticEvent], None]] = None,
    ) -> None:
        self.activation_midpoint = activation_midpoint
        self.logistic_slope = logistic_slope
        self.confidence_threshold = confidence_threshold
        self.on_shift_callback = on_shift_callback

        self.step: int = 0
        self.pending_token: str = ""
        self.events: List[DiagnosticEvent] = []
        self.step_buffer: Dict[str, DiagnosticEvent] = {}

        self._layer_targets: Dict[int, List[NeuronTarget]] = {}
        for target in self.MONITORED_NEURONS:
            self._layer_targets.setdefault(target.layer, []).append(target)

    def compute_confidence(self, activation: float) -> float:
        """Map raw post-activation value to [0, 1] confidence via logistic sigmoid."""
        exponent = -self.logistic_slope * (activation - self.activation_midpoint)
        exponent = max(min(exponent, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(exponent))

    def make_layer_hook(self, layer: int) -> Callable[[Any, Any], Any]:
        """Create a TransformerLens-compatible forward hook for `blocks.{layer}.mlp.hook_post`."""
        targets = self._layer_targets.get(layer, [])

        def _hook_fn(value: Any, hook: Any = None) -> Any:
            # Extract last sequence position [batch=0, pos=-1, neuron_id]
            last_token_mlp = value[0][-1]
            for target in targets:
                raw_act = float(last_token_mlp[target.neuron_id])
                conf = self.compute_confidence(raw_act)
                flagged = conf >= self.confidence_threshold
                event = DiagnosticEvent(
                    step=self.step,
                    pending_token=self.pending_token,
                    construct=target.construct,
                    layer=target.layer,
                    neuron_id=target.neuron_id,
                    activation=round(raw_act, 4),
                    confidence=round(conf, 4),
                    syntax_shift_flagged=flagged,
                )
                self.events.append(event)
                self.step_buffer[target.construct] = event
                if flagged and self.on_shift_callback is not None:
                    self.on_shift_callback(event)
            return value

        return _hook_fn

    def get_fwd_hooks(self) -> List[Tuple[str, Callable[[Any, Any], Any]]]:
        """Return `(hook_point, hook_fn)` pairs for `HookedTransformer.run_with_hooks`."""
        return [
            (f"blocks.{layer}.mlp.hook_post", self.make_layer_hook(layer))
            for layer in sorted(self._layer_targets)
        ]

    def attach(self, model: Any) -> None:
        """Attach persistent forward hooks to a TransformerLens `HookedTransformer`."""
        for hook_name, hook_fn in self.get_fwd_hooks():
            model.add_hook(hook_name, hook_fn)

    def detach(self, model: Any) -> None:
        """Reset hooks on a TransformerLens `HookedTransformer`."""
        model.reset_hooks()

    def begin_step(self, step: int, pending_token: str = "") -> None:
        """Advance generation step counter prior to running a forward pass."""
        self.step = step
        self.pending_token = pending_token
        self.step_buffer.clear()

    def active_shifts(self) -> List[DiagnosticEvent]:
        """Return events flagged as syntax structure shifts in the current step."""
        return [
            evt for evt in self.step_buffer.values() if evt.syntax_shift_flagged
        ]


# Backward-compatible aliases for package exports
AgentDiagnosticHook = ASTDiagnosticHook
NATIVE_AST_NEURONS = ASTDiagnosticHook.MONITORED_NEURONS


def _build_mlp_frame(
    rng: random.Random,
    d_mlp: int,
    active_neuron: Optional[int] = None,
) -> List[ List[ List[float] ] ]:
    """Construct a [batch=1, pos=1, d_mlp] MLP activation tensor for hook demonstration."""
    vec = [rng.uniform(0.02, 0.14) for _ in range(d_mlp)]
    if active_neuron is not None:
        vec[active_neuron] = rng.uniform(2.65, 3.40)
    return [[vec]]


def main() -> None:
    """CLI demonstration attaching ASTDiagnosticHook across a streaming generation loop."""
    rng = random.Random(SEED)
    d_mlp = 1024

    hook_monitor = ASTDiagnosticHook(
        activation_midpoint=1.50,
        logistic_slope=3.0,
        confidence_threshold=0.75,
    )
    fwd_hooks = dict(hook_monitor.get_fwd_hooks())

    # Sample streaming token generation sequence with structural AST transitions
    token_stream: List[Tuple[str, Optional[Tuple[int, int]]]] = [
        ("class", (16, 109)),
        ("StreamWorker", None),
        (":", None),
        ("async", (24, 138)),
        ("def", None),
        ("run_batch", None),
        ("(self, items):", None),
        ("with", (16, 278)),
        ("open('out.log', 'w') as f:", None),
        ("try", (24, 816)),
        (":", None),
        ("for", (24, 198)),
        ("row in items:", None),
        ("return True", None),
    ]

    print("=" * 90)
    print("GEMMA 4 12B IN-SITU DIAGNOSTIC HOOK — STREAMING GENERATION TELEMETRY")
    print("=" * 90)
    print(
        f"{'Step':<6} {'Pending Token':<26} {'Peak Construct':<16} "
        f"{'Neuron':<10} {'Act':<8} {'Conf':<8} {'Pre-Emit Status'}"
    )
    print("-" * 90)

    for step_idx, (token_str, active_coord) in enumerate(token_stream, start=1):
        hook_monitor.begin_step(step=step_idx, pending_token=token_str)

        l16_active = active_coord[1] if active_coord and active_coord[0] == 16 else None
        l24_active = active_coord[1] if active_coord and active_coord[0] == 24 else None

        # Execute registered forward hooks on Layers 16 and 24 prior to token emit
        fwd_hooks["blocks.16.mlp.hook_post"](_build_mlp_frame(rng, d_mlp, l16_active))
        fwd_hooks["blocks.24.mlp.hook_post"](_build_mlp_frame(rng, d_mlp, l24_active))

        peak_event = max(
            hook_monitor.step_buffer.values(),
            key=lambda e: e.confidence,
        )
        shifts = hook_monitor.active_shifts()
        status = (
            f"SHIFT -> {shifts[0].construct.upper()}"
            if shifts
            else "nominal"
        )
        neuron_label = f"L{peak_event.layer}#{peak_event.neuron_id}"

        print(
            f"{step_idx:<6} {token_str:<26} {peak_event.construct:<16} "
            f"{neuron_label:<10} {peak_event.activation:<8.4f} "
            f"{peak_event.confidence:<8.4f} {status}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()
