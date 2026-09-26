"""Experience Replay: a reservoir buffer of past inputs mixed into each batch."""
from __future__ import annotations

from typing import Any, Dict

from methods.base import ContinualMethod
from methods.replay_buffer import ReplayBuffer, replay_ce_loss


class Replay(ContinualMethod):
    name = "replay"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        self.buf = ReplayBuffer(int(self.cfg.get("buffer_size", 500)))
        self.replay_n = int(self.cfg.get("batch_size", 32))
        self.replay_loss_coefficient = float(self.cfg.get("replay_loss_coefficient", 1.0))
        self._replay_input_transform = None
        self._device = None

    def set_replay_input_transform(self, transform) -> None:
        """Install an R1-only training view for canonical buffer samples."""
        self._replay_input_transform = transform

    def on_task_start(self, task_id, model):
        self._device = next(model.parameters()).device

    def extra_loss(self, model, x, y, task_id):
        if len(self.buf) == 0:
            return x.new_zeros(())
        replay = replay_ce_loss(
            model,
            self.buf.sample(self.replay_n),
            self._device,
            input_transform=self._replay_input_transform,
        )
        return self.replay_loss_coefficient * replay

    def on_batch_end(self, model, x, y, task_id):
        # admit everything (standard ER); pass the true label only for the
        # purity audit -- the stored/rehearsed label is still the (noisy) y.
        self.buf.add(x, y, task_id, y_clean=self._batch_y_clean)

    def consolidation_state(self, model) -> Dict[str, Any]:
        return {
            "buffer_size": len(self.buf),
            "buffer_purity": self.buf.purity(),
            "replay_loss_coefficient": self.replay_loss_coefficient,
        }
