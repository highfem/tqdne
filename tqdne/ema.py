from collections import OrderedDict
from copy import deepcopy

import pytorch_lightning as pl
import torch as th


class EMA(pl.Callback):

    def __init__(self, decay: float = 0.999):
        self.decay = decay

    def on_fit_start(self, trainer, pl_module):
        self.training_state = OrderedDict(
            (name, p) for name, p in pl_module.named_parameters() if p.requires_grad
        )
        if not hasattr(self, "ema_state"):
            self.ema_state = OrderedDict(
                (name, p.detach().clone()) for name, p in self.training_state.items()
            )
        else:
            for k, v in self.ema_state.items():
                self.ema_state[k] = v.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        # ema_param = decay * ema_param + (1 - decay) * param
        th._foreach_lerp_(
            tuple(self.ema_state.values()), tuple(self.training_state.values()), 1 - self.decay
        )

    def on_validation_start(self, trainer, pl_module):
        self._original_state_dict = deepcopy(pl_module.state_dict())
        pl_module.load_state_dict(self.ema_state, strict=False)

    def on_validation_end(self, trainer, pl_module):
        pl_module.load_state_dict(self._original_state_dict)
        del self._original_state_dict

    def on_test_start(self, trainer, pl_module):
        return self.on_validation_start(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        return self.on_validation_end(trainer, pl_module)

    def on_predict_start(self, trainer, pl_module):
        return self.on_validation_start(trainer, pl_module)

    def on_predict_end(self, trainer, pl_module):
        return self.on_validation_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema_state"] = self.ema_state

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.ema_state = callback_state["ema_state"]
