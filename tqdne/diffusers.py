from typing import List, Optional, Tuple, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, AudioPipelineOutput


def to_inputs(low_res, high_res):
    """Build Unet inputs from low and high resolution data."""
    return torch.cat((low_res, high_res), dim=1)


class DDPMPipeline1DCond(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet1DModel`]):
            A `UNet1DModel` to denoise the encoded audio latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.num_inference_steps = scheduler.config.num_train_timesteps
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        low_res: torch.Tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: Optional[int] = None,
        return_dict: bool = True,
    ) -> Union[AudioPipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            low_res (`torch.Tensor`):
                A `torch.Tensor` of shape `(batch_size, channels, timesteps)` containing the low resolution audio
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        device = self.unet.device
        low_res = low_res.to(device)
        batch_size, channels, t = low_res.shape
        assert self.unet.config.in_channels == 2 * channels
        # Sample gaussian noise to begin loop
        sig_shape = low_res.shape
        assert self.unet.config.extra_in_channels == 0

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            sig = randn_tensor(sig_shape, generator=generator)
            sig = sig.to(self.device)
        else:
            sig = randn_tensor(sig_shape, generator=generator, device=self.device)

        # set step values
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            inputs = to_inputs(low_res, sig)

            # 1. predict noise model_output
            model_output = self.unet(inputs, t).sample

            # 2. compute previous image: x_t -> x_t-1
            sig = self.scheduler.step(
                model_output, t, sig, generator=generator
            ).prev_sample

        if not return_dict:
            return (sig,)

        return AudioPipelineOutput(audios=sig)
