import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
# from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
# from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder #, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler

from transformers.utils import ContextManagers
from data.thriple_image import EightAnchorImageDataset
# from models.unet_mv2d_condition import UNetMV2DConditionModel
from typing import List
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
# from pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from pipelines.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from collections import defaultdict
import os
import PIL.Image
from PIL import Image
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel#, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available #, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from utils.rmbg import BackgroundRemoval
@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path:str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation

    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool

def log_validation(dataloader, pipeline, cfg, weight_dtype, name, save_dir):

    def save_image_numpy(ndarr, fp):
        im = Image.fromarray(ndarr)
        im.save(fp)

    def save_image(tensor, fp):
        ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        # pdb.set_trace()
        im = Image.fromarray(ndarr)
        im.save(fp)
        return ndarr

    pipeline.set_progress_bar_config(disable=True)

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.device).manual_seed(cfg.seed)

    images_cond, images_pred = [], defaultdict(list)

    remove = BackgroundRemoval()

    for i, batch in tqdm(enumerate(dataloader)):

        with torch.autocast("cuda"):
            out = pipeline(
                image=batch["imgs_in"][0], first_image=batch["imgs_in"][1], last_image=batch["imgs_in"][2] ,generator=generator, output_type='pt',
            ).frames
            # logger.info(out.shape)
            bsz=len(out)
            num_frames = 9
            cur_dir = os.path.join(save_dir, f'{name}_val_out', f"{cfg.validation_dataset.scene}","s2_masked_rgb")
            rgb_dir = os.path.join(save_dir, f'{name}_val_out', f"{cfg.validation_dataset.scene}")
            os.makedirs(cur_dir, exist_ok=True)
            os.makedirs(rgb_dir, exist_ok=True)
            for b in range(bsz):
                for j in range(num_frames):
                    if j not in [0,4,8]:
                        idx = int((batch["seq"][0]*4 + j) % 32)
                        color = out[b][j]
                        rgb_filename = f"rgb{idx}.png"
                        color = save_image(color, os.path.join(rgb_dir, rgb_filename))
                        masked_color = remove(color)
                        masked_color = Image.fromarray(masked_color)
                        masked_color.save(os.path.join(cur_dir, rgb_filename))

def load_envision3d_pipeline(cfg):
    # feature_extractor = CLIPImageProcessor.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
    #                                                            subfolder="feature_extractor")
    #
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    #         "stabilityai/stable-video-diffusion-img2vid-xt", subfolder="image_encoder"
    #     )
    # vae = AutoencoderKLTemporalDecoder.from_pretrained(
    #         "stabilityai/stable-video-diffusion-img2vid-xt", subfolder="vae",
    #     )
    #
    # unet = UNetSpatioTemporalConditionModel.from_pretrained(
    #     cfg.pretrained_model_name_or_path, subfolder="unet",
    #     low_cpu_mem_usage=False,
    #     # sample_size=32, cd_attention_mid=True, low_cpu_mem_usage=False
    # )

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        safety_checker=None,
        torch_dtype=weight_dtype
    )

    # pipeline.to('cuda:0')
    pipeline.unet.enable_xformers_memory_efficient_attention()


    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    # sys.main_lock = threading.Lock()
    return pipeline


def main(
        cfg: TestConfig
):
    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    pipeline = load_envision3d_pipeline(cfg)

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipeline.unet.enable_xformers_memory_efficient_attention()
            print("use xformers.")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the  dataset
    validation_dataset = EightAnchorImageDataset(**cfg.validation_dataset)

    # DataLoaders creation:
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )

    os.makedirs(cfg.save_dir, exist_ok=True)

    log_validation(
            validation_dataloader,
            pipeline,
            cfg,
            weight_dtype,
            's2',
            cfg.save_dir)



if __name__ == '__main__':
    weight_dtype = torch.float16
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()

    from utils.misc import load_config
    from omegaconf import OmegaConf

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    print(cfg)
    schema = OmegaConf.structured(TestConfig)
    # cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)

    main(cfg)