
import os

import fire
import torch

from pathlib import Path
from typing import List
import re
import webdataset as wds
from safetensors.torch import load as load_safetensors

from sd3_infer import *


# TODO clean up or delete this file, just used for one off evals

def get_wds_file_list(input_dataset: str) -> List[str] | str:
    """
    Get a WebDataset object from a tarfile or directory.
    If a directory is passed, all tarfiles in the directory are used.
    """
    if re.search(r"[{}]", input_dataset):
        return input_dataset
    if input_dataset.endswith(".tar"):
        return [input_dataset]
    else:
        path = Path(input_dataset)
        all_files_in_dataset: List[str] = []
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".tar":
                all_files_in_dataset.append(str(file_path))
        all_files_in_dataset = [
            os.path.join(input_dataset, file)
            for file in all_files_in_dataset
            if file.endswith(".tar")
        ]
    if len(all_files_in_dataset) == 0:
        raise FileNotFoundError(f"No tarfiles found in {input_dataset}")
    return all_files_in_dataset


def sft_handler(value):
    return load_safetensors(value)


import os
@torch.no_grad()
def main(
    input_dataset = "/weka/home-brianf/controlnet_eval/canny_eval_dataset/",
    model=MODEL,
    out_dir="outputs/sd3_2b_canny_eval",
    seed=42,
    sampler=None,
    steps=None,
    cfg=None,
    shift=None,
    width=WIDTH,
    height=HEIGHT,
    controlnet_ckpt=None,
    vae=VAEFile,
    init_image=INIT_IMAGE,
    denoise=DENOISE,
    verbose=False,
    model_folder=MODEL_FOLDER,
    text_encoder_device="cuda",
    **kwargs,
):
    assert not kwargs, f"Unknown arguments: {kwargs}"
    steps = steps or CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {}).get(
        "steps", 50
    )
    cfg = cfg or CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {}).get(
        "cfg", 5
    )
    shift = shift or CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {}).get(
        "shift", 3
    )
    sampler = sampler or CONFIGS.get(
        os.path.splitext(os.path.basename(model))[0], {}
    ).get("sampler", "dpmpp_2m")

    inferencer = SD3Inferencer()
    
    wds_file_list = get_wds_file_list(input_dataset)
    dataset = wds.WebDataset(wds_file_list).decode(wds.handle_extension("sft", sft_handler))

    inferencer.load(
        model, vae, shift, controlnet_ckpt, model_folder, text_encoder_device, load_tokenizers=True
    )

    print(f"Saving images to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    def _get_precomputed_cond(sample):
        l_out, l_pooled = sample['clip_openai_l.sft']['seq'], sample['clip_openai_l.sft']['pooled']
        g_out, g_pooled = sample['clip_open_bigg.sft']['seq'], sample['clip_open_bigg.sft']['pooled']
        t5_out = sample['t5.sft']['seq']
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )


    for i, sample in tqdm(enumerate(dataset)):
        controlnet_cond = None
        if init_image:
            latent = inferencer._image_to_latent(init_image, width, height)
        else:
            latent = inferencer.get_empty_latent(1, width, height, seed, "cpu")
            latent = latent.cuda()
        controlnet_cond = inferencer.vae_encode_tensor(
            sample["vae_f8_ch16.cond.sft"]["latent"]
        )
        conditioning = inferencer.get_cond(sample["txt"])
        seed_num = 42
        neg_cond = inferencer.get_cond("")
        sampled_latent = inferencer.do_sampling(
            latent,
            seed_num,
            conditioning,
            neg_cond,
            steps,
            cfg,
            sampler,
            controlnet_cond,
            denoise if init_image else 1.0,
        )
        image = inferencer.vae_decode(sampled_latent)
        k = sample["__key__"]
        save_path = os.path.join(out_dir, f"{k}.png")
        image.save(save_path)


fire.Fire(main)
