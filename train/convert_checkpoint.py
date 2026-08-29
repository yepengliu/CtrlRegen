"""
Convert an accelerate training checkpoint into the weight format used by CtrlRegen inference.

Stage A (semantic)
    `accelerator.save_state` writes a single ~3.6 GB `pytorch_model.bin` that also contains
    the frozen UNet. Inference expects a ~197 MB file holding only the two trained parts:
        {"image_proj": <Resampler weights>, "ip_adapter": <to_k_ip / to_v_ip weights>}

Stage B (spatial)
    The ControlNet is already saved in diffusers format. At the end of training it is written
    to the top level of --output_dir; intermediate checkpoints keep it under
    `checkpoint-N/controlnet/`. This command just collects that directory.

Examples
    python convert_checkpoint.py --stage semantic \
        --input  out_semantic/checkpoint-100000/pytorch_model.bin \
        --output semanticnet_ckp/models/semantic_control_ckp_100000.bin

    python convert_checkpoint.py --stage spatial \
        --input  out_spatial/checkpoint-14000/controlnet \
        --output spatialnet_ckp/spatial_control_ckp_14000
"""
import argparse
import os
import shutil

import torch


def convert_semantic(input_path: str, output_path: str) -> None:
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, "pytorch_model.bin")
    print(f"Loading {input_path} ...", flush=True)
    state_dict = torch.load(input_path, map_location="cpu")

    image_proj, ip_adapter = {}, {}
    for key, value in state_dict.items():
        if key.startswith("unet"):
            continue  # frozen, not part of the released weights
        if key.startswith("image_proj_model."):
            image_proj[key[len("image_proj_model."):]] = value
        elif key.startswith("adapter_modules."):
            ip_adapter[key[len("adapter_modules."):]] = value

    if not image_proj or not ip_adapter:
        raise RuntimeError(
            f"Nothing to convert: found {len(image_proj)} image_proj and {len(ip_adapter)} "
            f"ip_adapter tensors. Is {input_path} a Stage A checkpoint?"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save({"image_proj": image_proj, "ip_adapter": ip_adapter}, output_path)
    print(
        f"Wrote {output_path}\n"
        f"  image_proj : {len(image_proj)} tensors\n"
        f"  ip_adapter : {len(ip_adapter)} tensors\n"
        f"  size       : {os.path.getsize(output_path):,} bytes"
    )


def convert_spatial(input_path: str, output_path: str) -> None:
    # Accept either `checkpoint-N` or `checkpoint-N/controlnet` or the output_dir top level.
    candidates = [
        input_path,
        os.path.join(input_path, "controlnet"),
    ]
    src = next(
        (c for c in candidates
         if os.path.isfile(os.path.join(c, "diffusion_pytorch_model.safetensors"))),
        None,
    )
    if src is None:
        raise RuntimeError(
            f"No diffusion_pytorch_model.safetensors under {input_path} "
            f"(tried {', '.join(candidates)})"
        )

    os.makedirs(output_path, exist_ok=True)
    for name in ("config.json", "diffusion_pytorch_model.safetensors"):
        source = os.path.join(src, name)
        if not os.path.isfile(source):
            raise RuntimeError(f"Missing {name} in {src}")
        shutil.copy2(source, os.path.join(output_path, name))
    weights = os.path.join(output_path, "diffusion_pytorch_model.safetensors")
    print(f"Wrote {output_path}/\n  size: {os.path.getsize(weights):,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["semantic", "spatial"])
    parser.add_argument("--input", required=True,
                        help="Stage A: pytorch_model.bin (or the checkpoint dir). "
                             "Stage B: the controlnet dir (or the checkpoint dir).")
    parser.add_argument("--output", required=True,
                        help="Stage A: destination .bin file. Stage B: destination directory.")
    args = parser.parse_args()

    if args.stage == "semantic":
        convert_semantic(args.input, args.output)
    else:
        convert_spatial(args.input, args.output)


if __name__ == "__main__":
    main()
