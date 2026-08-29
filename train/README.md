# Training CtrlRegen

CtrlRegen regenerates an image from noise under two conditions, and each one is a separately
trained module:

| Stage | Module | Script | What it controls |
|---|---|---|---|
| **A** | Semantic control adapter (*semantic-net*) | `train_semantic.py` | *what* the image depicts — DINOv2 features of the watermarked image, projected by a Resampler into IP cross-attention |
| **B** | Spatial control network (*spatial-net*) | `train_spatial.py` | *where* things are — Canny edges of the watermarked image |

Train **A first**, then freeze it and train **B** on top.

## Setup

```bash
cd train
pip install -r requirements-train.txt
```


## Data

Stage A reads a JSON list:

```json
[
    {"image_file": "/path/to/image.jpg", "text": ""},
    {"image_file": "/path/to/another.jpg", "text": ""}
]
```

**CtrlRegen does not use captions.** Every `text` field was empty when the released weights
were trained, and the text-drop rate is set to 0. Any collection of images works — you do not
need a captioned dataset. Build the index with:

```bash
python build_data_json.py --image_dir /path/to/images --output data.json
```

Stage B instead uses a Hugging Face dataset that provides image / Canny-edge / caption columns,
e.g. `AmritaBha/mscoco-controlnet-canny`. Captions are unused there too
(`--proportion_empty_prompts=1.0`).

## Stage A — Semantic control adapter

```bash
bash run_semantic.sh
```

Key arguments:

| Argument | Released-weights value | Notes |
|---|---|---|
| `--pretrained_model_name_or_path` | `SG161222/Realistic_Vision_V4.0_noVAE` | frozen base model |
| `--image_encoder_path` | `facebook/dinov2-giant` | hidden size 1536, `hidden_states[-2]` is used |
| `--vae_path` | `stabilityai/sd-vae-ft-mse` | **not** the base model's own VAE; matches inference |
| `--num_tokens` | `16` | Resampler queries |
| `--learning_rate` | `1e-4` | |
| `--train_batch_size` | `8` (x 8 GPUs) | |
| `--resolution` | `512` | |
| `--shuffle` | *off* | the released weights were trained without shuffling |

To resume after a job time limit, convert the latest checkpoint and pass it via
`--pretrained_ip_adapter_path`.

## Stage B — Spatial control network

```bash
bash run_spatial.sh    # needs a converted Stage A checkpoint
```

Stage B loads the Stage A weights through `--ip_checkpoint_path`, freezes them, and trains only
the spatial control network. Note that the spatial-net itself receives **text only**; the semantic
tokens are concatenated into the UNet's cross-attention input.


## Converting checkpoints

```bash
# Stage A: 3.6 GB accelerate state -> 197 MB inference weights
python convert_checkpoint.py --stage semantic \
    --input  out_semantic/checkpoint-100000/pytorch_model.bin \
    --output ../semanticnet_ckp/models/semantic_control_ckp_100000.bin

# Stage B: collect the spatial-net (diffusers ControlNetModel) directory
python convert_checkpoint.py --stage spatial \
    --input  out_spatial/checkpoint-14000/controlnet \
    --output ../spatialnet_ckp/spatial_control_ckp_14000
```

The outputs match the layout `ctrlregen_plus_demo.ipynb` expects, so you can point the notebook
at your own weights immediately. For reference, the released checkpoints
([huggingface.co/yepengliu/ctrlregen](https://huggingface.co/yepengliu/ctrlregen)) are the same
two artifacts:


## Training configuration of the released weights

| | Stage A | Stage B |
|---|---|---|
| Data | ~6.81 M images (COYO) + ~5.00 M images (LAION), no captions | `AmritaBha/mscoco-controlnet-canny` |
| Steps | 435,000 | 14,788 (released checkpoint: 14,000) |
| Learning rate | 1e-4 | 1e-5 |
| Batch size | 8 per GPU x 8 GPUs | 4 per GPU x 8 GPUs |
| Resolution | 512 | 512 |
| Precision | fp16 | fp16 |
| Hardware | 8 x A100 80GB | 8 x A100 80GB |

