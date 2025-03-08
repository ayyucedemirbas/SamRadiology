# Sam2Rad: A Segmentation Model for Medical Images with Learnable Prompts

Requirements

1.	Clone the Repository

```bash
git clone https://github.com/aswahd/SamRadiology.git
cd sam2rad
```

2.	Set Up a Virtual Environment
It’s recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3.	Install Dependencies

```bash
pip install -r requirements.txt
```


4.	Download Pre-trained Weights

Download the pre-trained weights from the following links:

- [Pre-trained weights for SAM2-Hiera-Tiny](https://huggingface.co/facebook/sam2-hiera-tiny/blob/f245b47be73d8858fb7543a8b9c1c720d9f98779/sam2_hiera_tiny.pt)
- [Trained model](https://huggingface.co/ayyuce/sam2rad)
  
Place the downloaded weights in the `weights` directory.


## Quickstart

File structure:
```markdown
root
├── Train
│   ├── imgs
            ├── 1.png
            ├── 2.png
            ├── ...
            |
│   └── gts
            ├── 1.png
            ├── 2.png
            ├── ...
└── Test
    ├── imgs
            ├── 1.png
            ├── 2.png
            ├── ...
    └── gts
            ├── 1.png
            ├── 2.png
            ├── ...
```


Download Sample Dataset:
- Download the preprocessed data from [ACDC dataset](https://drive.google.com/drive/folders/14WIOWTF1WWwMaHV7UVo5rjWujpUxGetJ?usp=sharing).
- Extract the data to `./datasets/ACDCPreprocessed`.



## Models

Sam2Rad supports various image encoders and mask decoders, allowing flexibility in model architecture.

**Supported Image Encoders**
-	sam_vit_b_adapter
-	sam_vit_l_adapter
-	sam_vit_h_adapter
-	sam_vit_b
-	sam_vit_l
-	sam_vit_h
-	vit_tiny
-	All versions of Sam2 image encoder with or without adapters

All supported image encoders are available in the [sam2rad/encoders/build_encoder.py](sam2rad/encoders/build_encoder.py).

**Supported Mask Decoders**

-	sam_mask_decoder
-	lora_mask_decoder
-	All versions of Sam2 mask decoder


All supported mask decoders are available in the [sam2rad/decoders/build_decoder.py](sam2rad/decoders/build_decoder.py).

## Training
Prepare a configuration file for training. Here is an example configuration file for training on the ACDC dataset:

```yaml
image_size: 1024
image_encoder: "sam2_tiny_hiera_adapter"
mask_decoder: "sam2_lora_mask_decoder"
sam_checkpoint: "weights/sam2_hiera_tiny.pt"
wandb_project_name: "ACDC"

dataset:
  name: acdc
  root: /path/to/your/dataset
  image_size: 1024
  split: 0.0526 # 0.0263 # training split
  seed: 42
  batch_size: 4
  num_workers: 4
  num_classes: 3
  num_tokens: 10

training:
  max_epochs: 200
  save_path: checkpoints/ACDC

inference:
  name: acdc_test
  root: /path/to/your/test_data
  checkpoint_path: /path/to/your/checkpoint 
```


```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python train.py --config /path/to/your/config.yaml

```
Replace `/path/to/your/config.yaml` with the actual path to your configuration file.

## Evaluation

Ensure your configuration file points to the correct checkpoint and data paths:

```yaml
inference:
  model_checkpoint: checkpoints/your_model_checkpoint
  input_images: /path/to/your/test_images
  output_dir: /path/to/save/segmentation_results
  image_size: 1024
```
Run the evaluation script:
```bash
python -m sam2rad.evaluation.eval_bounding_box --config /path/to/your/config.yaml
python -m sam2rad.evaluation.eval_prompt_learner --config /path/to/your/config.yaml
```

## Citation

If you use Sam2Rad in your research, please consider citing our paper:

```bibtex
@article{wahd2024sam2radsegmentationmodelmedical,
  title={Sam2Rad: A Segmentation Model for Medical Images with Learnable Prompts},
  author={Assefa Seyoum Wahd and Banafshe Felfeliyan and Yuyue Zhou and Shrimanti Ghosh and Adam McArthur and Jiechen Zhang and Jacob L. Jaremko and Abhilash Hareendranathan},
  year={2024},
  eprint={2409.06821},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2409.06821},
}
```
