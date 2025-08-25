# ✨ Highlights
- Architecture (MLLM): CLIP-ViT multi-layer features → Q-Former–style Projector → Zephyr-7B-β backbone for text generation (answer + rationale). 
- Visual Prompting (alpha-blending): overlay boxes, masks, arrows, scribbles, etc., directly on the image to steer attention—simple, fast, and effective. 
- Any-Resolution (anyres) input: tile high-res crops + a low-res global image to preserve detail beyond a fixed 336×336 input. 
- Training setup (Stage-1): Zephyr-7B-β with CLIP ViT-L/14-336; anyres enabled; effective batch size 64 on 4× A100; max length 2048. 
- Component ablations: Multi-layer Feature Fusion reduces loss vs. single-layer selection; Q-Former projector converges faster and lower loss than 2-layer MLP.

# 🧠 Model Overview
Vision-Zephyr follows the decoder-only MLLM recipe: CLIP-ViT encodes the image (multi-layer features), a Q-Former–style projector maps vision tokens into the LLM token space, and Zephyr-7B-β performs autoregressive decoding to produce answers and rationales. 

# 🚀 Quickstart
**System requirements**:
	- GPU: NVIDIA A100 (hoặc GPU hỗ trợ bfloat16)
 	- VRAM: ≥ 40GB.

**Docker**
```bash
docker pull tyzen/viszephyr:cli
```
**Git Clone**
```bash
git clone https://github.com/baohuyvanba/Vision-Zephyr.git
cd Vision-Zephyr
```
**Installing Package**
```bash
pip install -e .
```
**Download Model's Checkpoints**
```bash
cd checkpoints/vis-zephyr-7b-v1-pretrain
wget -c https://huggingface.co/datasets/tyzen27/vcr_finetune/resolve/main/checkpoints/vis-zephyr-7b-v1-pretrain/mm_projector.bin
```

**CLI**
```bash
chmod +x script/cli.sh
script/cli.sh
```

# 📂 Data
- LAIONCCSBU BLIP-Caption Concept-balanced 558K (subset used during Stage-1). 
- VCR Validation for downstream evaluation (~23K items; 26,5K processed entries). 
- MMBench Validation for broader multimodal capabilities (optional).

# 🤝 Acknowledgements
Built on top of Zephyr-7B-β and CLIP ViT-L/14-336, with design inspirations from ViP-LLaVA and anyres tiling strategies.
