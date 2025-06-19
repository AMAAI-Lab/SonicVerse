<div align="center">

# **SonicVerse: Multi-Task Learning for Music Feature-Informed Captioning**
[![arXiv](https://img.shields.io/badge/arXiv-2506.15154-b31b1b.svg)](https://arxiv.org/abs/2506.15154)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/amaai-lab/SonicVerse)
[![Demo](https://img.shields.io/badge/🎵-Demo-green)](https://huggingface.co/spaces/amaai-lab/SonicVerse)
[![Samples Page](https://img.shields.io/badge/Samples-Page-blue)](https://amaai-lab.github.io/SonicVerse/)

</div>

## Overview

SonicVerse is a multi-task music captioning model that integrates caption generation with auxiliary music feature detection tasks such as key detection, vocals detection, and more. The model directly captures both low-level acoustic details as well as high-level musical attributes through a novel projection-based architecture that transforms audio input into natural language captions while simultaneously detecting music features through dedicated auxiliary heads. Additionally, SonicVerse enables the generation of temporally informed long captions for extended music pieces by chaining outputs from short segments using large language models, providing detailed time-informed descriptions that capture the evolving musical narrative.

<div align="center">
<img src="music_captioning_overview_w_chaining-1.png" alt="SonicVerse Architecture" width="800"/>
<p><em>Figure 1: SonicVerse architecture for music captioning with feature detection.</em></p>
</div>

🔥 Live demo available on [Huggingface](https://huggingface.co/spaces/amaai-lab/SonicVerse)

## Key Features

- **Multi-Task Learning**: Combines caption generation with music feature detection (key detection, vocals detection, etc.)
- **Projection-Based Architecture**: Transforms audio input into language tokens while maintaining feature detection capabilities
- **Enhanced Captioning**: Produces rich, descriptive captions that incorporate detected music features
- **Long-Form Description**: Enables detailed time-informed descriptions for longer music pieces through LLM chaining

## Installation

```bash
git clone https://github.com/AMAAI-Lab/SonicVerse.git
cd SonicVerse
pip install -r requirements.txt
pip install -e .
```

### Quick App

```bash
python scripts/app.py
```

## Training
### Data Prcoessing
### Finetuning


## Citation

If you use SonicVerse in your research, please cite our paper:

```bibtex
@article{chopra2025sonicverse,
  title={SonicVerse: Multi-Task Learning for Music Feature-Informed Captioning},
  author={Chopra, Anuradha and Roy, Abhinaba and Herremans, Dorien},
  journal={Proceedings of the 6th Conference on AI Music Creativity (AIMC 2025)},
  year={2025},
  address={Brussels, Belgium},
  month={September},
  pages={10--12}
}
```

## Built With

- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Gradio](https://gradio.app/)
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [MERT 95M](https://huggingface.co/m-a-p/MERT-v1-95M)


<div align="center">
Made with ❤️ by the AMAAI Lab | Singapore
</div>
