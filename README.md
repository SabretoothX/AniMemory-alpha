![img](gallery_demo.png)

<div align="left">
  <a href='https://huggingface.co/animEEEmpire/AniMemory-alpha'><img src='https://img.shields.io/badge/Hugging%20Face-Model-yellow'></a> &ensp;
  <a href="https://github.com/animEEEmpire/AniMemory-alpha"><img src="https://img.shields.io/badge/Github-Code-blue"></a> &ensp;
</div>

Animemory Alpha is a bilingual model primarily focused on anime-style image generation. It utilizes a SDXL-type Unet
structure and a self-developed bilingual T5-XXL text encoder, achieving good alignment between Chinese and English. We
first developed our general model using billion-level data and then tuned the anime model through a series of
post-training strategies and curated data. By open-sourcing the Alpha version, we hope to contribute to the development
of the anime community, and we greatly value any feedback.

# News

* 2024.12.2 🔥 Model and code released!

# Key Features

- Good bilingual prompt following, effectively transforming certain Chinese concepts into anime style.
- The model is mainly にじげん(二次元) style, supporting common artistic styles and Chinese elements.
- Competitive image quality, especially in generating detailed characters and landscapes.
- Prediction mode is x-prediction, so the model tends to produce subjects with cleaner backgrounds; more detailed
  prompts can further refine your images.
- Impressive creative ability, the more detailed the descriptions are, the more surprises it can produce.
- Embracing open-source co-construction; we welcome anime fans to join our ecosystem and share your creative ideas
  through our workflow.
- Better support for Chinese-style elements.
- Compatible with both tag lists and natural language description-style prompts.

# Model Info

<table>
  <tr>
    <th>Developed by</th>
    <td>animEEEmpire</td>
  </tr>
  <tr>
    <th>Model Name</th>
    <td>AniMemory-alpha</td>
  </tr>
  <tr>
    <th>Model type</th>
    <td>Diffusion-based text-to-image generative model</td>
  </tr>
  <tr>
    <th>Download link</th>
    <td><a href="https://huggingface.co/animEEEmpire/AniMemory-alpha">Hugging Face</a></td>
  </tr>
  <tr>
    <th rowspan="4">Parameter</th>
    <td>TextEncoder_1: 5.6B</td>
  </tr>
  <tr>
    <td>TextEncoder_2: 950M</td>
  </tr>
  <tr>
    <td>Unet: 3.1B</td>
  </tr>
  <tr>
    <td>VAE: 271M</td>
  </tr>
  <tr>
    <th>Context Length</th>
    <td>227</td>
  </tr>
  <tr>
    <th>Resolution</th>
    <td>Multi-resolution</td>
  </tr>
</table>

# Key Problems and Notes

- Primarily focuses on text-following ability and basic image quality; it is not a strongly artistic or stylized
  version, making it suitable for open-source co-construction.
- Quantization and distillation are still in progress, leaving room for significant speed improvements and GPU memory
  savings. We are planning for this and looking forward to volunteers.
- A relatively complete data filtering and cleaning process has been conducted, so it is not adept at pornographic
  generation; any attempts to force it may result in image crashes.
- Simple descriptions tend to produce images with simple backgrounds and chibi-style illustrations; you can try to
  enhance the detail by providing comprehensive descriptions.
- For close-up shots, please use descriptions like "detailed face", "close-up view" etc. to enhance the impact of the
  output.
- Adding necessary quality descriptors can sometimes improve the overall quality.
- The issue with small faces still exists in the Alpha version, but it has been slightly improved; feel free to try it
  out.
- It is better to detail a single object rather than too many objects in one prompt.

# Limitations

- Although the model data has undergone extensive cleaning, there may still be potential gender, ethnic, or political
  biases.
- The model's open-sourcing is dedicated to enriching the ecosystem of the anime community and benefiting anime fans.
- The usage of the model shall not infringe upon the legal rights and interests of designers and creators.

# Quick Start

1. Install the necessary requirements.

- Recommended Python >= 3.10, PyTorch >= 2.3, CUDA >= 12.1.

- It is recommended to use Anaconda to create a new environment (Python >=
  3.10) `conda create -n animemory python=3.10 -y` to run the following example.

- run `pip install git+https://github.com/huggingface/diffusers.git torch==2.3.1 transformers==4.43.0 accelerate==0.31.0 sentencepiece`

2. Diffusers inference.

- The pipeline has not been merged yet. Please use the following code to setup the environment.
```shell
git clone https://github.com/huggingface/diffusers.git
git clone https://github.com/animEEEmpire/diffusers_animemory
cp diffusers_animemory/* diffusers -r

# Method 1: Re-install diffusers. (Recommended)
cd diffusers
pip install .

# Method 2: Call it locally. Change `YOUR_PATH` to the directory where you just cloned `diffusers` and `diffusers_animemory`.
import sys
sys.path.insert(0, 'YOUR_PATH/diffusers/src')
```
- And then, you can use the following code to generate images.

```python
from diffusers import AniMemoryPipeline
import torch

pipe = AniMemoryPipeline.from_pretrained("animEEEmpire/AniMemory-alpha", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "一只凶恶的狼，猩红的眼神，在午夜咆哮，月光皎洁"
negative_prompt = "nsfw, worst quality, low quality, normal quality, low resolution, monochrome, blurry, wrong, Mutated hands and fingers, text, ugly faces, twisted, jpeg artifacts, watermark, low contrast, realistic"

images = pipe(prompt=prompt,
              negative_prompt=negative_prompt,
              num_inference_steps=40,
              height=1024, width=1024,
              guidance_scale=7,
              num_images_per_prompt=1
              )[0]
images.save("output.png")
```

- Use `pipe.enable_sequential_cpu_offload()` to offload the model into CPU for less GPU memory cost (about 14.25 G,
compared to 25.67 G if CPU offload is not enabled), but the inference time will increase significantly(5.18s v.s. 
17.74s on A100 40G).

For faster inference, please refer to our future work.

For ComfyUI, please visit [ComfyUI-Animemory-Loader](https://github.com/animEEEmpire/ComfyUI-Animemory-Loader).

# Citation

```bibtex
@misc{animEEEmpire-2024-AniMemory-alpha,
  author = {animEEEmpire},
  title = {AniMemory-Alpha: A Bilingual Anime Image Generation Diffusion Model},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/animEEEmpire/AniMemory-alpha}}
}
```

# License

This repo is released under the Apache 2.0 License.
