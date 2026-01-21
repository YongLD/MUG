<div align="center">

<h1>
    <img src="assets/logo.png" alt="Logo" width="40" style="vertical-align: middle;" /> 
    <b>Multi-agent Undercover Gaming</b>: Hallucination Removal through Counterfactual Test for Multimodal Reasoning
  </h1>

[Dayong Liang](https://github.com/YongLD)<sup>1,4,\*</sup>, [Xiao-Yong Wei](https://scholar.google.com/citations?user=8kxWTokAAAAJ&hl=en)<sup>2,3,4,\*</sup>, [Changmeng Zheng](https://github.com/thecharm)<sup>3,†</sup>

<p><sup>1</sup>South China University of Technology &nbsp;&nbsp;<sup>2</sup>Sichuan University &nbsp;&nbsp;<sup>3</sup>The Hong Kong Polytechnic &nbsp;&nbsp;<sup>4</sup>Peng Cheng Laboratory
<br><sup>*</sup>Contributed equally &nbsp;&nbsp; <sup>†</sup>Corresponding author
<h5 align="center">

<!-- [![arXiv](https://img.shields.io/badge/Arxiv-2406.07476-AD1C18.svg?logo=arXiv)](https://arxiv.org/pdf/2403.14972) -->

</h5>

<!-- <img src="assets/mug.png" alt="framework" title="MUG" width="700" /> -->

</div>

<!-- The MUG protocol introduces a novel approach to mitigate hallucinations in large language models through systematic counterfactual validation. By leveraging principles from social deduction games, our framework enhances reasoning reliability among multi-agent systems. -->

---

*Latest News* 🔥

- 🔄 **Counterfactual Image Generation Pipeline**: In progress.
- ✅ [2026/01] **Code Implementation**: Inference code is released.
- ✅ [2025/11] **Paper Accepted**: Our paper has been accepted at AAAI2026! 🎉

## Features

- Systematic counterfactual testing for hallucination detection
- Dynamic cross-evidence reasoning capabilities
- Active reasoning paradigms facilitating exploratory dialogue among agents

## 🛠️ Installation
To set up the environment for MUG, please follow the steps below. We recommend using Conda to manage your dependencies.

**1. Clone the repository**
```bash
git clone https://github.com/YongLD/MUG.git
cd MUG
```

**2. Create a new Conda environment**
```bash
conda create -n mug python==3.12
conda activate mug
```

**3. Install the dependencies**

```bash
pip install -e .
```
**4.Setup Qwen3 Environment**
This project relies on Qwen2.5VL. You must install the dependencies required by the official Qwen3VL repository. Please refer to the [Qwen3VL GitHub Repository](https://github.com/QwenLM/Qwen3-VL) for the most up-to-date installation instructions.

Generally, you will need to install the necessary libraries (like transformers, flash_attention, accelerate, torch, etc.) compatible with Qwen3VL.

##🚀 Usage & Testing
Once the environment is configured and the Qwen3 dependencies are installed, you can run the main testing script.

Run the prediction script: 
```bash
sh predict.sh
```

## 📂 Counterfactual Data

We provide access to a subset of our generated counterfactual images via Google Drive.

- **Download Link**: [Google Drive Link](https://drive.google.com/drive/folders/1MpaSvRVG5wWqBMn0Yn3YUKFokaSXVXDw?usp=sharing)

> ⚠️ **Note on Data Quality**:
> The current batch of counterfactual images was generated using **Step1X-Edit**. Please note that the quality of these samples is preliminary and may be suboptimal.
>
> 🚀 **Future Plans**: We are actively working on utilizing more advanced generative models to produce high-quality counterfactual images. We will release these improved datasets along with the complete **Counterfactual Image Generation Pipeline** soon.

## ❤️ Acknowledgments

- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): An open-source evaluation toolkit of large vision-language models (LVLMs).

## 📑 Citation

If this repo is useful to you, please cite using this BibTeX.
```bibtex
@article{liang2025multi,
  title={Multi-agent Undercover Gaming: Hallucination Removal via Counterfactual Test for Multimodal Reasoning},
  author={Liang, Dayong and Wei, Xiao-Yong and Zheng, Changmeng},
  journal={arXiv preprint arXiv:2511.11182},
  year={2025}
}
``` 
