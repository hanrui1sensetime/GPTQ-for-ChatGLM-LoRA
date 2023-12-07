# GPTQ-for-MedGPT
<img src = https://user-images.githubusercontent.com/64115820/235287009-2d07bba8-9b85-4973-9e06-2a3c28777f06.png width="50%" height="50%">

4 bits quantization of [LLaMA](https://arxiv.org/abs/2302.13971) and [Bloom](https://arxiv.org/abs/2211.05100) using [GPTQ](https://arxiv.org/abs/2210.17323)

This repo is modified from [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)

GPTQ is SOTA one-shot weight quantization method

**Supports the fastest speed, but uses both triton and cuda.**
**Triton only supports Linux, so if you are a Windows user, please use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install).**

## News or Update
**Support pulse model with lora finetuning 4-bit quantization.**
## Result
MedGPT results are evaluated on the medical dataset.

Quantization requires a large amount of CPU memory. However, the memory required can be reduced by using swap memory.

Depending on the GPUs/drivers, there may be a difference in performance, which decreases as the model size increases.(https://github.com/IST-DASLab/gptq/issues/1)

According to [GPTQ paper](https://arxiv.org/abs/2210.17323), As the size of the model increases, the difference in performance between FP16 and GPTQ decreases.

## Installation
If you don't have [conda](https://docs.conda.io/en/latest/miniconda.html), install it first.
```
conda create --name gptq python=3.9 -y
conda activate gptq
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Or, if you're having trouble with conda, use pip with python3.9:
# pip3 install torch torchvision torchaudio

git clone -b pulse https://github.com/hanrui1sensetime/GPTQ-for-MedGPT.git
cd GPTQ-for-MedGPT
pip install -r requirements.txt
python setup_cuda.py install
```
## Dependencies

* `torch`: tested on v2.0.0+cu117
* `transformers`: tested on v4.34.0
* `datasets`: tested on v2.13.1
* `safetensors`: tested on v0.3.1
* `peft`: tested on v0.7.0

All experiments were run on a single NVIDIA RTX3090.

# Language Generation
## PULSE

PULSE-7B model is implemented by bloomz.

```
# Generate 4-bit PULSE-7B model
CUDA_VISIBLE_DEVICES=0 python bloom.py ${MODEL_DIR} custom --wbits 4 --act-order --groupsize 128 --save pulse7b-4bit-128g.bin --calib_data ${CALIB_DATA_PATH}
```

```
# Generate 4-bit PULSE-7B with lora model
CUDA_VISIBLE_DEVICES=0 python bloom_lora.py ${MODEL_DIR} custom --wbits 4 --act-order --groupsize 128 --save pulse7b-4bit-128g.bin --calib_data ${CALIB_DATA_PATH} --peft_path ${PEFT_PATH}
```

# Acknowledgements
This code is based on [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)

Thanks to Meta AI for releasing [LLaMA](https://arxiv.org/abs/2302.13971), a powerful LLM.

Triton GPTQ kernel code is based on [GPTQ-triton](https://github.com/fpgaminer/GPTQ-triton)
