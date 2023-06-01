# LoRA 학습에 대하여

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)（arxiv）、[LoRA](https://github.com/microsoft/LoRA)（github）를 Stable Diffusion에 적용된 모델입니다.

[cloneofsimo의 레포지토리](https://github.com/cloneofsimo/lora)의 저장소를 크게 참고하였습니다. 감사합니다.

일반적인 LoRA는 Linear 및 1x1 커널 크기의 Conv2d에만 적용됩니다. 하지만 3x3 커널 크기의 Conv2d에도 적용할 수 있습니다.

Conv2d 3x3으로의 확장은 [cloneofsimo](https://github.com/cloneofsimo/lora)가 처음으로 릴리스하고, KohakuBlueleaf가 [LoCon](https://github.com/KohakuBlueleaf/LoCon)에서 그 효과를 밝혀냈습니다. KohakuBlueleaf에게 깊은 감사를 전합니다.

8GB VRAM에서도 작동하는 것으로 보입니다.

[공통 문서인 학습에 대한 README](./train_README-ja.md)도 참고하시기 바랍니다.

# 학습 가능한 LoRA 종류

다음 두 가지 유형을 지원합니다. 이는 이 저장소에서 사용하는 고유한 이름입니다.

1. LoRA-LierLa: (LoRA for __Li__near lay__e__r __La__yers, 리에라로 발음합니다)

  Linear 및 1x1 커널 크기의 Conv2d에 적용되는 LoRA

1. LoRA-C3Lier: (LoRA for __C__onvolutional layers with __3__x3 Kernel and __Li__near layers, 세리아로 발음합니다)

  1.에 추가로 3x3 커널 크기의 Conv2d에 적용되는 LoRA

LoRA-C3Lier는 LoRA-LierLa보다 적용되는 층이 더 많아 더 높은 정확도를 기대할 수 있습니다.

또한 학습 시에는 __DyLoRA__ 를 사용할 수도 있습니다 (뒷부분에서 설명합니다).

## 학습된 모델에 대한 주의 사항

LoRA-LierLa는 AUTOMATIC1111의 웹 UI의 LoRA 기능에서 사용할 수 있습니다.

LoRA-C3Liar를 사용하여 웹 UI에서 생성하려면 이 [웹 UI용 extension](https://github.com/kohya-ss/sd-webui-additional-networks)을 사용하십시오.

둘 다 학습한 LoRA 모델을 이 리포지토리의 스크립트를 사용하여 미리 Stable Diffusion 모델에 병합할 수도 있습니다.

cloneofsimo의 저장소와 d8ahazard의 [Dreambooth Extension for Stable-Diffusion-WebUI](https://github.com/d8ahazard/sd_dreambooth_extension)는 현재 호환되지 않습니다. 이는 일부 기능 확장을 수행하기 때문입니다 (이후 설명).

# 학습 절차

이 리포지토리의 README를 참조하여 미리 환경을 설정하십시오.

## 데이터 준비

[데이터 준비에 대한 README](./train_README-ja.md) 를 참조하십시오.

## 학습 실행

`train_network.py`를 사용합니다.

`train_network.py`에서는 `--network_module` 옵션에 학습할 모듈의 이름을 지정합니다. LoRA에 해당하는 모듈은 `network.lora`이므로 해당 모듈을 지정하십시오.

학습률은 일반적인 DreamBooth나 fine tuning보다 더 높은 `1e-4`~`1e-3` 정도로 지정하는 것이 좋습니다.

다음은 명령줄의 예시입니다.

```
accelerate launch --num_cpu_threads_per_process 1 train_network.py 
    --pretrained_model_name_or_path=<.ckpt 또는 .safetensor 또는 Diffusers 버전 모델 디렉토리> 
    --dataset_config=<데이터 준비에서 생성한 .toml 파일> 
    --output_dir=<학습된 모델의 출력 폴더>  
    --output_name=<학습된 모델 출력 파일 이름> 
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=400 
    --learning_rate=1e-4 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
    --save_every_n_epochs=1 
    --network_module=networks.lora
```

이 명령줄은 LoRA-LierLa를 학습합니다.

`--output_dir` 옵션으로 지정한 폴더에 LoRA 모델이 저장됩니다. 다른 옵션 및 옵티마이저에 대해서는 [공통 학습 문서](./train_README-ja.md)의 "자주 사용되는 옵션"도 참조하십시오.

그 외에도 다음과 같은 옵션을 지정할 수 있습니다.

* `--network_dim`
  * LoRA의 RANK를 지정합니다 (``--networkdim=4`` 등). 기본값은 4입니다. 숫자가 많을수록 표현 능력이 증가하지만, 필요한 메모리와 시간이 증가합니다. 무작정 증가시키는 것은 좋지 않은 것 같습니다.
* `--network_alpha`
  * 언더플로우를 방지하고 안정적인 학습을 위해 ``alpha`` 값을 지정합니다. 기본값은 1입니다. ``network_dim``과 동일한 값을 지정하면 이전 버전과 동일한 동작이 가능합니다.
* `--persistent_data_loader_workers`
  * Windows 환경에서 지정하면 에포크 간 대기 시간이 크게 줄어듭니다.
* `--max_data_loader_n_workers`
  * 데이터 로더의 프로세스 수를 지정합니다. 프로세스 수가 많을수록 데이터 로딩 속도가 빨라지고 GPU를 효율적으로 사용할 수 있지만, 메인 메모리를 소비합니다. 기본값은 「8 또는 `CPU 동시 실행 스레드 수 - 1` 중 작은 값」이므로 메인 메모리 여유가 없거나, GPU 사용률이 90% 정도 이상이라면 해당 값을 확인하고 2 또는 1 정도로 낮추십시오.
* `--network_weights`
  * 학습 이전에 학습된 LoRA의 가중치를 불러와 추가로 학습합니다.
* `--network_train_unet_only`
  * U-Net에 관련된 LoRA 모듈만 활성화합니다. Fine-tuning과 같은 학습에 사용하면 좋을 수 있습니다.
* `--network_train_text_encoder_only`
  * Text Encoder에 관련된 LoRA 모듈만 활성화합니다. Textual Inversion과 같은 효과가 기대될 수 있습니다.
* `--unet_lr`
  * U-Net에 관련된 LoRA 모듈에 일반적인 학습률(--learning_rate 옵션으로 지정)과는 다른 학습률을 사용할 때 지정합니다.
* `--text_encoder_lr`
  * Text Encoder에 관련된 LoRA 모듈에 일반적인 학습률(--learning_rate 옵션으로 지정)과는 다른 학습률을 사용할 때 지정합니다. Text Encoder 쪽을 약간 낮은 학습률(예: 5e-5)로 지정하는 것이 좋다는 이야기도 있습니다.
* `--network_args`
  * 여러 개의 인수를 지정할 수 있습니다. 후술하겠습니다.

`--network_train_unet_only`와 `--network_train_text_encoder_only` 모두 지정하지 않은 경우(기본값)에는 Text Encoder와 U-Net의 모든 LoRA 모듈을 활성화합니다.

# 기타 학습 방법

## LoRA-C3Lier의 학습

`--network_args`에 다음과 같이 지정하십시오. `conv_dim`으로 Conv2d (3x3)의 rank를, `conv_alpha`로 alpha를 지정하십시오.

```
--network_args "conv_dim=4" "conv_alpha=1"
```

alpha를 지정하지 않을 경우 기본값인 1로 설정됩니다.

```
--network_args "conv_dim=4"
```

## DyLoRA

DyLoRA는 다음 논문에서 제안된 방법입니다. [DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation](https://arxiv.org/abs/2210.07558) 공식 구현은 [여기](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)에서 찾을 수 있습니다.

논문에 따르면, LoRA의 rank가 높을수록 항상 더 좋은 것은 아니며, 대상 모델, 데이터셋, 태스크 등에 따라 적절한 rank를 찾아야 합니다. DyLoRA를 사용하면 지정한 dim(rank) 이하의 다양한 rank를 동시에 학습하여 각각의 최적 rank를 찾을 수 있습니다.

이 리포지토리의 구현은 공식 구현을 기반으로 독자적인 확장을 추가했습니다(따라서 버그 등이 있을 수 있습니다).

### 이 리포지토리의 DyLoRA 특징

학습 후의 DyLoRA 모델 파일은 LoRA와 호환됩니다. 또한 모델 파일에서 지정한 dim(rank) 이하의 다양한 dim의 LoRA를 추출할 수 있습니다.

DyLoRA-LierLa와 DyLoRA-C3Lier를 모두 학습할 수 있습니다.

### DyLoRA로 학습하기
`--network_module=networks.dylora`와 같이 DyLoRA에 해당하는 `network.dylora`를 지정하십시오.

또한 `--network_args`에 `--network_args "unit=4"`와 같이 `unit`을 지정합니다. unit은 rank를 분할하는 단위입니다. 예를 들어 `--network_dim=16` `--network_args "unit=4"`와 같이 지정합니다. `unit`은 `network_dim`으로 나누어 떨어지는 값(즉, `network_dim`은 `unit`의 배수)으로 지정하십시오.

`unit`을 지정하지 않을 경우 `unit=1`로 처리됩니다.

다음은 기술 예시입니다.

```
--network_module=networks.dylora --network_dim=16 --network_args "unit=4"

--network_module=networks.dylora --network_dim=32 --network_alpha=16 --network_args "unit=4"
```

DyLoRA-C3Lier의 경우, `--network_args`에 `"conv_dim=4"`와 같이 `conv_dim`을 지정합니다. 일반적인 LoRA와 달리 `conv_dim`은 `network_dim`과 동일한 값이어야 합니다. 기술 예시는 다음과 같습니다.

```
--network_module=networks.dylora --network_dim=16 --network_args "conv_dim=16" "unit=4"

--network_module=networks.dylora --network_dim=32 --network_alpha=16 --network_args "conv_dim=32" "conv_alpha=16" "unit=8"
```

예를 들어 dim=16, unit=4로 학습하면 4, 8, 12, 16의 4개 rank의 LoRA를 학습하고 추출할 수 있습니다. 추출한 각 모델로 이미지를 생성하여 비교함으로써 최적의 rank의 LoRA를 선택할 수 있습니다.

그 외의 옵션은 일반적인 LoRA와 동일합니다.

※ unit은 이 리포지토리의 독자적인 확장입니다. DyLoRA의 경우 동일한 dim(rank)의 일반적인 LoRA에 비해 학습 시간이 길어질 수 있으므로 분할 단위를 크게 한 것입니다.


### DyLoRA 모델에서 LoRA 모델 추출하기
`networks` 폴더 내의 `extract_lora_from_dylora.py`를 사용합니다. 지정한 `unit` 단위로 DyLoRA 모델에서 LoRA 모델을 추출합니다.

명령줄 예시는 다음과 같습니다.

```powershell
python networks\extract_lora_from_dylora.py --model "foldername/dylora-model.safetensors" --save_to "foldername/dylora-model-split.safetensors" --unit 4
```

`--model`에는 DyLoRA 모델 파일을 지정하고, `--save_to`에 추출한 모델을 저장할 파일 이름을 지정합니다(랭크 값이 파일 이름에 추가됩니다). `--unit`에는 DyLoRA 학습 시 사용한 `unit`을 지정합니다.

## 계층별 학습률

자세한 내용은 [PR #355](https://github.com/kohya-ss/sd-scripts/pull/355)를 참조하세요.

전체 모델의 25개 블록의 가중치를 지정할 수 있습니다. 첫 번째 블록에 해당하는 LoRA는 존재하지 않지만, 계층별 LoRA 적용 등의 호환성을 위해 항상 25개 값으로 지정합니다. 또한 conv2d3x3로 확장하지 않는 경우에도 일부 블록에는 LoRA가 존재하지 않지만, 통일된 기술을 위해 항상 25개 값을 지정해야 합니다.

`--network_args`를 사용하여 다음과 같은 인수를 지정합니다.

- `down_lr_weight`: U-Net의 down 블록의 학습률 가중치를 지정합니다. 다음과 같은 값이 지정 가능합니다.
  - 블록별 가중치: `"down_lr_weight=0,0,0,0,0,0,1,1,1,1,1,1"`과 같이 12개의 숫자를 지정합니다.
  - 프리셋 지정: `"down_lr_weight=sine"`과 같이 지정합니다(사인 곡선을 사용하여 가중치를 지정합니다). sine, cosine, linear, reverse_linear, zeros가 지정 가능합니다. 또한 `"down_lr_weight=cosine+.25"`와 같이 `+숫자`를 추가하면 지정한 숫자가 더해집니다(0.25~1.25가 됩니다).
- `mid_lr_weight`: U-Net의 mid 블록의 학습률 가중치를 지정합니다. `"down_lr_weight=0.5"`와 같이 하나의 숫자만 지정합니다.
- `up_lr_weigh`t: U-Net의 up 블록의 학습률 가중치를 지정합니다. down_lr_weight와 동일합니다.
- 지정을 생략한 부분은 1.0으로 처리됩니다. 또한 가중치를 0으로 설정하면 해당 블록의 LoRA 모듈이 생성되지 않습니다.
- `block_lr_zero_threshold`: 가중치가 이 값 이하인 경우 LoRA 모듈을 생성하지 않습니다. 기본값은 0입니다.

### 계층별 학습률 명령줄 예시:

```powershell
--network_args "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5" "mid_lr_weight=2.0" "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5"

--network_args "block_lr_zero_threshold=0.1" "down_lr_weight=sine+.5" "mid_lr_weight=1.5" "up_lr_weight=cosine+.5"
```

### 계층별 학습률 toml 파일 예시:

```toml
--network_args = [ "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5", "mid_lr_weight=2.0", "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5",]

--network_args = [ "block_lr_zero_threshold=0.1", "down_lr_weight=sine+.5", "mid_lr_w
```

## 계층별 dim (rank)

전체 모델의 25개 블록의 dim (rank)을 지정할 수 있습니다. 계층별 학습률과 마찬가지로 일부 블록에는 LoRA가 존재하지 않을 수 있지만, 항상 25개의 값을 지정해야 합니다.

`--network_args`를 사용하여 다음과 같은 인수를 지정합니다.

- `block_dims`: 각 블록의 dim (rank)을 지정합니다. `"block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"`와 같이 25개의 숫자를 지정합니다.
- `block_alphas`: 각 블록의 alpha를 지정합니다. block_dims와 동일하게 25개의 숫자를 지정합니다. 생략하면 network_alpha의 값이 사용됩니다.
- `conv_block_dims`: LoRA를 Conv2d 3x3로 확장한 경우 각 블록의 dim (rank)을 지정합니다.
- `conv_block_alphas`: LoRA를 Conv2d 3x3로 확장한 경우 각 블록의 alpha를 지정합니다. 생략하면 conv_alpha의 값이 사용됩니다.

## 계층별 dim (rank) 명령줄 예시:

```powershell
--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "conv_block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"
```

## 계층별 dim (rank) toml 파일 지정 예시:

``` toml
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2",]
  
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,
```

# 기타 스크립트

LoRA와 관련된 다른 스크립트에 대해 설명합니다.

## 병합 스크립트 정보

merge_lora.py 스크립트를 사용하면, Stable Diffusion 모델에 LoRA 모델의 학습 결과를 병합하거나 여러 개의 LoRA 모델을 병합할 수 있습니다.

### Stable Diffusion 모델에 LoRA 모델 병합하기

병합된 모델은 일반적인 Stable Diffusion ckpt와 동일하게 사용할 수 있습니다. 다음과 같은 명령줄을 사용할 수 있습니다.

```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors --ratios 0.8
```

Stable Diffusion v2.x 모델에 학습한 후에 병합하는 경우, --v2 옵션을 지정해야 합니다.

--sd_model 옵션에 병합할 Stable Diffusion 모델 파일을 지정합니다 (ckpt 또는 .safetensors만 지원됨, Diffusers는 현재 지원되지 않음).

--save_to 옵션에 병합된 모델의 저장 경로를 지정합니다 (ckpt 또는 .safetensors, 확장자에 따라 자동으로 판별).

--models에 학습한 LoRA 모델 파일을 지정합니다. 여러 개 지정도 가능하며, 순서대로 병합됩니다.

--ratios에 각 모델의 적용 비율을 0~1.0 사이의 숫자로 지정합니다 (모델 수와 동일한 개수로 지정). 예를 들어, 과적합에 가까운 경우 비율을 낮추면 더 좋은 결과를 얻을 수 있을 것입니다. 모델 수와 동일한 개수로 지정해야 합니다.

여러 개의 모델을 지정하는 경우 다음과 같이 사용합니다.

```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors --ratios 0.8 0.5
```

### 여러 개의 LoRA 모델 병합하기

여러 개의 LoRA를 병합하는 경우에는 원칙적으로 `svd_merge_lora.py`를 사용하십시오. 간단한 up 또는 down 간의 병합은 계산 결과가 올바르지 않을 수 있기 때문입니다.

`merge_lora.py`를 사용한 병합은 차분 추출 방법을 사용하여 LoRA를 생성하는 경우 등, 매우 제한적인 경우에만 유효합니다.

다음과 같은 명령줄을 사용할 수 있습니다.

```
python networks\merge_lora.py 
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors --ratios 0.6 0.4
```

--sd_model 옵션은 필요하지 않습니다.

--save_to 옵션에 병합된 LoRA 모델의 저장 경로를 지정합니다 (ckpt 또는 .safetensors, 확장자에 따라 자동으로 판별).

--models에 학습한 LoRA 모델 파일을 지정합니다. 세 개 이상도 지정 가능합니다.

--ratios에 각 모델의 비율을 0~1.0 사이의 숫자로 지정합니다. 두 개의 모델을 일대일로 병합하는 경우 "0.5 0.5"가 됩니다. "1.0 1.0"으로 설정하면 총 가중치가 너무 크게 되어 결과가 그다지 좋지 않을 것으로 예상됩니다.

v1에서 학습한 LoRA와 v2에서 학습한 LoRA, 차원(rank)이 다른 LoRA는 병합할 수 없습니다. 오직 U-Net만 있는 LoRA와 U-Net+Text Encoder가 있는 LoRA는 병합할 수 있을 것으로 예상되지만 결과는 알 수 없습니다.

### 그 외의 옵션

* precision
  * 병합 계산 시 사용할 정밀도를 float, fp16, bf16 중에서 선택할 수 있습니다. 생략할 경우 정밀도를 유지하기 위해 float로 설정됩니다. 메모리 사용량을 줄이고 싶은 경우 fp16 또는 bf16을 선택하십시오.
* save_precision
  * 모델 저장 시 사용할 정밀도를 float, fp16, bf16 중에서 선택할 수 있습니다. 생략할 경우 precision과 동일한 정밀도가 적용됩니다.


## 다른 rank를 가진 여러 개의 LoRA 모델 병합하기

여러 개의 LoRA 모델을 하나의 LoRA로 근사화합니다 (완벽히 재현은 불가능합니다). `svd_merge_lora.py` 스크립트를 사용합니다. 다음과 같은 명령줄을 사용합니다.

```
python networks\svd_merge_lora.py 
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors 
    --ratios 0.6 0.4 --new_rank 32 --device cuda
```

`merge_lora.py`와 주요 옵션은 동일합니다. 다음 옵션을 추가할 수 있습니다.

- `--new_rank`
  - 생성할 LoRA의 rank를 지정합니다.
- `--new_conv_rank`
  - 생성할 Conv2d 3x3 LoRA의 rank를 지정합니다. 생략하면 `new_rank`와 동일한 값으로 설정됩니다.
- `--device`
  - `--device cuda`를 지정하여 cuda를 사용하면 GPU에서 계산이 수행됩니다. 처리 속도가 향상됩니다.

## 이미지 생성 스크립트에서 사용하기

gen_img_diffusers.py 스크립트에서는 --network_module 및 --network_weights 옵션을 사용할 수 있습니다. 이들의 의미는 학습 시와 동일합니다.

--network_mul 옵션을 사용하여 0에서 1.0 사이의 값을 지정하면 LoRA의 적용 비율을 조정할 수 있습니다.

## Diffusers의 pipeline에서 생성하기

아래의 예시를 참고하여주세요. 필요한 파일은 networks/lora.py만 필요합니다. Diffusers의 버전은 0.10.2 이외에는 작동하지 않을 수 있습니다.

```python
import torch
from diffusers import StableDiffusionPipeline
from networks.lora import LoRAModule, create_network_from_weights
from safetensors.torch import load_file

# 만약 ckpt 파일이 CompVis 기반인 경우, 먼저 tools/convert_diffusers20_original_sd.py를 사용하여 Diffusers로 변환해야 합니다. 자세한 내용은 --help를 참조하세요.

model_id_or_dir = r"model_id_on_hugging_face_or_dir"
device = "cuda"

# 파이프라인 생성
print(f"{model_id_or_dir}에서 파이프라인 생성 중...")
pipe = StableDiffusionPipeline.from_pretrained(model_id_or_dir, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to(device)
vae = pipe.vae
text_encoder = pipe.text_encoder
unet = pipe.unet

# LoRA 네트워크 로드
print("LoRA 네트워크 로드 중...")

lora_path1 = r"lora1.safetensors"
sd = load_file(lora_path1)   # 파일이 .ckpt인 경우에는 torch.load를 사용해야 합니다.
network1, sd = create_network_from_weights(0.5, None, vae, text_encoder, unet, sd)
network1.apply_to(text_encoder, unet)
network1.load_state_dict(sd)
network1.to(device, dtype=torch.float16)

# # apply_to+load_state_dict 대신에 weights를 병합할 수도 있습니다. network.set_multiplier는 동작하지 않습니다.
# network.merge_to(text_encoder, unet, sd)

lora_path2 = r"lora2.safetensors"
sd = load_file(lora_path2) 
network2, sd = create_network_from_weights(0.7, None, vae, text_encoder, unet, sd)
network2.apply_to(text_encoder, unet)
network2.load_state_dict(sd)
network2.to(device, dtype=torch.float16)

lora_path3 = r"lora3.safetensors"
sd = load_file(lora_path3)
network3, sd = create_network_from_weights(0.5, None, vae, text_encoder, unet, sd)
network3.apply_to(text_encoder, unet)
network3.load_state_dict(sd)
network3.to(device, dtype=torch.float16)

# 프롬프트
prompt = "masterpiece, best quality, 1girl, in white shirt, looking at viewer"
negative_prompt = "bad quality, worst quality, bad anatomy, bad hands"

# 실행
print("이미지 생성 중...")
with torch.autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5, negative_prompt=negative_prompt).images[0]

# 병합되지 않은 경우, set_multiplier를 사용할 수 있습니다.
# network1.set_multiplier(0.8)
# 그리고 이미지를 다시 생성합니다.

# 이미지 저장
image.save(r"by_diffusers..png")
```

## 두 개의 모델 차이에서 LoRA 모델 생성하기

[이것은 이 디스커션](https://github.com/cloneofsimo/lora/discussions/56) 을 참고하여 구현된 것입니다. 수식은 동일하게 사용되었습니다
(이해는 잘 못했지만, 특이값 분해를 사용한 근사화로 보입니다).

두 개의 모델(예: Fine-tuning 이전의 모델과 Fine-tuning 후의 모델) 간의 차이를 LoRA로 근사화합니다.

### 스크립트 실행 방법

다음과 같이 지정하십시오.

```
python networks\extract_lora_from_models.py --model_org base-model.ckpt
    --model_tuned fine-tuned-model.ckpt 
    --save_to lora-weights.safetensors --dim 4
```

--model_org 옵션에 원래 Stable Diffusion 모델을 지정합니다. 생성한 LoRA 모델을 적용하려면 이 모델을 지정하여 적용하면 됩니다. .ckpt 또는 .safetensors를 지정할 수 있습니다.

--model_tuned 옵션에 차이를 추출할 대상 Stable Diffusion 모델을 지정합니다. 예를 들어 Fine-tuning이나 DreamBooth 후의 모델을 지정합니다. .ckpt 또는 .safetensors를 지정할 수 있습니다.

--save_to에 LoRA 모델의 저장 위치를 지정합니다. --dim에 LoRA의 차원 수를 지정합니다.

생성된 LoRA 모델은 학습한 LoRA 모델과 동일하게 사용할 수 있습니다.

Text Encoder가 두 모델에서 동일한 경우 LoRA는 U-Net만으로 이루어진 LoRA가 됩니다.

### 기타 옵션

- `--v2`
  - v2.x의 Stable Diffusion 모델을 사용할 경우 지정하세요.
- `--device`
  - ``--device cuda``로 지정하여 계산을 GPU에서 수행합니다. 처리 속도가 빨라집니다(하지만 CPU에서도 크게 느려지지 않으므로 2배에서 몇 배 정도로 생각하면 됩니다).
- `--save_precision`
  - LoRA의 저장 형식을 "float", "fp16", "bf16" 중에서 선택합니다. 기본값은 float입니다.
- `--conv_dim`
  - 지정하면 LoRA의 적용 범위를 Conv2d 3x3로 확장합니다. Conv2d 3x3의 rank를 지정합니다.

## 이미지 리사이즈 스크립트

(나중에 문서를 정리하지만 일단 여기에 설명을 작성합니다.)

Aspect Ratio Bucketing 기능을 확장하여 작은 이미지는 확대하지 않고 원래의 교사 데이터로 사용할 수 있게 되었습니다. 원본 교사 이미지를 축소한 이미지를 교사 데이터에 추가하면 정확도가 향상된다는 보고와 함께 전처리용 스크립트를 받았으므로 추가하였습니다. bmaltais님에게 감사드립니다.

### 스크립트 실행 방법

다음과 같이 지정하십시오. 원래 이미지와 리사이즈된 이미지가 변환 대상 폴더에 저장됩니다. 리사이즈된 이미지의 파일명에는 ``+512x512``와 같이 리사이즈 대상 해상도가 추가됩니다 (이미지 크기와는 다릅니다). 리사이즈 대상 해상도보다 작은 이미지는 확대되지 않습니다.

```
python tools\resize_images_to_resolution.py --max_resolution 512x512,384x384,256x256 --save_as_png 
    --copy_associated_files 원본_이미지_폴더 변환_대상_폴더
```

원본 이미지 폴더 내의 이미지 파일은 지정한 해상도 (여러 개 지정 가능)와 동일한 면적이 되도록 리사이즈되어 변환 대상 폴더에 저장됩니다. 이미지 이외의 파일은 그대로 복사됩니다.

``--max_resolution`` 옵션에 리사이즈 대상 크기를 위의 예시와 같이 지정하십시오. 면적이 해당 크기가 되도록 리사이즈됩니다. 여러 개 지정하면 각 해상도로 리사이즈됩니다. ``512x512,384x384,256x256``이라면 변환 대상 폴더의 이미지는 원래 크기와 리사이즈 후 크기×3으로 총 4개의 이미지가 됩니다.

``--save_as_png`` 옵션을 지정하면 png 형식으로 저장됩니다. 생략하면 jpeg 형식 (품질=100)으로 저장됩니다.

``--copy_associated_files`` 옵션을 지정하면 이미지와 동일한 파일명 (예: 캡션 등)의 파일이 리사이즈된 이미지의 파일명과 동일한 이름으로 복사됩니다.

### 그 외의 옵션

- divisible_by
  - 리사이즈된 이미지의 크기 (가로, 세로 각각)가 이 값으로 나누어 떨어지도록 이미지 중심을 잘라냅니다.
- interpolation
  - 축소 시 보간 방법을 지정합니다. ``area, cubic, lanczos4`` 중 선택할 수 있으며, 기본값은 ``area``입니다.

# 추가 정보

## cloneofsimo님의 저장소와의 차이점

2022/12/25 기준으로 이 저장소는 LoRA의 적용 범위를 Text Encoder의 MLP, U-Net의 FFN, Transformer의 in/out projection으로 확장하여 표현력을 향상시켰습니다. 하지만 그에 대한 대가로 메모리 사용량이 증가하여 8GB에 근접하게 되었습니다.

또한 모듈 교체 메커니즘은 전혀 다릅니다.

## 향후 확장에 대해

LoRA뿐만 아니라 다른 확장도 지원할 수 있으므로, 해당 확장도 추가 예정입니다.