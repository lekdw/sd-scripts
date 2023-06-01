__문서 업데이트 중이므로 내용에 오류가 있을 수 있습니다.__

# 학습에 대한 공통 편

이 저장소에서는 모델의 fine tuning, DreamBooth, LoRA, 그리고 Textual Inversion（[XTI:P+](https://github.com/kohya-ss/sd-scripts/pull/327)을 포함한) 의 학습을 지원합니다. 이 문서에서는 이러한 학습에 공통으로 적용되는 학습 데이터의 준비 방법과 옵션에 대해 설명합니다.

## 개요

이 저장소의 README를 참고하여 환경 설정을 미리 완료해주세요.

다음과 같은 내용에 대해 설명합니다.

1. 학습 데이터의 준비 방법 (설정 파일을 사용하는 새로운 형식)
1. 학습에 사용되는 용어의 간단한 설명
1. 이전의 지정 형식 (설정 파일 없이 명령줄에서 지정)
1. 학습 중에 생성되는 샘플 이미지
1. 각 스크립트에서 공통으로 사용되는 자주 사용되는 옵션
1. fine tuning 방법의 메타데이터 준비: 캡션 생성 등

1번만 실행하면 일단 학습이 가능합니다 (학습에 대한 자세한 내용은 각 스크립트의 문서를 참조하세요). 2번부터는 필요에 따라 참고하시면 됩니다.

# 학습 데이터의 준비에 대하여

학습 데이터의 이미지 파일을 원하는 폴더(복수 가능)에 준비합니다. `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` 형식을 지원합니다. 기본적으로 리사이즈 등의 전처리는 필요하지 않습니다.

하지만 학습 해상도보다 극단적으로 작은 이미지는 사용하지 않거나, 미리 초고해상도 AI 등을 사용하여 확대하는 것을 권장합니다. 또한 극단적으로 큰 이미지(약 3000x3000 픽셀 이상)는 오류가 발생할 수 있으므로 사전에 축소해주시기 바랍니다.

학습 시에는 모델에 학습시킬 이미지 데이터를 정리하고 스크립트에 지정해주어야 합니다. 학습 데이터의 수, 학습 대상, 캡션(이미지 설명)의 유무 등에 따라 여러 가지 방법으로 학습 데이터를 지정할 수 있습니다. 다음과 같은 방식이 있습니다(각각의 이름은 일반적인 것이 아닌 이 저장소 고유의 정의입니다). 정규화 이미지에 대해서는 이후에 설명합니다.

1. DreamBooth, class+identifier 방식 (정규화 이미지 사용 가능)

    특정 단어 (identifier)와 학습 대상을 연결하여 학습합니다. 캡션을 준비할 필요는 없습니다. 예를 들어, 특정 캐릭터를 학습할 때는 캡션을 준비할 필요가 없어 편리하지만, 학습 데이터의 모든 요소가 identifier에 연결되어 학습되므로 생성 시에는 옷을 바꿀 수 없는 등의 문제가 발생할 수 있습니다.

1. DreamBooth, 캡션 방식 (정규화 이미지 사용 가능)

    이미지마다 캡션이 기록된 텍스트 파일을 준비하여 학습합니다. 예를 들어, 특정 캐릭터를 학습할 때 캡션에 이미지의 세부 사항을 기술함으로써 (흰 옷을 입은 캐릭터 A, 빨간 옷을 입은 캐릭터 A 등) 캐릭터와 다른 요소가 분리되어 모델이 캐릭터만을 학습할 수 있습니다.

1. fine tuning 방식 (정규화 이미지 사용 불가)

    미리 캡션을 메타데이터 파일에 정리합니다. 태그와 캡션을 별도로 관리하거나, 학습을 가속화하기 위해 미리 latents를 캐시하는 등의 기능을 지원합니다 (별도의 문서에서 설명하고 있습니다). (fine tuning이라는 이름이지만 fine tuning 이외에도 사용할 수 있습니다.)

학습하고자 하는 대상과 사용 가능한 지정 방식의 조합은 다음과 같습니다.

| 학습 대상 또는 방식 | 스크립트 | DB / class+identifier | DB / 캡션 | fine tuning |
| ----- | ----- | ----- | ----- | ----- |
| 모델을 fine tuning | `fine_tune.py`| x | x | o |
| 모델을 DreamBooth | `train_db.py`| o | o | x |
| LoRA | `train_network.py`| o | o | o |
| Textual Inversion | `train_textual_inversion.py`| o | o | o |

## 어떤 것을 선택할까요

LoRA와 Textual Inversion의 경우, 캡션 파일을 간편하게 준비하지 않고 학습하고 싶다면 DreamBooth class+identifier 방식을 사용하는 것이 좋습니다. 캡션 파일을 준비할 수 있는 경우에는 DreamBooth 캡션 방식을 고려해보세요. 데이터 수가 많고 정규화된 이미지를 사용하지 않는 경우에는 fine-tuning 방식도 고려해 볼 수 있습니다.

DreamBooth의 경우에도 비슷하게 적용됩니다. fine-tuning 방식은 사용할 수 없습니다. fine-tuning의 경우 fine-tuning 방식만 사용할 수 있습니다.

# 각 방식의 지정 방법에 대해

여기서는 각 방식의 일반적인 패턴에 대해서만 설명하겠습니다. 더 자세한 지정 방법은 [데이터셋 설정](./config_README-ja.md)을 참조해주세요.

# DreamBooth, class+identifier 방식 (정규화된 이미지 사용 가능)

이 방식에서는 각 이미지는 `class identifier`라는 캡션으로 학습된 것과 동일한 역할을 합니다 (`shs dog`와 같은 형식으로 표시됩니다).


## 단계 1. identifier와 class 결정하기

학습하려는 대상을 연결하는 단어인 identifier와 대상이 속하는 class를 결정합니다.

(class라는 용어 대신 instance 등 다양한 용어가 있지만, 일단 원본 논문에 맞춥니다.)

아래에 간단히 설명합니다 (자세한 내용은 검색해보세요).

class는 학습 대상의 일반적인 범주입니다. 예를 들어 특정 개 종을 학습하려면 class는 dog가 됩니다. 애니메이션 캐릭터의 경우 모델에 따라 boy, girl, 1boy, 1girl 등이 될 수 있습니다.

identifier는 학습 대상을 식별하고 학습하기 위한 것입니다. 임의의 단어를 사용해도 상관 없지만, 원 논문에 따르면 「tokenizer로 생성된 토큰 중 3자 이하이면서 드문 단어」가 좋다고 합니다.

identifier와 class를 사용하여 「shs dog」와 같이 모델을 학습하면, class를 통해 학습 대상을 식별하여 학습할 수 있습니다.

이미지를 생성할 때 「shs dog」와 같이 지정하면 학습한 개 종류의 이미지가 생성됩니다.

(최근 사용하는 identifier 예시로는 ``shs sts scs cpc coc cic msm usu ici lvl cic dii muk ori hru rik koo yos wny`` 등이 있습니다. 실제로는 Danbooru 태그에 포함되지 않는 단어가 더 좋습니다.)

## 단계 2. 정규화된 이미지를 사용할지 결정하고, 사용하는 경우 정규화된 이미지 생성하기

정규화된 이미지는 학습 대상이 전체 class에 영향을 주는 것을 방지하기 위한 이미지입니다 (언어의 흐름 방지). 정규화된 이미지를 사용하지 않으면, 예를 들어 `shs 1girl`로 특정 캐릭터를 학습시키면, 단순히 `1girl`이라는 프롬프트로 생성하더라도 해당 캐릭터와 유사해질 수 있습니다. 이는 `1girl`이 학습시의 캡션에 포함되어 있기 때문입니다.

학습 대상의 이미지와 정규화된 이미지를 동시에 학습시킴으로써, class는 그대로 유지되고, identifier를 프롬프트에 붙였을 때에만 학습 대상이 생성되도록 할 수 있습니다.

LoRA나 DreamBooth의 경우 특정 캐릭터만 나오면 되기 때문에 정규화된 이미지를 사용하지 않아도 됩니다.

Textual Inversion의 경우 사용하지 않아도 됩니다 (학습하려는 토큰 문자열이 캡션에 포함되지 않으면 아무것도 학습되지 않기 때문입니다).

정규화된 이미지로는 학습 대상의 모델에서 class 이름만을 사용하여 생성한 이미지를 일반적으로 사용합니다 (예: `1girl`). 그러나 생성된 이미지의 품질이 낮을 경우 프롬프트를 조정하거나, 인터넷에서 별도로 다운로드한 이미지를 사용할 수도 있습니다.

(정규화된 이미지도 학습되므로, 해당 이미지의 품질은 모델에 영향을 미칩니다.)

일반적으로 수백 장 정도를 준비하는 것이 좋습니다 (이미지의 수가 적으면 class 이미지가 일반화되지 않고 해당 특징을 학습하게 됩니다).

생성 이미지를 사용하는 경우, 일반적으로 생성 이미지의 크기를 학습 해상도에 맞추어야 합니다 (더 정확하게는 bucket 해상도, 아래 참조).

## 단계 2. 설정 파일의 기술

텍스트 파일을 생성하고 확장자를 .toml로 설정합니다. 예를 들어 다음과 같이 작성합니다.

(#으로 시작하는 부분은 주석이므로 그대로 복사하여 사용하거나 삭제해도 괜찮습니다.)

```toml
[general]
enable_bucket = true                        # 종횡비 버킷팅 사용 여부

[[datasets]]
resolution = 512                            # 학습 해상도
batch_size = 4                              # 배치 크기

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 학습용 이미지가 있는 폴더 경로 지정
  class_tokens = 'hoge girl'                # 식별자와 클래스 지정
  num_repeats = 10                          # 학습용 이미지 반복 횟수

  # 정규화된 이미지를 사용하는 경우에만 작성합니다. 사용하지 않는 경우 삭제하세요.
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # 정규화된 이미지가 있는 폴더 경로 지정
  class_tokens = 'girl'                     # 클래스 지정
  num_repeats = 1                           # 정규화된 이미지 반복 횟수, 보통 1로 설정
```

기본적으로 아래 항목만 수정하면 학습할 수 있습니다:

1. 학습 해상도

    숫자 1개를 지정하면 정사각형 이미지가 됩니다 (예: 512x512). 대괄호와 쉼표로 구분하여 2개의 숫자를 지정하면 가로×세로 이미지가 됩니다 (예: [512,768]). SD1.x 시리즈의 기본 학습 해상도는 512입니다. [512,768]과 같이 큰 해상도를 지정하면 세로로 긴 또는 가로로 긴 이미지를 생성할 때의 문제를 줄일 수 있습니다. SD2.x 768 시리즈의 경우 768입니다.

1. 배치 크기

    한 번에 학습하는 데이터의 수를 지정합니다. GPU의 VRAM 크기와 학습 해상도에 따라 다릅니다. 자세한 내용은 이후 설명에서 다루도록 하겠습니다. 또한 fine tuning/DreamBooth/LoRA 등에서도 다를 수 있으므로 해당 스크립트의 설명을 참고해주세요.

1. 폴더 지정

    학습용 이미지와 정규화된 이미지(사용하는 경우에만)의 폴더 경로를 지정합니다. 이미지 데이터가 포함된 폴더 자체를 지정합니다.

1. 식별자와 클래스 지정

    앞서 예시에 설명한 대로 식별자와 클래스를 지정합니다.

1. 반복 횟수

    이후에 설명하겠지만, 학습용 이미지의 반복 횟수를 지정합니다.

### 반복 횟수에 대해

반복 횟수는 정규화된 이미지의 수와 학습용 이미지의 수를 조정하기 위해 사용됩니다. 정규화된 이미지의 수가 학습용 이미지보다 많기 때문에, 학습용 이미지를 반복하여 수를 맞추어서 1:1 비율로 학습할 수 있도록 합니다.

반복 횟수는 「__학습용 이미지의 반복 횟수 × 학습용 이미지의 수 ≥ 정규화된 이미지의 반복 횟수 × 정규화된 이미지의 수__」로 지정해야 합니다.

(1 epoch(데이터가 한 바퀴 돌 때 1 epoch)의 데이터 수는 「학습용 이미지의 반복 횟수 × 학습용 이미지의 수」가 됩니다. 정규화된 이미지의 수가 그보다 많으면 남은 정규화된 이미지는 사용되지 않습니다.)

## 단계 3. 학습

각각의 문서를 참고하여 학습을 진행해주세요.

# DreamBooth, 캡션 방식 (정규화된 이미지 사용 가능)

이 방식에서는 각 이미지가 캡션을 통해 학습됩니다.

## 단계 1. 캡션 파일 준비

학습용 이미지 폴더에 이미지와 동일한 파일 이름으로 `.caption` 확장자 (설정에서 변경 가능)를 가진 파일을 둬주세요. 각 파일은 한 줄로 작성되어야 합니다. 인코딩은 `UTF-8`입니다.

## 단계 2. 정규화된 이미지 사용 여부 결정 및 사용 시 정규화된 이미지 생성

class+identifier 형식과 동일합니다. 정규화된 이미지에도 캡션을 추가할 수 있지만, 보통은 필요하지 않을 것입니다.

## 단계 2. 설정 파일 작성

텍스트 파일을 생성하고 확장자를 `.toml`로 지정합니다. 예를 들어 다음과 같이 작성합니다.

```toml
[general]
enable_bucket = true                        # Aspect Ratio Bucketing 사용 여부

[[datasets]]
resolution = 512                            # 학습 해상도
batch_size = 4                              # 배치 크기

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 학습용 이미지가 있는 폴더 경로 지정
  caption_extension = '.caption'            # 캡션 파일의 확장자, .txt를 사용할 경우에는 변경해주세요
  num_repeats = 10                          # 학습용 이미지 반복 횟수

  # 정규화된 이미지를 사용하는 경우에만 작성합니다. 사용하지 않는 경우 삭제하세요.
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # 정규화된 이미지가 있는 폴더 경로 지정
  class_tokens = 'girl'                     # 클래스 지정
  num_repeats = 1                           # 정규화된 이미지 반복 횟수, 일반적으로 1로 설정
```

기본적으로 아래 항목만 수정하면 학습할 수 있습니다. 특별한 언급이 없는 부분은 class+identifier 방식과 동일합니다.

1. 학습 해상도
1. 배치 크기
1. 폴더 지정
1. 캡션 파일의 확장자

    원하는 확장자를 지정할 수 있습니다.
1. 반복 횟수
    
## 단계 3. 학습

학습은 각각의 문서를 참고하여 진행하시면 됩니다. 해당 문서들은 학습에 필요한 자세한 정보와 지침을 제공합니다. 각 문서를 참고하여 모델을 학습시키시면 됩니다.

# Fine Tuning 방식

## 단계 1. 메타데이터 준비

캡션 및 태그를 포함하는 관리용 파일을 메타데이터라고 합니다. 확장자는 `.json`이며, JSON 형식으로 작성합니다. 메타데이터 작성 방법은 길어서 이 문서의 끝에 작성하였습니다.

## 단계 2. 설정 파일 작성

텍스트 파일을 생성하고 확장자를 `.toml`로 지정합니다. 예를 들어 다음과 같이 작성합니다.

```toml
[general]
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = 512                                    # 학습 해상도
batch_size = 4                                      # 배치 크기

  [[datasets.subsets]]
  image_dir = 'C:\piyo'                             # 학습용 이미지가 있는 폴더 경로 지정
  metadata_file = 'C:\piyo\piyo_md.json'            # 메타데이터 파일명
```

기본적으로 아래 항목만 수정하면 학습할 수 있습니다. 특별한 언급이 없는 부분은 DreamBooth, class+identifier 방식과 동일합니다.

1. 학습 해상도
1. 배치 크기
1. 폴더 지정
1. 메타데이터 파일명

    메타데이터 파일을 작성하는 방법은 아래에서 설명하겠습니다.

## 단계 3. 학습
각각의 문서를 참고하여 학습을 진행해주세요. 필요한 정보와 지침은 해당 문서에서 확인하실 수 있습니다.

# 학습에 사용되는 용어 간단 설명

세부 사항은 생략되어 있으며, 저도 완전히 이해하지 못하였기 때문에 자세한 내용은 개별적으로 찾아보시기 바랍니다.

## Fine Tuning (파인 튜닝)

모델을 학습한 후 세밀하게 조정하는 것을 의미합니다. 사용 방법에 따라 의미가 달라지지만, 좁은 의미의 Fine Tuning은 Stable Diffusion에서 모델을 이미지와 캡션으로 학습하는 것을 의미합니다. DreamBooth는 좁은 의미의 Fine Tuning의 특수한 방법 중 하나입니다. 넓은 의미의 Fine Tuning은 LoRA, Textual Inversion, Hypernetworks 등을 포함하여 모델을 학습하는 모든 것을 포괄합니다.

## 스텝 (Step)

간단히 말하면 학습 데이터로 1회 계산하는 것을 1 스텝이라고 합니다. 「학습 데이터의 캡션을 현재 모델에 통과시켜 생성된 이미지와 학습 데이터의 이미지를 비교하여 모델을 약간 변경하여 학습 데이터에 근접하도록 한다」는 것이 1 스텝입니다.

## 배치 사이즈 (Batch Size)

배치 사이즈는 한 번에 계산하는 데이터의 수를 지정하는 값입니다. 한 번에 계산하기 때문에 상대적으로 속도가 향상됩니다. 일반적으로 정확도도 향상된다고 알려져 있습니다.

`배치 사이즈 × 스텝 수`가 학습에 사용되는 데이터의 수가 됩니다. 따라서 배치 사이즈를 늘린 만큼 스텝 수를 줄이는 것이 좋습니다.

(하지만, 예를 들어 "배치 사이즈 1로 1600 스텝"과 "배치 사이즈 4로 400 스텝"은 동일한 결과를 내지 않습니다. 동일한 학습 속도인 경우 일반적으로 후자는 학습이 부족하게 됩니다. 학습 속도를 약간 높이거나 (예: `2e-6` 등), 스텝 수를 500 스텝으로 조정하는 등 조정을 해보세요.)

배치 사이즈를 크게 하면 GPU 메모리를 그만큼 소비합니다. 메모리 부족으로 인해 오류가 발생하거나, 오류가 발생하지 않더라도 학습 속도가 감소할 수 있습니다. 작업 관리자나 `nvidia-smi` 명령을 사용하여 메모리 사용량을 확인하면서 조정하는 것이 좋습니다.

또한, 배치는 「한덩어리의 데이터」 정도로 이해하면 됩니다.

## 학습률 (Learning Rate)

간단히 말하면 1 스텝마다 얼마나 많이 변화시킬지를 나타냅니다. 큰 값을 지정하면 학습이 빠르게 진행되지만, 너무 많이 변하면 모델이 손상되거나 최적의 상태에 도달하지 못할 수 있습니다. 작은 값을 지정하면 학습 속도가 느려지며, 마찬가지로 최적의 상태에 도달하지 못할 수 있습니다.

Fine Tuning, DreamBooth, LoRA 각각에 따라 크게 다르며, 학습 데이터, 원하는 모델, 배치 사이즈, 스텝 수에 따라 달라집니다. 일반적인 값을 기준으로 시작하여 학습 상태를 확인하면서 조절하세요.

기본적으로 학습률은 학습 전체에 걸쳐 고정됩니다. 스케줄러를 지정하여 학습률을 어떻게 변화시킬지 결정할 수 있으며, 이에 따라 결과가 달라질 수 있습니다.

## 에포크 (Epoch)

학습 데이터가 한 번 모두 학습되는 것을 의미합니다. 반복 횟수를 지정한 경우, 해당 반복 이후의 데이터가 한 번 모두 학습되면 1 에포크입니다.

1 에포크의 스텝 수는 기본적으로 `데이터 수 ÷ 배치 사이즈`입니다. 그러나 Aspect Ratio Bucketing을 사용하면 약간 증가합니다 (서로 다른 버킷의 데이터는 동일한 배치에 포함될 수 없으므로 스텝 수가 증가합니다).

## Aspect Ratio Bucketing

Stable Diffusion v1은 512\*512로 학습되어 있지만, 이외에도 256\*1024, 384\*640과 같은 해상도에서도 학습합니다. 이를 통해 잘린 부분이 줄어들고 캡션과 이미지의 관계가 보다 정확하게 학습될 것으로 기대됩니다.

또한 원하는 해상도로 학습할 수 있기 때문에 이미지 데이터의 가로 세로 비율을 사전에 통일시킬 필요가 없어집니다.

설정에서 활성화되며, 상대방이 전환할 수 있지만, 이 문서의 설정 파일 예시에서는 활성화되어 있습니다 (`true`로 설정됨).

학습 해상도는 파라미터로 지정된 해상도의 면적(=메모리 사용량)를 초과하지 않는 범위에서 64 픽셀 단위 (기본값, 변경 가능)

기계 학습에서는 일반적으로 입력 크기를 모두 통일하는 것이 권장됩니다. 하지만 특별한 제약은 없으며, 사실은 동일한 배치 내에서 통일되어 있다면 문제없습니다. NovelAI가 언급하는 bucketing은 사전에 교사 데이터를 종횡비에 따라 학습 해상도별로 분류하는 것을 의미하는 것 같습니다. 그리고 각 버킷 내의 이미지로 배치를 생성하여 배치의 이미지 크기를 통일합니다.

# 이전의 지정 형식 (설정 파일을 사용하지 않고 명령 줄에서 지정)

`.toml` 파일을 지정하지 않고 명령 줄 옵션으로 지정하는 방법입니다. DreamBooth 클래스+식별자 방식, DreamBooth 캡션 방식, fine tuning 방식이 있습니다.

## DreamBooth, class+identifier 방식

폴더 이름으로 반복 횟수를 지정합니다. `train_data_dir` 옵션과 `reg_data_dir` 옵션을 사용합니다.

### step 1. 학습용 이미지 준비

학습용 이미지를 저장할 폴더를 생성합니다. __그리고 해당 폴더 내부에__ 다음과 같은 이름의 디렉토리를 만듭니다.

```
<반복 횟수>_<identifier> <class>
```

사이의 ``_``를 빠뜨리지 않도록 주의하세요.

예를 들어, 「sls frog」 프롬프트로 데이터를 20번 반복하는 경우, 「20_sls frog」가 됩니다. 다음과 같이 됩니다.

![image](https://user-images.githubusercontent.com/52813779/210770636-1c851377-5936-4c15-90b7-8ac8ad6c2074.png)

### 여러 class, 여러 대상 (identifier) 학습

여러 클래스와 대상(identifier)을 동시에 학습하는 경우, 학습용 이미지 폴더 내에 ``반복 횟수_<identifier> <class>`` 폴더를 여러 개, 정규화 이미지 폴더에도 동일하게 ``반복 횟수_<class>`` 폴더를 여러 개 준비하세요.

예를 들어, 「sls frog」와 「cpc rabbit」를 동시에 학습하는 경우, 다음과 같습니다.

![image](https://user-images.githubusercontent.com/52813779/210777933-a22229db-b219-4cd8-83ca-e87320fc4192.png)

클래스가 하나이고 대상이 여러 개인 경우, 정규화 이미지 폴더는 하나만 있어도 됩니다. 예를 들어, 1girl에 캐릭터 A와 캐릭터 B가 있는 경우 다음과 같이 준비합니다.

- train_girls
  - 10_sls 1girl
  - 10_cpc 1girl
- reg_girls
  - 1_1girl

### 단계 2. 정규화된 이미지 준비

정규화된 이미지를 사용하는 경우의 절차입니다.

정규화된 이미지를 저장할 폴더를 생성합니다. __그 폴더 안에__ ``<반복 횟수>_<클래스>``라는 이름의 디렉토리를 생성합니다.

예를 들어, 「frog」라는 프롬프트에서 데이터를 반복하지 않고(한 번만) 사용하는 경우, 다음과 같이 됩니다.

![image](https://user-images.githubusercontent.com/52813779/210770897-329758e5-3675-49f1-b345-c135f1725832.png)

### 단계 3. 학습 실행
각 학습 스크립트를 실행합니다. `--train_data_dir` 옵션을 사용하여 이전에 언급한 학습 데이터 폴더를 (이미지가 포함된 폴더가 아니라 상위 폴더를) 지정하고, `--reg_data_dir` 옵션을 사용하여 정규화된 이미지의 폴더 (이미지가 포함된 폴더가 아니라 상위 폴더)를 지정해주세요.

## DreamBooth, 캡션 방식
학습용 이미지와 정규화된 이미지 폴더에 이미지와 동일한 파일 이름으로 .caption (선택적으로 변경 가능) 확장자의 파일을 놓으면 해당 파일에서 캡션을 읽어와 프롬프트로서 학습합니다.

※ 해당 이미지들의 학습에서는 폴더 이름 (식별자 클래스)은 더 이상 사용되지 않습니다.

캡션 파일의 기본 확장자는 .caption입니다. 학습 스크립트의 `--caption_extension` 옵션을 사용하여 변경할 수 있습니다. `--shuffle_caption` 옵션을 사용하여 학습 중 캡션에 대해 각 부분을 쉼표로 구분하여 셔플하면서 학습할 수 있습니다.

## fine tuning 방식

메타데이터를 생성하는 부분은 설정 파일을 사용하는 경우와 동일합니다. `in_json` 옵션을 사용하여 메타데이터 파일을 지정합니다.

# 학습 중 샘플 출력

학습 중인 모델을 사용하여 이미지를 생성하여 학습 진행 상황을 확인할 수 있습니다. 학습 스크립트에 다음 옵션을 지정합니다.

- `--sample_every_n_steps` / `--sample_every_n_epochs`

    샘플 출력을 수행할 스텝 수 또는 에폭 수를 지정합니다. 이 수마다 샘플을 출력합니다. 둘 다 지정하는 경우, 에폭 수가 우선됩니다.

- `--sample_prompts`

    샘플 출력에 사용할 프롬프트 파일을 지정합니다.

- `--sample_sampler`

    샘플 출력에 사용할 샘플러를 지정합니다.
    'ddim', 'pndm', 'heun', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a' 중 선택할 수 있습니다.

샘플 출력을 수행하려면 미리 프롬프트가 작성된 텍스트 파일을 준비해야 합니다. 한 줄에 하나의 프롬프트를 작성합니다.

예를 들어 다음과 같습니다.

```txt
# prompt 1
masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

`#`로 시작하는 행은 주석입니다. `--n`과 같이 「`--` + 영어 소문자」로 이미지 생성 옵션을 지정할 수 있습니다. 다음 옵션을 사용할 수 있습니다.

- `--n` 다음 옵션까지를 부정적인 프롬프트로 간주합니다.
- `--w` 생성된 이미지의 가로 크기를 지정합니다.
- `--h` 생성된 이미지의 세로 크기를 지정합니다.
- `--d` 생성된 이미지의 시드를 지정합니다.
- `--l` 생성된 이미지의 CFG 스케일을 지정합니다.
- `--s` 생성 시 스텝 수를 지정합니다.

# 각 스크립트에서 공통적으로 사용되는 일반적인 옵션

스크립트가 업데이트되었지만 문서가 따라잡지 못한 경우가 있습니다. 이 경우 `--help` 옵션을 사용하여 사용 가능한 옵션을 확인하십시오.

## 학습에 사용할 모델 지정

- `--v2` / `--v_parameterization`

    학습 대상 모델로 Hugging Face의 stable-diffusion-2-base 또는 해당 모델에서의 fine tuning 모델을 사용하는 경우 (`v2-inference.yaml`을 사용하여 추론하는 모델인 경우) `--v2` 옵션을 사용하고, stable-diffusion-2 또는 768-v-ema.ckpt 및 해당 fine tuning 모델을 사용하는 경우 (`v2-inference-v.yaml`을 사용하여 추론하는 모델인 경우) `--v2` 및 `--v_parameterization` 옵션을 모두 지정하십시오.

    Stable Diffusion 2.0에서는 다음과 같은 주요 변경 사항이 있습니다.

    1. 사용되는 Tokenizer
    2. 사용되는 Text Encoder 및 사용되는 출력 레이어 (2.0은 두 번째 마지막 레이어를 사용)
    3. Text Encoder의 출력 차원 수 (768->1024)
    4. U-Net의 구조 (CrossAttention의 헤드 수 등)
    5. v-parameterization (샘플링 방법이 변경됨)

    이 중 base 모델에서는 1~4가 채택되고, base가 없는 모델 (768-v)에서는 1~5가 채택됩니다. 1~4를 활성화하려면 v2 옵션을, 5를 활성화하려면 v_parameterization 옵션을 사용합니다.

- `--pretrained_model_name_or_path`

    추가 학습을 진행할 기본 모델을 지정합니다. Stable Diffusion의 체크포인트 파일(.ckpt 또는 .safetensors), 로컬 디스크에 있는 Diffusers 모델 디렉토리, Diffusers 모델 ID("stabilityai/stable-diffusion-2" 등)를 지정할 수 있습니다.

## 학습에 관련된 설정

- `--output_dir`

    학습 후 모델을 저장할 폴더를 지정합니다.

- `--output_name`

    모델 파일의 이름을 확장자를 제외하고 지정합니다.

- `--dataset_config`

    데이터셋 설정이 기재된 `.toml` 파일을 지정합니다.

- `--max_train_steps` / `--max_train_epochs`

    학습할 스텝 수나 에폭 수를 지정합니다. 둘 다 지정하는 경우, 에폭 수가 우선됩니다.

- `--mixed_precision`

    메모리 절약을 위해 mixed precision(혼합 정밀도)로 학습합니다. `--mixed_precision="fp16"`과 같이 지정합니다. mixed precision을 사용하지 않을 경우(기본 설정)보다 정확성이 낮을 수 있지만, 필요한 GPU 메모리 양이 크게 줄어듭니다.

    (RTX 30 시리즈 이상의 경우 bf16도 지정할 수 있습니다. 환경 설정 시 accelerate과 일치시켜주세요).

- `--gradient_checkpointing`

    학습 시 가중치 계산을 한 번에 하는 것이 아닌 조금씩 진행하여 필요한 GPU 메모리 양을 줄입니다. 온/오프는 정확성에 영향을 주지 않지만, 켜면 배치 크기를 크게 할 수 있으므로 그 측면에서 영향이 있습니다.

    일반적으로 켜면 속도가 감소하지만, 배치 크기를 크게할 수 있기 때문에 총 학습 시간은 더 빨라질 수 있습니다.

- `--xformers` / `--mem_eff_attn`

    xformers 옵션을 지정하면 xformers의 CrossAttention을 사용합니다. xformers가 설치되어 있지 않거나 오류가 발생하는 경우 (환경에 따라 `mixed_precision="no"`인 경우 등), `mem_eff_attn` 옵션을 지정하여 메모리 효율이 좋은 CrossAttention을 사용할 수 있습니다 (xformers보다 속도가 느립니다).

- `--clip_skip`

    `2`를 지정하면 Text Encoder (CLIP)의 뒤에서 두 번째 레이어의 출력을 사용합니다. 1 또는 옵션을 생략하면 마지막 레이어를 사용합니다.

    ※ SD2.0은 기본적으로 뒤에서 두 번째 레이어를 사용하므로, SD2.0 학습 시에는 지정하지 마십시오.

    대상 모델이 원래 두 번째 레이어를 사용하여 학습된 경우, 2를 지정하는 것이 좋습니다.

    마지막 레이어를 사용하는 경우 모델 전체가 해당 가정으로 학습되었습니다. 따라서 다시 두 번째 레이어를 사용하여 학습하면 원하는 학습 결과를 얻으려면 일정 수의 교사 데이터와 긴 학습 시간이 필요할 수 있습니다.

- `--max_token_length`

    기본값은 75입니다. `150` 또는 `225`를 지정하여 토큰 길이를 확장하여 학습할 수 있습니다. 긴 캡션으로 학습하는 경우에 지정하십시오.

    그러나 학습 시의 토큰 확장 사항은 Automatic1111님의 Web UI와 약간 다를 수 있으므로 (분할 사양 등), 필요하지 않다면 75로 학습하는 것을 권장합니다.

    clip_skip과 마찬가지로 모델의 학습 상태와 다른 길이로 학습하려면 일정 수의 교사 데이터, 긴 학습 시간이 필요할 것으로 생각됩니다.

- `--weighted_captions`

    이 옵션을 지정하면 Automatic1111님의 Web UI와 동일한 가중치가 적용된 캡션을 사용할 수 있습니다. 「Textual Inversion and XTI」 이외의 학습에 사용할 수 있습니다. 캡션뿐만 아니라 DreamBooth 방식의 토큰 문자열에서도 유효합니다.

    가중치가 지정된 캡션의 표기법은 Web UI와 거의 동일하며, (abc) 또는 [abc], (abc:1.23) 등을 사용할 수 있습니다. 중첩도 가능합니다. 괄호 안에 쉼표를 포함하면 괄호의 대응이 잘못되는 프롬프트의 섞기/드롭아웃 때문에 괄호 내에 쉼표를 포함하지 마세요.

- `--persistent_data_loader_workers`

    Windows 환경에서 지정하면 에폭 간 대기 시간이 크게 단축됩니다.

- `--max_data_loader_n_workers`

    데이터 로더 프로세스 수를 지정합니다. 프로세스 수가 많을수록 데이터 로드 속도가 빨라지고 GPU를 효율적으로 사용할 수 있지만, 주 메모리를 소비합니다. 기본값은 「`8` 또는 `CPU 동시 실행 스레드 수-1` 중 작은 값」이므로, 주 메모리 여유가 없거나 GPU 사용률이 90% 이상인 경우 해당 값들을 참고하여 `2` 또는 `1` 정도로 줄여주세요.

- `--logging_dir` / `--log_prefix`

    학습 로그 저장에 관련된 옵션입니다. logging_dir 옵션에 로그 저장 위치 폴더를 지정하세요. TensorBoard 형식의 로그가 저장됩니다.

    예를 들어, --logging_dir=logs로 지정하면 작업 폴더에 logs 폴더가 생성되고, 그 안의 날짜 폴더에 로그가 저장됩니다.
    또한 --log_prefix 옵션을 지정하면 날짜 앞에 지정한 문자열이 추가됩니다. " 「--logging_dir=logs --log_prefix=db_style1_」과 같이 식별용으로 사용할 수 있습니다.

    TensorBoard에서 로그를 확인하려면 별도의 명령 프롬프트를 열고 작업 폴더에서 다음과 같이 입력합니다.

    ```
    tensorboard --logdir=logs
    ```

    (tensorboard는 환경 설정에 맞게 설치되어 있을 것으로 생각되지만, 설치되어 있지 않은 경우 `pip install tensorboard`로 설치하세요.)

    그런 다음 브라우저를 열고 http://localhost:6006/로 이동하면 표시됩니다.

- `--log_with` / `--log_tracker_name`

    학습 로그 저장에 관련된 옵션입니다. `tensorboard` 외에 `wandb`로도 저장할 수 있습니다. 자세한 내용은 [PR#428](https://github.com/kohya-ss/sd-scripts/pull/428)을 참조하세요.

- `--noise_offset`

    이 구현은 다음 기사를 참고하십시오: https://www.crosslabs.org//blog/diffusion-with-offset-noise

    전반적으로 어두운, 밝은 이미지의 생성 결과가 개선될 수 있습니다. LoRA 학습에도 효과적인 것 같습니다. `0.1` 정도의 값을 지정하는 것이 좋습니다.

- `--adaptive_noise_scale` (실험적 옵션)

    Noise offset의 값을 latent의 각 채널의 평균 값의 절댓값에 따라 자동으로 조정하는 옵션입니다. `--noise_offset`와 함께 지정하면 활성화됩니다. Noise offset의 값은 `noise_offset + abs(mean(latents, dim=(2,3))) * adaptive_noise_scale`로 계산됩니다. latent는 정규 분포에 가깝기 때문에 noise_offset의 1/10~동일한 값 정도를 지정하는 것이 좋습니다.

    음수 값을 지정할 수도 있으며, 그 경우 noise offset은 0 이상으로 클리핑됩니다.

- `--multires_noise_iterations` / `--multires_noise_discount`

    Multi resolution noise (피라미드 노이즈)의 설정입니다. 자세한 내용은 [PR#471](https://github.com/kohya-ss/sd-scripts/pull/471)와 이 페이지 [Multi-Resolution Noise for Diffusion Model Training](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2)를 참조하세요.

    `--multires_noise_iterations`에 숫자를 지정하면 활성화됩니다. 6~10 정도의 값을 사용하는 것이 좋습니다. `--multires_noise_discount`에는 0.1~0.3 정도의 값(LoRA 학습 등 비교적 데이터셋이 작은 경우의 PR 작성자의 권장) 또는 0.8 정도의 값(원래 기사의 권장)을 지정하세요 (기본값은 0.3입니다).

- `--debug_dataset`

    이 옵션을 추가하면 학습을 시작하기 전에 어떤 이미지 데이터와 캡션으로 학습되는지 사전에 확인할 수 있습니다. Esc 키를 누르면 종료되고 명령 프롬프트로 돌아갑니다. `S` 키로 다음 단계(배치), `E` 키로 다음 에폭으로 진행합니다.

    ※ Linux 환경(Colab 포함)에서는 이미지가 표시되지 않습니다.

- `--vae`

    vae 옵션에 Stable Diffusion의 체크포인트, VAE의 체크포인트 파일, Diffusers의 모델 또는 VAE(로컬 또는 Hugging Face 모델 ID 지정 가능) 중 하나를 지정하면 해당 VAE를 사용하여 학습합니다(러턴트의 캐시 시 또는 학습 중 러턴트 가져오기 시).

    DreamBooth 및 fine tuning에서는 저장되는 모델이 해당 VAE가 포함된 모델이 됩니다.

- `--cache_latents` / `--cache_latents_to_disk`

    VRAM 사용량을 줄이기 위해 VAE의 출력을 메인 메모리에 캐시합니다. `flip_aug`를 제외한 augmentation을 사용할 수 없게 됩니다. 또한 전체적인 학습 속도가 약간 더 빨라집니다.

    cache_latents_to_disk를 지정하면 캐시를 디스크에 저장합니다. 스크립트를 종료한 다음 다시 시작해도 캐시가 유지됩니다.

- `--min_snr_gamma`

    Min-SNR 가중치 전략을 지정합니다. 자세한 내용은 [여기](https://github.com/kohya-ss/sd-scripts/pull/308)를 참조하세요. 논문에서는 `5`를 권장합니다.

## 모델 저장에 관련된 설정

- `--save_precision`

    모델 저장 시 데이터 정밀도를 지정합니다. float, fp16, bf16 중 하나를 save_precision 옵션에 지정하여 해당 형식으로 모델을 저장합니다 (DreamBooth, fine tuning에서 Diffusers 형식으로 모델을 저장하는 경우에는 사용되지 않습니다). 모델 크기를 줄이고자 할 때 사용할 수 있습니다.

- `--save_every_n_epochs` / `--save_state` / `--resume`

    save_every_n_epochs 옵션에 숫자를 지정하면 해당 에포크마다 학습 중인 모델을 저장합니다.

    save_state 옵션을 함께 지정하면 옵티마이저 등의 상태를 포함한 학습 상태를 함께 저장합니다 (저장된 모델에서도 학습을 재개할 수 있지만, 그에 비해 정확도 향상과 학습 시간 단축이 기대됩니다). 저장 위치는 폴더가 됩니다.

    학습 상태는 저장 위치 폴더에 `<output_name>-??????-state` (??????은 에포크 수)라는 이름의 폴더로 출력됩니다. 장시간의 학습에 사용하십시오.

    저장된 학습 상태에서 학습을 재개하려면 resume 옵션을 사용합니다. 학습 상태 폴더 (`output_dir`가 아닌 해당 폴더의 state 폴더)를 지정하세요.

    또한 Accelerator의 사양에 따라 에포크 수, 전역 단계(global step)는 저장되지 않으며, 재개(resume)할 때에도 1부터 시작됩니다. 이 점 양해 부탁드립니다.

- `--save_every_n_steps`

    save_every_n_steps 옵션에 숫자를 지정하면 해당 스텝마다 학습 중인 모델을 저장합니다. save_every_n_epochs와 함께 지정할 수 있습니다.

- `--save_model_as` (DreamBooth, fine tuning 전용)

    모델 저장 형식을 `ckpt, safetensors, diffusers, diffusers_safetensors` 중에서 선택할 수 있습니다.

    `--save_model_as=safetensors`와 같이 지정합니다. Stable Diffusion 형식(ckpt 또는 safetensors)을 불러와 Diffusers 형식으로 저장하는 경우, 부족한 정보는 Hugging Face에서 v1.5 또는 v2.1의 정보를 가져와 보완합니다.

- `--huggingface_repo_id` 등

    huggingface_repo_id가 지정된 경우 모델 저장 시 동시에 HuggingFace에 업로드합니다. 액세스 토큰 처리에 주의하세요 (HuggingFace 문서를 참조하세요).

    다른 인수를 다음과 같이 지정하세요.

    - `--huggingface_repo_id "your-hf-name/your-model" --huggingface_path_in_repo "path" --huggingface_repo_type model --huggingface_repo_visibility private --huggingface_token hf_YourAccessTokenHere`

    huggingface_repo_visibility에 `public`을 지정하면 리포지토리가 공개됩니다. 생략하거나 `private` (public 외의 값)를 지정하면 비공개됩니다.

    `--save_state` 옵션 지정 시 `--save_state_to_huggingface`를 지정하면 상태도 업로드합니다.

    `--resume` 옵션 지정 시 `--resume_from_huggingface`를 지정하면 HuggingFace에서 상태를 다운로드하여 재개합니다. 이때 --resume 옵션은 `--resume {repo_id}/{path_in_repo}:{revision}:{repo_type}`로 지정됩니다.

    예: `--resume_from_huggingface --resume your-hf-name/your-model/path/test-000002-state:main:model`

    `--async_upload` 옵션을 지정하면 업로드를 비동기적으로 수행합니다.

## 옵티마이저 관련 설정

- `--optimizer_type`
    -- 옵티마이저의 종류를 지정합니다. 다음 옵션을 사용할 수 있습니다:
    - AdamW: [torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
    - 이전 버전의 옵션 미지정과 동일
    - AdamW8bit: 인수는 동일
    - 이전 버전의 --use_8bit_adam 지정과 동일
    - Lion: https://github.com/lucidrains/lion-pytorch
    - 이전 버전의 --use_lion_optimizer 지정과 동일
    - Lion8bit: 인수는 동일
    - SGDNesterov:  [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html), nesterov=True
    - SGDNesterov8bit: 인수는 동일
    - DAdaptation(DAdaptAdamPreprint): https://github.com/facebookresearch/dadaptation
    - DAdaptAdam: 인수는 동일
    - DAdaptAdaGrad: 인수는 동일
    - DAdaptAdan: 인수는 동일
    - DAdaptAdanIP: 인수는 동일
    - DAdaptLion: 인수는 동일
    - DAdaptSGD: 인수는 동일
    - AdaFactor: [Transformers AdaFactor](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
    - 사용자 정의 옵티마이저

- `--learning_rate`

학습률을 지정합니다. 적절한 학습률은 학습 스크립트에 따라 다르므로 해당 스크립트의 설명을 참조하십시오.

- `--lr_scheduler` / `--lr_warmup_steps` / `--lr_scheduler_num_cycles` / `--lr_scheduler_power`

    학습률 스케줄러 관련 설정입니다.

    lr_scheduler 옵션을 사용하여 linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup 또는 사용자 정의 스케줄러 중 하나를 선택할 수 있습니다. 기본값은 constant입니다.

    lr_warmup_steps는 스케줄러의 워밍업 (점점 학습률을 증가시키는 과정) 스텝 수를 지정할 수 있습니다.

    lr_scheduler_num_cycles는 cosine with restarts 스케줄러의 리스타트 횟수이고, lr_scheduler_power는 polynomial 스케줄러의 polynomial power입니다.

    자세한 내용은 해당 스케줄러에 대한 자료를 참조하십시오.

    사용자 정의 스케줄러를 사용하는 경우 `--scheduler_args`를 사용하여 옵션 인수를 지정하십시오.

### 옵티마이저 지정에 대해

옵티마이저의 옵션 인수는 --optimizer_args 옵션을 사용하여 지정하십시오. key=value 형식으로 여러 값을 지정할 수 있습니다. 또한 값은 쉼표로 구분하여 여러 값을 지정할 수 있습니다. 예를 들어 AdamW 옵티마이저에 인수를 지정하는 경우 ``--optimizer_args weight_decay=0.01 betas=.9,.999``와 같습니다.

옵션 인수를 지정하는 경우 각각의 옵티마이저 사양을 확인하십시오.

일부 옵티마이저에는 필수 인수가 있으며, 생략하면 자동으로 추가됩니다(SGDNesterov의 momentum 등). 콘솔 출력을 확인하십시오.

D-Adaptation 옵티마이저는 학습률을 자동으로 조정합니다. 학습률 옵션에 지정한 값은 학습률 그 자체가 아니라 D-Adaptation이 결정한 학습률 적용 비율이므로 일반적으로 1.0을 지정하십시오. 텍스트 인코더에 U-Net의 절반 학습률을 지정하려면 ``--text_encoder_lr=0.5 --unet_lr=1.0``와 같이 지정합니다.

AdaFactor 옵티마이저는 relative_step=True를 지정하면 학습률을 자동으로 조정할 수 있습니다(생략 시 기본적으로 추가됩니다). 자동 조정하는 경우 학습률 스케줄러로 adafactor_scheduler가 강제로 사용됩니다. 또한 scale_parameter와 warmup_init을 지정하는 것이 좋습니다.

자동 조정하는 경우 옵션 지정은 예를 들어 ``--optimizer_args "relative_step=True" "scale_parameter=True" "warmup_init=True"``와 같습니다.

학습률을 자동 조정하지 않는 경우 옵션 인수에 ``relative_step=False``를 추가하십시오. 이 경우 학습률 스케줄러로 constant_with_warmup가 사용되며, 그래디언트 클리핑을 하지 않는 것이 권장되므로 인수는 ``--optimizer_type=adafactor --optimizer_args "relative_step=False" --lr_scheduler="constant_with_warmup" --max_grad_norm=0.0``와 같습니다.

### 임의의 옵티마이저를 사용하는 방법

``torch.optim``의 옵티마이저를 사용하는 경우 클래스 이름만 지정하십시오 (``--optimizer_type=RMSprop`` 등). 다른 모듈의 옵티마이저를 사용하는 경우 "모듈명.클래스명"을 지정하십시오 (``--optimizer_type=bitsandbytes.optim.lamb.LAMB`` 등).

(내부적으로는 importlib만 수행되며 동작은 확인되지 않았습니다. 필요한 경우 해당 패키지를 설치하십시오.)


<!-- 
## 任意サイズの画像での学習 --resolution
正方形以外で学習できます。resolutionに「448,640」のように「幅,高さ」で指定してください。幅と高さは64で割り切れる必要があります。学習用画像、正則化画像のサイズを合わせてください。

個人的には縦長の画像を生成することが多いため「448,640」などで学習することもあります。

## Aspect Ratio Bucketing --enable_bucket / --min_bucket_reso / --max_bucket_reso
enable_bucketオプションを指定すると有効になります。Stable Diffusionは512x512で学習されていますが、それに加えて256x768や384x640といった解像度でも学習します。

このオプションを指定した場合は、学習用画像、正則化画像を特定の解像度に統一する必要はありません。いくつかの解像度（アスペクト比）から最適なものを選び、その解像度で学習します。
解像度は64ピクセル単位のため、元画像とアスペクト比が完全に一致しない場合がありますが、その場合は、はみ出した部分がわずかにトリミングされます。

解像度の最小サイズをmin_bucket_resoオプションで、最大サイズをmax_bucket_resoで指定できます。デフォルトはそれぞれ256、1024です。
たとえば最小サイズに384を指定すると、256x1024や320x768などの解像度は使わなくなります。
解像度を768x768のように大きくした場合、最大サイズに1280などを指定しても良いかもしれません。

なおAspect Ratio Bucketingを有効にするときには、正則化画像についても、学習用画像と似た傾向の様々な解像度を用意した方がいいかもしれません。

（ひとつのバッチ内の画像が学習用画像、正則化画像に偏らなくなるため。そこまで大きな影響はないと思いますが……。）

## augmentation --color_aug / --flip_aug
augmentationは学習時に動的にデータを変化させることで、モデルの性能を上げる手法です。color_augで色合いを微妙に変えつつ、flip_augで左右反転をしつつ、学習します。

動的にデータを変化させるため、cache_latentsオプションと同時に指定できません。


## 勾配をfp16とした学習（実験的機能） --full_fp16
full_fp16オプションを指定すると勾配を通常のfloat32からfloat16（fp16）に変更して学習します（mixed precisionではなく完全なfp16学習になるようです）。
これによりSD1.xの512x512サイズでは8GB未満、SD2.xの512x512サイズで12GB未満のVRAM使用量で学習できるようです。

あらかじめaccelerate configでfp16を指定し、オプションで ``mixed_precision="fp16"`` としてください（bf16では動作しません）。

メモリ使用量を最小化するためには、xformers、use_8bit_adam、cache_latents、gradient_checkpointingの各オプションを指定し、train_batch_sizeを1としてください。

（余裕があるようならtrain_batch_sizeを段階的に増やすと若干精度が上がるはずです。）

PyTorchのソースにパッチを当てて無理やり実現しています（PyTorch 1.12.1と1.13.0で確認）。精度はかなり落ちますし、途中で学習失敗する確率も高くなります。
学習率やステップ数の設定もシビアなようです。それらを認識したうえで自己責任でお使いください。

-->

# 메타데이터 파일 생성

## 학습 데이터 준비

학습하고자 하는 이미지 데이터를 준비하고 원하는 폴더에 넣어주세요.

예를 들어, 다음과 같이 이미지를 저장합니다.

![학습 데이터 폴더 스크린샷](https://user-images.githubusercontent.com/52813779/208907739-8e89d5fa-6ca8-4b60-8927-f484d2a9ae04.png)

## 자동 캡션 생성

캡션 없이 태그만을 사용하여 학습하는 경우, 이 부분은 건너뛰시면 됩니다.

수동으로 캡션을 준비하는 경우, 캡션은 학습 데이터 이미지와 동일한 디렉토리에 동일한 파일명과 확장자 ".caption" 등으로 준비해주세요. 각 파일은 한 줄의 텍스트 파일로 구성됩니다.

## BLIP를 사용한 캡션 생성

최신 버전에서는 BLIP의 다운로드, 가중치 다운로드, 가상 환경 추가가 필요하지 않습니다. 그대로 작동합니다.

finetune 폴더 내의 make_captions.py를 실행합니다.

```
python finetune\make_captions.py --batch_size <배치 사이즈> <학습 데이터 폴더>
```

배치 사이즈를 8로 설정하고 학습 데이터를 상위 폴더의 train_data에 넣은 경우, 다음과 같이 실행합니다.

```
python finetune\make_captions.py --batch_size 8 ..\train_data
```

캡션 파일은 학습 데이터 이미지와 동일한 디렉토리에 동일한 파일명과 확장자 .caption으로 생성됩니다.

batch_size는 GPU의 VRAM 용량에 맞게 조정해주세요. 더 큰 값일수록 속도가 빨라집니다 (VRAM 12GB의 경우 더 늘릴 수 있을 것입니다).
max_length 옵션으로 캡션의 최대 길이를 지정할 수 있습니다. 기본값은 75입니다. 모델을 225 토큰 길이로 학습하는 경우 더 길게 설정하는 것이 좋을 수 있습니다.
caption_extension 옵션으로 캡션의 확장자를 변경할 수 있습니다. 기본값은 .caption입니다 (.txt로 변경하면 아래에서 설명하는 DeepDanbooru와 충돌합니다).

여러 개의 학습 데이터 폴더가 있는 경우, 각각의 폴더에 대해 실행해주세요.

추론에는 랜덤성이 있기 때문에 실행할 때마다 결과가 달라집니다. 결과를 고정하려면 --seed 옵션으로 `--seed 42`와 같이 난수 시드를 지정해주세요.

기타 옵션은 `--help`로 도움말을 참조해주세요 (파라미터의 의미에 대한 문서화가 잘 정리되어 있지 않아 소스 코드를 확인해야 합니다).

기본적으로 .caption 확장자로 캡션 파일이 생성됩니다.

![caption이 생성된 폴더](https://user-images.githubusercontent.com/52813779/208908845-48a9d36c-f6ee-4dae-af71-9ab462d1459e.png)

다음과 같이 이미지와 함께 캡션이 부여됩니다.

![캡션과 이미지](https://user-images.githubusercontent.com/52813779/208908947-af936957-5d73-4339-b6c8-945a52857373.png)

## DeepDanbooru를 사용하여 태그를 부여합니다.

danbooru 태그의 태깅 작업 자체를 수행하지 않을 경우 「캡션과 태그 정보의 전처리」로 이동하십시오.

태그 부여 작업은 DeepDanbooru 또는 WD14Tagger로 수행할 수 있습니다. WD14Tagger가 더 정확성이 높아 보입니다. WD14Tagger를 사용하여 태그를 부여하려면 다음 장으로 넘어가주십시오.

### 환경 설정

작업 폴더에 DeepDanbooru (https://github.com/KichangKim/DeepDanbooru)를 클론하거나 zip 파일을 다운로드하여 풀어주세요. 저는 zip으로 풀었습니다.
또한 DeepDanbooru의 릴리스 페이지(https://github.com/KichangKim/DeepDanbooru/releases)에서 "DeepDanbooru Pretrained Model v3-20211112-sgd-e28"의 Assets에서 deepdanbooru-v3-20211112-sgd-e28.zip을 다운로드하여 DeepDanbooru 폴더에 풀어주세요.

아래에서 다운로드합니다. Assets를 클릭하여 열고 거기에서 다운로드합니다.

![DeepDanbooru 다운로드 페이지](https://user-images.githubusercontent.com/52813779/208909417-10e597df-7085-41ee-bd06-3e856a1339df.png)

다음과 같은 디렉토리 구조를 생성해주세요.

![DeepDanbooru의 디렉토리 구조](https://user-images.githubusercontent.com/52813779/208909486-38935d8b-8dc6-43f1-84d3-fef99bc471aa.png)

Diffusers에 필요한 환경을 위해 다음 명령을 사용하여 라이브러리를 설치합니다. DeepDanbooru 폴더로 이동하여 설치합니다(실제로는 tensorflow-io가 추가될 것입니다).

```
pip install -r requirements.txt
```

그런 다음 DeepDanbooru 자체를 설치합니다.

```
pip install .
```

태그 부여 환경 설정이 완료되었습니다.

### 태그 부여 실행

DeepDanbooru 폴더로 이동한 후, deepdanbooru를 실행하여 태그 부여 작업을 수행합니다.

```
deepdanbooru evaluate <학습 데이터 폴더> --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

만약 학습 데이터를 상위 폴더인 train_data에 위치시켰다면, 다음과 같은 명령어를 사용합니다.

```
deepdanbooru evaluate ../train_data --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

태그 파일은 학습 데이터 이미지와 동일한 디렉토리에 동일한 파일 이름과 확장자인 .txt로 생성됩니다. 개별 이미지를 하나씩 처리하기 때문에 상대적으로 속도가 느릴 수 있습니다.

여러 개의 학습 데이터 폴더가 있는 경우, 각각의 폴더에 대해 위의 명령어를 실행하십시오.

아래와 같은 형식으로 파일이 생성됩니다.

![DeepDanbooru 생성 파일](https://user-images.githubusercontent.com/52813779/208909855-d21b9c98-f2d3-4283-8238-5b0e5aad6691.png)

다음과 같이 이미지에 태그가 부여됩니다. (정말로 많은 정보입니다...)

![DeepDanbooru 태그와 이미지](https://user-images.githubusercontent.com/52813779/208909908-a7920174-266e-48d5-aaef-940aba709519.png)

## WD14Tagger를 사용한 태그 부여
WD14Tagger를 사용하여 태그 부여를 진행하는 방법에 대해 설명하겠습니다.

WD14Tagger는 Automatic1111님의 WebUI에서 사용되는 태거입니다. 해당 github 페이지 (https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger)의 정보를 참고하였습니다.

첫 번째로 필요한 환경 설정은 이미 설치되어 있는 모듈입니다. 또한 가중치는 Hugging Face로부터 자동으로 다운로드됩니다.

### 태그 부여 실행

스크립트를 실행하여 태그 부여를 진행합니다.

```
python tag_images_by_wd14_tagger.py --batch_size <배치 크기> <학습 데이터 폴더>
```

학습 데이터를 상위 폴더인 train_data에 위치시켰다면, 다음과 같이 입력합니다.

```
python tag_images_by_wd14_tagger.py --batch_size 4 ..\train_data
```

최초 실행 시, 모델 파일은 wd14_tagger_model 폴더에 자동으로 다운로드됩니다 (폴더는 옵션으로 변경 가능). 다음과 같습니다.

![다운로드된 파일](https://user-images.githubusercontent.com/52813779/208910447-f7eb0582-90d6-49d3-a666-2b508c7d1842.png)

태그 파일은 학습 데이터 이미지와 동일한 디렉토리에 동일한 파일 이름과 확장자인 .txt로 생성됩니다.

![생성된 태그 파일](https://user-images.githubusercontent.com/52813779/208910534-ea514373-1185-4b7d-9ae3-61eb50bc294e.png)

![태그와 이미지](https://user-images.githubusercontent.com/52813779/208910599-29070c15-7639-474f-b3e4-06bd5a3df29e.png)

thresh 옵션을 사용하여, 태그의 confidence (확신도)가 얼마 이상인 경우에 태그를 부여할지 지정할 수 있습니다. 기본값은 WD14Tagger의 샘플과 동일한 0.35입니다. 값이 낮을수록 더 많은 태그가 부여되지만 정확도가 낮아집니다.

batch_size는 GPU의 VRAM 용량에 따라 조정하십시오. 크기가 클수록 속도가 빨라집니다 (VRAM 12GB에서도 더 크게 설정할 수 있습니다). caption_extension 옵션을 사용하여 태그 파일의 확장자를 변경할 수 있습니다. 기본값은 .txt입니다.

model_dir 옵션을 사용하여 모델을 저장할 폴더를 지정할 수 있습니다.

또한 force_download 옵션을 사용하면 저장할 폴더가 있더라도 모델을 다시 다운로드합니다.

여러 개의 학습 데이터 폴더가 있는 경우, 각각의 폴더에 대해 위의 명령어를 실행하십시오.

## 캡션 및 태그 정보의 전처리

스크립트에서 처리하기 쉽도록 캡션과 태그를 메타데이터로 통합하여 하나의 파일에 저장합니다.

### 캡션의 전처리

캡션을 메타데이터에 포함시키려면 작업 폴더에서 다음을 실행하십시오 (학습에 캡션을 사용하지 않을 경우 실행할 필요가 없습니다) (실제로는 한 줄로 작성합니다. 아래도 동일합니다). `--full_path` 옵션을 사용하여 이미지 파일의 위치를 전체 경로로 메타데이터에 저장합니다. 이 옵션을 생략하면 상대 경로로 기록되지만, 폴더 지정은 별도로 `.toml` 파일 내에서 필요합니다.

```
python merge_captions_to_metadata.py --full_path <학습 데이터 폴더>
    --in_json <로드할 메타데이터 파일명> <메타데이터 파일명>
```

메타데이터 파일명은 임의의 이름입니다.
학습 데이터가 train_data이고, 로드할 메타데이터 파일이 없고, 메타데이터 파일이 meta_cap.json인 경우, 다음과 같이 입력합니다.

```
python merge_captions_to_metadata.py --full_path train_data meta_cap.json
```

caption_extension 옵션을 사용하여 캡션의 확장자를 지정할 수 있습니다.

여러 개의 학습 데이터 폴더가 있는 경우, full_path 인수를 지정하고 각각의 폴더에 대해 실행하십시오.

```
python merge_captions_to_metadata.py --full_path 
    train_data1 meta_cap1.json
python merge_captions_to_metadata.py --full_path --in_json meta_cap1.json 
    train_data2 meta_cap2.json
```

in_json을 생략하면 기록 대상 메타데이터 파일이 있으면 해당 파일에서 로드하고 덮어쓰기를 수행합니다.

__※ in_json 옵션과 대상 파일을 매번 변경하여 다른 메타데이터 파일로 저장하는 것이 안전합니다.__

### 태그의 전처리

태그도 마찬가지로 메타데이터로 통합합니다 (학습에 태그를 사용하지 않을 경우 실행할 필요가 없습니다).

```
python merge_dd_tags_to_metadata.py --full_path <학습 데이터 폴더>
    --in_json <로드할 메타데이터 파일명> <메타데이터 파일명>
```

같은 디렉토리 구조에서 meta_cap.json을 읽고 meta_cap_dd.json에 저장하는 경우, 다음과 같이 입력합니다.

```
python merge_dd_tags_to_metadata.py --full_path train_data --in_json meta_cap.json meta_cap_dd.json
```

여러 개의 학습 데이터 폴더가 있는 경우, full_path 인수를 지정하고 각각의 폴더에 대해 실행하십시오.

```
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap2.json
    train_data1 meta_cap_dd1.json
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap_dd1.json 
    train_data2 meta_cap_dd2.json
```

in_json을 생략하면 기록 대상 메타데이터 파일이 있으면 해당 파일에서 로드하고 덮어쓰기를 수행합니다.

__※ in_json 옵션과 대상 파일을 매번 변경하여 다른 메타데이터 파일로 저장하는 것이 안전합니다.__

### 캡션 및 태그 정리

지금까지 메타데이터 파일에 캡션과 DeepDanbooru의 태그가 통합되었습니다. 그러나 자동 캡션 생성의 경우 캡션에 표기 차이 등이 있어 적절하지 않을 수 있으며 (※), 태그에는 언더스코어(_)가 포함되거나 등급이 부여될 수도 있습니다 (DeepDanbooru의 경우). 따라서 에디터의 치환 기능 등을 사용하여 캡션과 태그를 정리하는 것이 좋습니다.

※ 예를 들어 애니메이션 소녀를 학습하는 경우, 캡션에는 girl/girls/woman/women과 같은 다양한 표현이 있을 수 있습니다. 또한 「anime girl」과 같은 표현도 단순히 「girl」로 정리하는 것이 적절할 수 있습니다.

정리를 위한 스크립트가 준비되어 있으므로, 상황에 맞게 스크립트 내용을 편집하여 사용하시기 바랍니다.

(학습 데이터 폴더의 지정은 더 이상 필요하지 않습니다. 메타데이터 내의 모든 데이터를 정리합니다.)

```
python clean_captions_and_tags.py <로드할 메타데이터 파일명> <메타데이터 파일명>
```

--in_json 옵션이 필요하지 않으므로 유의하십시오. 예를 들어 다음과 같이 사용할 수 있습니다.

```
python clean_captions_and_tags.py meta_cap_dd.json meta_clean.json
```

위 단계로 캡션 및 태그의 전처리가 완료되었습니다.

## latents 사전 획득

※ 이 단계는 필수적인 단계는 아닙니다. 생략하고 학습 중에 latents를 획득하면서 학습할 수도 있습니다.
또한 `random_crop`나 `color_aug`와 같은 학습 중에 이미지를 매번 변경하는 경우에는 latents 사전 획득을 할 수 없습니다. (이미지를 매번 변경하면서 학습하기 때문입니다.) 사전 획득을 하지 않는 경우, 지금까지의 메타데이터로 학습할 수 있습니다.

미리 이미지의 잠재 표현(latents)을 가져와 디스크에 저장합니다. 이를 통해 학습을 더 빠르게 진행할 수 있습니다. 동시에 bucketing(학습 데이터를 종횡비에 따라 분류)을 수행합니다.

작업 폴더에서 다음과 같이 입력하세요.

```
python prepare_buckets_latents.py --full_path <학습 데이터 폴더> 
    <로드할 메타데이터 파일명> <저장할 메타데이터 파일명> 
    <파인튜닝할 모델명 또는 체크포인트> 
    --batch_size <배치 크기> 
    --max_resolution <해상도 가로,세로> 
    --mixed_precision <정밀도>
```

모델이 model.ckpt이고, 배치 크기가 4이며, 학습 해상도가 512\*512이고, 정밀도가 no(float32)이고, meta_clean.json에서 메타데이터를 읽고, meta_lat.json에 저장하는 경우, 다음과 같이 입력합니다.

```
python prepare_buckets_latents.py --full_path 
    train_data meta_clean.json meta_lat.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no
```

학습 데이터 폴더에는 numpy의 npz 형식으로 latents가 저장됩니다.

최소 해상도는 --min_bucket_reso 옵션으로 설정할 수 있으며, 최대 해상도는 --max_bucket_reso로 지정할 수 있습니다. 기본값은 각각 256과 1024입니다. 예를 들어 최소 해상도로 384를 지정하면, 256\*1024나 320\*768과 같은 해상도는 사용하지 않게 됩니다.
해상도를 768\*768과 같이 크게 설정한 경우, 최대 해상도로 1280과 같은 값을 지정하는 것이 좋습니다.

--flip_aug 옵션을 지정하면 좌우 반전 augmentation(데이터 증강)을 수행합니다. 데이터 양을 가상적으로 두 배로 늘릴 수 있지만, 데이터가 좌우 대칭이 아닌 경우에 지정하면 (예: 캐릭터의 외모, 헤어 스타일 등) 학습이 제대로 진행되지 않을 수 있습니다.

(반전된 이미지에 대해서도 latents를 획득하고, \*\_flip.npz 파일을 저장하는 단순한 구현입니다. fline_tune.py에는 특별한 옵션 지정이 필요하지 않습니다. \_flip이 포함된 파일이 있는 경우, flip 포함 및 미포함 파일을 무작위로 로드합니다.)

배치 크기는 VRAM 12GB에서도 약간 늘릴 수 있습니다.
해상도는 64로 나누어 떨어지는 숫자로 "가로,세로"로 지정합니다. 해상도는 파인튜닝 시 메모리 크기에 직접적으로 영향을 미칩니다. VRAM 12GB에서는 512,512가 한계로 보입니다(※). 16GB이라면 512,704 또는 512,768까지 올릴 수 있을 것입니다. 또한 256,256으로 설정하더라도 VRAM 8GB에서는 어려워 보입니다(파라미터와 optimizer 등은 해상도와 관련 없이 일정한 메모리가 필요하기 때문입니다).

※ 배치 크기 1로 학습시 12GB VRAM, 640,640에서 작동하는 보고도 있습니다.

다음과 같이 bucketing 결과가 표시됩니다.

![bucketing 결과](https://user-images.githubusercontent.com/52813779/208911419-71c00fbb-2ce6-49d5-89b5-b78d7715e441.png)

학습 데이터 폴더에 여러 개의 폴더가 있는 경우, full_path 인수를 지정하면서 각각의 폴더에 대해 실행하세요.

```
python prepare_buckets_latents.py --full_path  
    train_data1 meta_clean.json meta_lat1.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

python prepare_buckets_latents.py --full_path 
    train_data2 meta_lat1.json meta_lat2.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no
```

로드할 메타데이터 파일과 저장할 메타데이터 파일을 동일하게 설정할 수도 있지만, 따로 설정하는 것이 안전합니다.

__※인수를 각각 변경하여 다른 메타데이터 파일에 저장하면 안전합니다.__