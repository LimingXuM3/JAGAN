# The implementation for the paper "JAGAN: Joint Attention GAN for Medical Report Generation with Clinical Style Preservation"

## Requirements

- `python 3.6`
- `Pytorch >= 1.7`
- `torchvison`
- `opencv-python`

## Data

Download IU and MIMIC-CXR datasets, and place them in `data` folder.

- IU dataset from [here](https://iuhealth.org/find-medical-services/x-rays)
- MIMIC-CXR dataset from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
    
    
## Folder Structure
- data : store dataset
- models: basic model

## Training and Testing
- rename and pre-process the images
```Run imgrename.py```
```Run preprocess.py```
```Run CXRpre.py```

- get the text of the images
```Run getreports.py```

- build MIRQI loss based on TN, FP and FN by matching paired graphs in node-by-node fashion
```Run CEmetric.py```

- pre-train the generator and discriminator
```Run train_mle.py```

- pre-train language style evaluator when one of the checkpoints that were created when train_mle.py was run
```Run pretrain_evaluator.py```

- fine-tune the evaluator, discriminator and generator
```Run train_pg.py```

## Reference codes:
- https://github.com/cuhksz-nlp/R2Genhttps://github.com/cuhksz-nlp/R2Gen
- https://github.com/cuhksz-nlp/R2GenCMNhttps://github.com/cuhksz-nlp/R2GenCMN
- The evaluation metrics are from pycocoevalcap: https://github.com/tylin/coco-caption.
