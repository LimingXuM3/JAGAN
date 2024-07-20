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
- The validation and testing will run after training.
- More options can be found in `main_train` file.
- The model will be trained using command:
```bash train_{$dataset_name}.sh``` 
- You can use the file ```bash test_{$dataset_name}.sh``` to test the model.
    - $dataset_name:
        - iu: IU dataset
        - mimic: MIMIC dataset
- A pre-trained model is available [here](https://pan.quark.cn/s/f81481f4be44
) with 4z5v

## To run the program:
- Run imgrename.py rename the images
- Run getreports.py get the text of the images
- Run preprocess.py
- Run CXRpre.py
- Run cider_cache.py
- Run train_mle.py
- Run pretrain_discriminator.py. ```Make sure to provide the name of one of the checkpoints that were created when train_mle.py was run.```
- Run train_pg.py. ```provide the names of checkpoints for both the generator and discriminator as command-line arguments.```

## Reference codes:
- https://github.com/cuhksz-nlp/R2Genhttps://github.com/cuhksz-nlp/R2Gen
- https://github.com/cuhksz-nlp/R2GenCMNhttps://github.com/cuhksz-nlp/R2GenCMN
- The evaluation metrics are from pycocoevalcap: https://github.com/tylin/coco-caption.
