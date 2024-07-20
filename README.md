# JAGAN

# A implementation for the paper "CGFTrans: Cross-modal Global Feature Fusion Transformer for Medical Report Generation"

## Citation
```
@inproceedings{chen2021cross,
  title={Cross-modal Memory Networks for Radiology Report Generation},
  author={Chen, Zhihong and Shen, Yaling and Song, Yan and Wan, Xiang},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={5904--5914},
  year={2021}
 }
 
@inproceedings{chen2020generating,
  title={Generating Radiology Reports via Memory-driven Transformer},
  author={Chen, Zhihong and Song, Yan and Chang, Tsung-Hui and Wan, Xiang},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={1439--1449},
  year={2020}
}

@article{xu2024cgftrans,
  title={CGFTrans: Cross-Modal Global Feature Fusion Transformer for Medical Report Generation},
  author={Xu, Liming and Tang, Quan and Zheng, Bochuan and Lv, Jiancheng and Li, Weisheng and Zeng, Xianhua},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```

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
- modules: 
    - the layer define of our model 
    - dataloader
    - loss function
    - metrics
    - tokenizer
    - some utils
- pycocoevalcap: Microsoft COCO Caption Evaluation Tools

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

## Reference codes:
- https://github.com/cuhksz-nlp/R2Genhttps://github.com/cuhksz-nlp/R2Gen
- https://github.com/cuhksz-nlp/R2GenCMNhttps://github.com/cuhksz-nlp/R2GenCMN
- The evaluation metrics are from pycocoevalcap: https://github.com/tylin/coco-caption.
