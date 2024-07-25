# JAGAN

The implmentation for the paper "JAGAN: Joint Attention GAN for Medical Report Generation with Clinical Style Preservation"

## Table of Contents

- [Pre-requisites](#pre-requisites)
- [Setup](#setup)
- [Running the System](#running-the-system)
- [Training the Models](#training-the-models)
- [Testing the System](#testing-the-system)
- [Converting Predictions to Sentences](#converting-predictions-to-sentences)
- [Reference](#reference)
- [Notes](#notes)

## Pre-requisites

Ensure you have Python and the necessary libraries installed. You may need packages such as `torch`, `numpy`, `PIL`, `jieba`, and others.

## Setup

1. Clone the repository or download the code to your local machine.
2. Install the required Python packages.

## Data Sets and Additional Components

- Download IU X-ray, MIMIC-CXR and LGK dataset, and place in `images` directory within the project folder.
  - [Download IU X-ray Dataset](https://iuhealth.org/find-medical-services/x-rays)
  - [Download MIMIC-CXR- Dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
  - [Download LGK Dataset](https://pan.quark.cn/s/e9cf4c649b8f) Extraction code：LNTL

- Place your image data sets in the `images` directory within the project folder. The system expects the images to be organized in a specific structure that it can recognize. 
  
- The `cococaption` folder contains necessary code components that are vital for the system's operation. It has been made available for separate download:
  - [Download cococaption Code Component](https://pan.quark.cn/s/88a390069059) Extraction code：2Y18

## Running the System

1. **Extract Image Text**
   - Run `getreports.py` to extract the text of the images.

2. **Preprocess Data**
   - Run `preprocess.py` to obtain training and validation features.

3. **Move Features**
   - Manually move all features to the designated location. ```./processed_data/LGK/feature```

4. **Split Data**
   - Run `./processed_data/cider_cache.py` to split the dataset.

5. **Pre-train the generator with MLE**
   - Run `train_mle.py` to train the Diagnostic Report Generator.

6. **Pre-train the discriminator**
   - Run `predis.py`. Ensure to provide the name of one of the checkpoints created during the `train_mle.py` run.

7. **Fine-tune the evaluator, generator, and discriminator**
   - Run `train_pg.py`. Provide the names of checkpoints for both the generator and discriminator as command-line arguments.

8. **Compare with Transformer-based Structure**
   - To train with Transformer structure, save the training results and use the configuration files `configtrans.yml` and `configtranspg.yml`.

## Testing the System

- Run `test.py` to test the dataset with the trained models.

## Converting Predictions to Sentences

- Run `code2word.py` to convert utf-8 encoded prediction results into sentences.

## References

- [Generating Radiology Reports via Memory-driven Transformer](https://github.com/cuhksz-nlp/R2Gen)
- [Improving Image Captioning with Conditional Generative Adversarial Nets](https://github.com/beckamchen/ImageCaptionGAN)
- [Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization](https://github.com/dbmptr/EPOSearch)
- [Pareto Multi-Task Learning](https://github.com/Xi-L/ParetoMTL)
- [Cocoevalcap evaluation metrics](https://github.com/tylin/coco-caption)

## Notes

- Ensure that all command-line arguments are correctly specified when running the scripts.
- Check the documentation for each script for additional usage information and options.
