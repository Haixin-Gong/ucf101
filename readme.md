# UCF101 Action Recognition

This project implements action recognition models on the UCF101 dataset using deep learning approaches.

## Dataset Preparation

1. Download the UCF101 dataset from [Kaggle](https://www.kaggle.com/api/v1/datasets/download/matthewjansen/ucf101-action-recognition)
2. Extract the downloaded file
3. Organize the data in the following structure:

```
.
└── UCF101
    ├── test
    ├── test.csv
    ├── train
    ├── train.csv
    ├── val
    └── val.csv
```

## Environment Setup

- Create and activate a conda environment:

```bash
conda create -n HFAI python=3.9
conda activate HFAI
cd ./ucf101
pip install -r requirements.txt
```

- Configuration

    Open `config.py` and modify the following parameters:
   - `DATASET_PATH`: Path to your UCF101 dataset
   - `MODEL_TYPE`: Choose between available model architectures (`3dcnn` or `efficientnet`)
   - `NUM_CLASSES`: Number of action classes to classify (default: 10)
   - Other trainning parameters interesting you

## Training and Evaluation

- Train

    ```bash
    python train.py
    ```

- Evaluation

    ```bash
    python evaluate.py
    ```

- Results

    The results will be saved in the folder `./results/{MODEL_TYPE}_{NUM_CLASSES}`, which include:

    - Model weights
    - Curves of training and validation loss and accuracy 
    - Confusion matrix for evaluation

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Citation
If you use this code in your research, please cite the following paper:
```
@article{soomro2012ucf101,
  title={UCF101: A dataset of 101 human actions classes from videos in the wild},
  author={Soomro, K},
  journal={arXiv preprint arXiv:1212.0402},
  year={2012}
}
```
```
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}

```

## Acknowledgements

This project makes use of TensorFlow, an open-source machine learning framework. We would like to thank the TensorFlow team and contributors for their hard work and dedication in developing and maintaining this powerful tool. For more information about TensorFlow, please visit the [TensorFlow website](https://www.tensorflow.org/).
