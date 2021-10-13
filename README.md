# StyleMotion

Our code is based on [Glow](https://github.com/chaiyujin/glow-pytorch/) and [MoGlow](https://github.com/simonalexanderson/StyleGestures).

## Prerequisites
The conda environment defined in 'environment.yml' contains the required dependencies.

## Data & Pretrained Model
Our training data is available [here](https://drive.google.com/file/d/1tcuEVSaKis263o2Vve-440V7SB5KUIW1/view?usp=sharing).

Our pretrained model is available [here](https://drive.google.com/file/d/1_wE9RX3Xae_9KA1r5kiciHL_Kql8kzCN/view?usp=sharing).

Prepare the environment by: conda env create -f environment.yml 

Download and extract the data and pretained model to the project.

For training:
```
python train.py hparams/locomotion.json locomotion.
```
For style transfer, , and then run:
```
python train.py hparams/locomotion_test.json locomotion
```


## License
Please see the included [LICENSE](LICENSE) file for licenses and citations.
