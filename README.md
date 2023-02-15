# StyleMotion

Our code is based on [Glow](https://github.com/chaiyujin/glow-pytorch/) and [MoGlow](https://github.com/simonalexanderson/StyleGestures).

## Prerequisites
The conda environment defined in 'environment.yml' contains the required dependencies.

## Data & Pretrained Model
Our training data is available [here](https://drive.google.com/file/d/1v_lAIvJAsv334k18gAn8a0tn5zSmx3bD/view?usp=sharing).

Our pretrained model is available [here](https://drive.google.com/file/d/1RRp8ylFDSrD_H_4tXHDKFudqA9uMc95v/view?usp=sharing).

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
