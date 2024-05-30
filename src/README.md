# DM2F-Net-ViT

By Peng Kaixin 21307140077, Fudan University.

## Installation & Preparation

Make sure you have `Python>=3.7` installed on your machine.

**Environment setup:**

1. Create conda environment

       conda create -n dm2f
       conda activate dm2f

2. Install dependencies (test with PyTorch 1.8.0):

   1. Install pytorch==1.8.0 torchvision==0.9.0 (via conda, recommend).

   2. Install other dependencies

          pip install -r requirements.txt

* Prepare the dataset

   * Download the RESIDE dataset from the [official webpage](https://sites.google.com/site/boyilics/website-builder/reside).

   * Download the O-Haze dataset from the [official webpage](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/).

   * Make a directory `./data` and create a symbolic link for uncompressed data, e.g., `./data/RESIDE`.

## Training

1. ~~Set the path of pretrained ResNeXt model in resnext/config.py~~
2. Set the path of datasets in tools/config.py
3. Run by ```python train.py``` and ```python train_ohaze.py```

~~The pretrained ResNeXt model is ported from the [official](https://github.com/facebookresearch/ResNeXt) torch version,
using the [convertor](https://github.com/clcarwin/convert_torch_to_pytorch) provided by clcarwin. 
You can directly [download](https://drive.google.com/open?id=1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ) the pretrained model ported by me.~~

Use pretrained ResNeXt (resnext101_32x8d) from torchvision.

Use pretrained Vision Transformer (vit_base_patch16_224) from timm.

*Hyper-parameters* of training were set at the top of *train.py*, and you can conveniently
change them as you need.

Training a model on a single ~~GTX 1080Ti~~ L20 GPU takes about ~~4~~ 3 hours.

## Testing

1. Set the path of five benchmark datasets in tools/config.py.
2. Put the trained model in `./ckpt/`.
2. Run by ```python test.py```

*Settings* of testing were set at the top of `test.py`, and you can conveniently
change them as you need.

## License

DM2F-Net is released under the [MIT license](LICENSE).

