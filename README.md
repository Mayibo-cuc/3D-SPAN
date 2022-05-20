# 3D-SPAN
## Environments
- python                             3.8.3
- torch                              1.7.0
- matplotlib                         3.2.2                 

## Dataset
We referenced original 3D scene data from the SUNCG dataset. Please see the [webpage](http://suncg.cs.princeton.edu) and [paper](https://arxiv.org/pdf/1611.08974v1.pdf) for more details about the data. The 3D-SPAD dataset we proposed is preprocessed and stored in the JSON file and placed in a folder named data. If you want to visualize, you can visit this [webpage](https://github.com/Mayibo-cuc/3D-SPAD) to get the original 3D scene data.

## Training and Test 
To train the 3D-SPAN framework on 3D-SPAD dataset, please run:

1、`python train.py` 

2、`python train_scene.py` 

To test the trained model, please run the following code:

1、`python test.py` 

2、`python test_scene.py` 
