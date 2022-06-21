# 3D-SPAN
## Environments
- python                             3.8.3
- torch                              1.7.0              

## Dataset
The 3D-SPAD dataset we proposed is preprocessed and stored in the JSON file and placed in a folder named data. If you want to visualize, you can visit this [webpage](https://github.com/Mayibo-cuc/3D-SPAD) to get the original 3D scene data.

Data structure(Json)：
  -graph:
    -scene_name: Scene ID.
    -zero_num: Number of nodes with empty label. (from the end)
    -feature: Node eigenvector matrix.
    -adj: adjacency matrix.
    -gmm: Gaussian mixture model output probability matrix.
    -supported_mask: Invalid matrix.
    -zero_mask: Invalid matrix.
    -evaluate: Object label.
    -scene_score: Scene label.

## Training and Test 
To train the 3D-SPAN framework on 3D-SPAD dataset, please run:

1、`python train.py` 

2、`python train_scene.py` 

To test the trained model, please run the following code:

1、`python test.py` 

2、`python test_scene.py` 
