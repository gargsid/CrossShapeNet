# CrossShapeNet

This repository contains the implementation of Cross-Shape Attention on top of the point-cloud features extracted from MID-FC architecture that led to state-of-the-art results on the PartNet Semantic Shape Segmentation benchmark. Please checkout the following links for details on our work. 

## Extracting PartNet features

To extract the features from the MID-Net (Octree based HRNet) architecture, first setup the Tensorflow implementation of the original O-CNN repo from [here](https://github.com/Microsoft/O-CNN). Specifically see [Installation](https://github.com/microsoft/O-CNN/blob/master/docs/installation.md) and set up [Octree](https://github.com/microsoft/O-CNN/blob/master/docs/installation.md) and [Tensorflow code](https://github.com/microsoft/O-CNN/blob/master/docs/installation.md)

## Training CrossShapeNet

### Self-Shape Attention 

First, the architecture is trained with conventional self-attention to generate meaningfull dense point-wise representation to help sample the most relevant shapes for Cross-Attention. 

To run the training for single-shape (eg Bed, Bottle, Chair, ....), use the following command

```
python run_training.py --logs_dir='PATH_TO_LOGS' --attention_type='ssa' --start=PART_INDEX --end=PART_INDEX --n_heads=1 --batch_size=4 --lr=0.001 --cmd
```

For more arguments please check the [run_training.py](https://github.com/gargsid/CrossShapeNet/blob/main/run_training.py).

Here:
- `logs_dir`: Path to the folder that will store the trained model and logs. Folder will be created if not already present
- `attention_type`: 'ssa' for self-attention and 'csa' for cross-attention
- `start`: starting index in the range of categories for which we want to run the training. Please check [this](https://github.com/gargsid/CrossShapeNet/blob/988c1c480e1b5fb221b0521757fa00244dde3731/run_training.py#L7). 
- `end`: ending inde in the range of categories for which we want to run the training. So start=0 and end=0 will the run the training for Bed 
- `cmd`: initiating it will run the training on console
To submit the training to a GPU, please modify [these](https://github.com/gargsid/CrossShapeNet/blob/988c1c480e1b5fb221b0521757fa00244dde3731/run_training.py#L112C14-L125) lines according to your system configurations and replace `--cmd` with `--job` argument. 

**To submit jobs for all the shapes simultaneously use** `start=0` and `end=16` with `--job` flag. 

### Generating KNN graph 

Once the models are trained using self-attention, we might want to precompute the KNN graphs so that they are not computed everytime we initiate cross-shape attention training because some categories like Lamp, Chair takes a large amount of time for graph construction because of large number of point-cloud shapes in each categories. 

To generate the precomputed graphs use

`python run_save_knn.py --ssa_logs_dir=PATH_TO_LOGS_DIR --n_heads=ATTENTION_HEADS --batch_size=4 --num_workers=3 --start=0 --end=16 --job`

Here:
- `ssa_logs_dir`: Name of the directory (excluding the part-name) where self-attention trained models are saved
- `n_heads`: number of self-attention heads used in the training

To submit jobs for particular shapes modify the `start` and `end` flags according to [this](https://github.com/gargsid/CrossShapeNet/blob/988c1c480e1b5fb221b0521757fa00244dde3731/run_training.py#L7). 

To run the construction on console replace `--job` with `--cmd`. Note that this will run the construction for the shapes in range between start to end shapes one by one. 

The final graphs will be saved in the folder: `ssa_logs_dir/knn_graphs/`

### Cross-Shape Attention 



### Result

### Acknowledgements