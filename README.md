
# Large-Scale 3D Representations for Continuous American Sign Language Understanding

This is the official Repo for "Large-Scale 3D Representations for Continuous American Sign Language Understanding". This sepcific codebase is for Sign Language Production.

## Dataset:
Prepare the data, we uploaded the data to HuggingFace. use  ```hugginface-cli``` to download the data to under ```data-bin``` under the main folder. Create one, if not present. The folder should looks like:

```
|--data-bin/
|  |--motions/
|  |  |--<file1.pt>
|  |  |--<file2.pt>
|  |-- texts/
|  |--meta/
|  |  |--top35k_unique.tsv
```


### For OpenASL-35k
```
|--data-bin/
|  |--motions/
|  |  |--<file.train.pkl>
|  |  |--<file.test.pkl>
|  |-- texts/
|  |--meta/
|  |  |--top35k_unique.tsv
```

> Sign Representations can be found here: [Data](https://huggingface.co/datasets/dongludeeplearning/OpenASL3D_Dataset/), ```texts``` folder can be found here: [texts](https://buffalo.box.com/s/5fbomz6rr007dcrvc15ek7uzur13wypl). For OpenASl-35k, [download](https://buffalo.box.com/s/15nanqj5bmw0vnab6ekk0qrmyrge602r). Please ensure to unzip the ```texts``` and ```openasl-35k pickle``` for ruuning



## Train
```
bash run_train.sh
```
> To load the '.pt' files. set ```data_type` == 'pt'``` and to load directly from ```openasl-35k pickle``` set ```load_direct==True``` in ```data_loaders/humanml/data/sign_datasets.py``` and ```data_loaders/humanml/data/signgt_datasets.py```

> Change the ```--save_dir``` to your desired path for saving the trained model checkpoints

> Change the ```data_root``` path in ```data_loaders/humanml/utils/get_opt.py``` to the ```data-bin```

> If loading directly from ```openasl-35k pickle``` set ```save_pkl``` to the path for train and test .pkl 

## Eval
```
bash run_eval.sh
```


## Codebases Reference

> For Sign Language Translation pipeline, please [refer](https://github.com/neccam/slt)

> This repository is adapted from [github](https://github.com/GuyTevet/motion-diffusion-model)
