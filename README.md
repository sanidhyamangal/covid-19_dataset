# covid-19
This repo constitues of two different types of CNN following transerfer learning architecture, implemented in Tensorflow. The only difference between both the iterations of the models is, in one the base model of imagenet is freezed and is not allowed to be trained, but in other cases the model is allowed to be trained after 100th layer.  

This is a repo which contains X-Ray images of COVID-19 Datset from  ieee8023/covid-chestxray-dataset. The provided repo is an updated source of data from the cohen, it is structured in such a way that it can be directly feeded into any of the provided neural network using tensorflow's `ImageGenerator` and flow from directory feature.


### Requirememts
```txt
tensorflow >= 2.0.0
```

### How to run the project
For trainable base model:
```shell
python main.py
```


For non trainable base model:
```shell
python main_non.py
```

### Visualizing the training results

```shell
tensorboard --logdir="."
```

### Credit for the dataset goes to:
```
@article{cohen2020covid,
  title={COVID-19 image data collection},
  author={Joseph Paul Cohen and Paul Morrison and Lan Dao},
  journal={arXiv 2003.11597},
  url={https://github.com/ieee8023/covid-chestxray-dataset},
  year={2020}
}
```
