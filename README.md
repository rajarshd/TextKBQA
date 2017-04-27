### This repo contains the tensorflow implementation of the paper "Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks".

### Dependencies
* TensorFlow <= 0.12

## Training
I have set up training with most default params on a very small dataset so that it is easier to get started. Just running the script should work.
```
/bin/bash run.sh ./config.sh
```
### Data
The processed data (train/dev/test split) is stored in data_formatted/ directory.
To download the KB files used for the project run,
```
sh get_data.sh
```
After downloading the data, you will have to change the appropriate entries in the config.sh file (kb_file and text_kb_file).


### Model outputs
We are also releasing the output predictions of our model for comparison. Find them in the model_outputs directory.

### Pretrained model
We are also releasing our trained model. It can be downloaded here.

