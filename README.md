### This repo contains the tensorflow implementation of the paper "Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks".

### Dependencies
* TensorFlow <= 0.12
* Download the kb files from here <do you guys have a public url to host the Kbs, otherwise I will ask my sysadmin>

### Data
The processed data is stored in data_formatted/ directory

## Training
I have set up training with most default params on a very small dataset so that it is easier to get started. Just running the script should work.
```
/bin/bash run.sh ./config.sh
```

### Data Processing
Several data processing options are implented in the scripts below.
```
/bin/bash run_data.sh ./config_data.sh
```

### Model outputs
We are also releasing the output predictions of our model for comparison. Find them in the model_outputs directory.

### Pretrained model
We are also releasing our trained model. It can be downloaded here <ask Sysadmin for public link>

