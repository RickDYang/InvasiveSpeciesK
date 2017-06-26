My implementation for Invasive Spepcies Monitoring in Kaggle.

Based on Keras/Theano

steps to use:

1. Prepare data

put images data into data/train, data/test, data/validation

train_labels.csv in data/train_labels.csv

2. Run 'python3 divide_images.py'

To divide images into 'true/false' sub folders

3. Change code in InvasiveModelsFactory.py

To specify where model to run

4. Run tasks by the following separate steps

'python3 InvasiveTask.py p' - to generate pretrain data by ImageNet pretrained models in Kera

'python3 InvasiveTask.py t' - to train by train data

'python3 InvasiveTask.py e' - to generate submissions

'python3 InvasiveTask.py d' - to check which incorrect prediction in validation data

Will keep improving if I got time.
