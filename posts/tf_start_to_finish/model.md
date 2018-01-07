---
layout: default
title: Defining a Model

---

# Part 4: Defining a Model

<div style="text-align: center">
    <a href='https://www.tensorflow.org/programmers_guide/estimators' target="_blank">TensorFlow Programmer's Guide -  Estimator</a><br>
    <a href="https://www.tensorflow.org/extend/estimators" target="_blank">TensorFlow Extend - Estimators</a><br>
</div>

Now for the part you've all been waiting for - defining a model.

To build the model we are going to use one of TensorFlow's new (as of version 1.1) <a href="https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator" target="_blank">Estimator API</a>.  Estimators accept as arguments the things we do want to handle ourselves: 
* a ```model_fn``` that defines the model's logic
* a dictionary of hyperparameters that define things like the learning rate, dropout rate, etc.)

and abstracts away the things that should be automatic:
* writing TensorBoard logs
* saving checkpoints
* looping over batches of data
* exporting a servable model

To create an Estimator we follow these steps:
1. Define the Estimator object
2. Construct the model's logic in a function model_fn
3. Set parameters such as model_dir, config, and params

<span class="protip"><b>Tip: </b>I like to keep this model code in a separate python file, let's call it ```model.py```, that is imported from the main training script.  Since research usually involves trying several architecture variants, separate model files help keep all the subtle differences in order.  Later, we will see how to copy over the exact code that is used for each training run to make completely reproducible results.</span>

To follow along with the running example, create a new file called ```model.py```.

## Defining the Estimator object
Inside ```model.py``` I add one addition function ```load_estimator``` which handles both loading a new model (if ```model_dir``` is not speicified) and loads the Estimator's most recent checkpoint found in ```model_dir``` if it is specified.  This is a nice abstraction to have if you are restarting training or have another script, say a visualization demo, where you want to load the trained model.

<span class="protip"><b>Tip: </b>When an Estimator is initialized, it looks in ```model_dir``` and uses the latest saved checkpoint if it exists.  You create a new model by passing ```model_dir=None```.</span>

```
'''
Constructs and returns a TensorFlow Estimator with the model function and parameters provided.  
If model_dir is specified and an existing checkpoint exists, the Estimator is initialized from the  
most recent checkpoint.  Otherwise, a new Estimator is initialized.

    Inputs:
        model_fn: function handle to function that defines the logic of the model
        model_dir: output directory where checkpoints and summary statistics will be stored 
        config: RunConfig object that contains runtime variables 
            (i.e. random seed, checkpoint frquency, etc.)
        params: dictionary of hyperparameters that need to be accessible inside model_fn
    Outputs:
        TensorFlow Estimator object that encapsulates your model
'''
def load_estimator(model_fn=None, model_dir=None, config=None, params=None):
    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=config, params=params)
```

## Constructing the Model Function
Now, let's define the core logic of the model, ```model_fn```.  

This function is called when:
* <b>Predicting:</b> predictions are computed and then immediately returned
* <b>Evaluating:</b> predictions are made and evaluation metrics are computed but no step is taken with the optimizer
* <b>Training:</b> predictions are made, evaluation metrics are computed, and optimization is performed

For all projects we do this in a function with the following signature:

```
'''
Defines the model function passed into tf.estimator.  
This function defines the computational logic for the model.

Implementation:
    1. Define the model's computations with TensorFlow operations
    2. Generate predictions and return a prediction EstimatorSpec
    3. Define the loss function for training and evaluation
    4. Define the training operation and optimizer
    5. Return loss, train_op, eval_metric_ops in an EstimatorSpec

    Inputs:
        features: A dict containing the features passed to the model via input_fn
        labels: A Tensor containing the labels passed to the model via input_fn
        mode: One of the following tf.estimator.ModeKeys string values indicating
               the context in which the model_fn was invoked 
                  - tf.estimator.ModeKeys.TRAIN ---> model.train()
                  - tf.estimator.ModeKeys.EVAL, ---> model.evaluate()
                  - tf.estimator.ModeKeys.PREDICT -> model.predict()

    Outputs:
        tf.EstimatorSpec that defines the model in different modes.
'''
def model(features, labels, mode, params):
    # 1. Define model structure
    
    # ...
    # convolutions, denses, and batch norms, oh my!
    # ...

    # 2. Generate predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'output_str': output_var} # alter this dictionary for your model
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # 3. Define the loss functions
    loss = ...
    
    # 3.1 Additional metrics for monitoring
    eval_metric_ops = {"rmse": tf.metrics.root_mean_squared_error(
          tf.cast(labels, tf.float64), output)}
    
    # 4. Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    
    # 5. Return training/evaluation EstimatorSpec
    return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)
    
```


<span class="warning"><b>Warning:</b> The main gotchya in here is making sure to put your logic in the correct order.  In predict mode you don't have access to the label so you should make predictions first and then return.  Then, after where you will return in predict mode, you can define the loss function and optimizer.</span>

Here is a simple convolutional neural network for our running example:

```
##########################################
###   Inside <project>/code/model.py   ###
##########################################

```

## Setting Additional Parameters
Before we're done, we need to set a few final variables.

* params: a dictionary of model hyperparameters that is accessible in ```model_fn```
* config: a tf.estimator.RunConfig object of runtime parameters
* output_dir: the output directory where Estimator will write summary statistics, training checkpoints, and the graph structure

### params
You may have noticed in the running example ```model_fn``` several arguments being pulled from the params dictionary.  This is a perfect place to store things like the learning rate for your optimizer, the number of layers in part of your network, the number of units in a layer, etc.

I also like defining ```params``` in ```model.py``` since the parameters are logically connected to the model logic and to keep the main training script clean.

For the running example let's use the following:

```
##########################################
###   Inside <project>/code/model.py   ###
##########################################
params = {
    layers = [
        {'num_outputs' : 10, 'kernel_size' : 3, 'stride' : 2, 'activation' : tf.nn.relu, 'regularizer' : tf.nn.l2_loss},
    ],
    learning_rate = 0.001,
    train_epochs = 30
}
```

### config
Not to be confused with ```params```, ```config``` is a <a href="https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig" target="_blank">tf.estimator.RunConfig</a> object that contains parameters that affect the Estimator while it is running such as ```tf_random_seed```, ```save_summary_steps```, ```keep_checkpoint_max```, etc.  Passing config to the Estimator is optional and mostly something you can avoid unless you need something fairly specific.

We don't need this for the running example, but for completeness:

```
config = tf.estimator.RunConfig(
    tf_random_seed=0,
    save_checkpoints_steps=250,
    save_checkpoints_secs=None,
    save_summary_steps=10,
)
```

### output_dir
This one is easy - pick some output directory where you want data related to training this Estimator to be saved.  

For me, these are all in a results directory \<project\>/results/.  Make sure to add ```results/``` to your ```.gitignore```.

I simply name the output directory by a timestamp of when the script is run.  You might modify this when testing different versions of a model (\<project\>/results/v1/..., \<project\>/results/v2/..., etc).

```
################################################
###   Inside <project>/code/train_model.py   ###
################################################

import time, datetime

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

output_dir = '../results/' + timestamp
```

## Putting it All Together
Now that everything we need is defined, we can create the Estimator from the main training script.
```
################################################
###   Inside <project>/code/train_model.py   ###
################################################

from model import load_estimator, model_fn, params

estimator = load_estimator(model_fn=model_fn, output_dir=output_dir, config=config, params=params)
```

## Improving Reproducibility
For better reproducibility I use the following lines to copy over the main training script and the model file that defines the Estimator.  This makes everything much more straight-forward when comparing several slightly varying architectures. Technically you can look up the exact architecture of the model you ran in the <em>Graph</em> tab of TensorBoard, but I'll take the bet you'd rather take a quick peek at the python file you wrote than dig 4 levels into the TensorBoard graph visualization.

```
################################################
###   Inside <project>/code/train_model.py   ###
################################################

import os, shutil

# Find the path of the current running file (train script)
curr_path = os.path.realpath(__file__)

# Define the path of your factored out model.py file
model_file = '/some/path/model.py'

# Now copy the training script and the model file to 
#   output_dir -- the same directory specified when creating the Estimator
# Note: copy over more files if there are other important dependencies.
shutil.copy(curr_path, output_dir)
shutil.copy(model_path, output_dir)

```


<span class="protip"><b>Tip: </b>If you are using Jupyter Notebooks (which you should be!), calling ```tf.reset_default_graph()``` before initializing your model is a good practice.  Doing this avoids creating extra variables and naming confusions.  One of your cells may look like: </span>
```
tf.reset_default_graph()
estimator = load_estimator(...)
```

And that's it!  The data is ready, the model is fully defined, and we are ready to start training.

<hr>
## Continue Reading

<button onclick="location.href='traineval'" class='continue-links'>Continue to Part 5</button>
In Part 5 we will train and evaluate the Estimator.

<hr>
<div style="text-align: center;">
    <button onclick="location.href='introduction'" class='continue-links'>Introduction</button>
    <button onclick="location.href='setup'" class='continue-links'>Part 1: Setup</button>
    <button onclick="location.href='dataprep'" class='continue-links'>Part 2: Preparing Data</button>
    <button onclick="location.href='dataload'" class='continue-links'>Part 3: Consuming Data</button>
    <button onclick="location.href='model'" class='continue-links'>Part 4: Defining a Model</button>
    <button onclick="location.href='traineval'" class='continue-links'>Part 5: Training and Evaluating</button>
    <button onclick="location.href='export'" class='continue-links'>Part 6: Exporting, Testing, and Deploying</button>
    <button onclick="location.href='summary'" class='continue-links'>Part 7: All Together Now</button>
    <button onclick="location.href='references'" class='continue-links'>Part 8: Furthur Reading and References</button>
</div>
