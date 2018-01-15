---
layout: default
title: Exporting, Testing, and Deploying

---

# Part 6: Exporting, Testing, and Deploying

## Exporting
One of the nice things about using <a href="https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator" target="_blank">Estimator</a> is that it handles the basic export for us.  Estimator saves checkpoints of all learnable weights so we can load them back in later.  The clunkiest part of saving/restoring models in TensorFlow is keeping the graph structure and graph weights (checkpoints) together.

An Estimator can't be restored from just a checkpoint file (since it does not contain the structure of the model!) so we also need to save the structure.  TensorFlow does save a ```graph.pbtxt``` file with Estimator, but there doesn't seem to be a way to load the Estimator back in with it.  You still have to define ```model_fn``` when constructing a ```tf.estimator.Estimator``` so we will need to load back in ```model.py``` eventually.  

The best way I've found to load back in the model for simple test cases or visualization is to import the ```model_fn``` an create an Estimator like we did during training.  This is why we copied over the ```model.py``` file and now we pass the ```output_dir``` with the path to our saved checkpoints.

Exporting your model in production adds another level of complexity and is something I haven't had to do yet.  See <a href="https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators" target="_blank">here</a> for details on exporting for production or wait for the next update to this page.

## Testing
Your testing might come in all different forms such as:
* Evaluating on a final test set
* A visualization that relies on predictions from the model
* Production web service

Evaluating on a test set is easy.  Once again, just use the <a href="https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#evaluate" target="_blank">evaluate</a> function the same way we did in the training/validation loop.  This time you just swap out the final testing set.

For a visualization or service that relies on the model you will probably want to use the <a href="https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#predict" target="_blank">predict</a> function which is also part of estimator.  

Predict is one of the modes we specified in the ```model_fn``` of our Estimator.  If you remember, we have the following:

```
# 2. Generate predictions
if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {'output': h}
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
```

When we call the predict function of Estimator, mode will get passed as ```tf.estimator.ModeKeys.PREDICT``` so the predictions are computed and then immediately returned.

The most import thing we need to use ```predict``` is an ```input_fn```.  For a lot of use cases, such as visualizations, creating a Dataset and Iterator might be overkill and ill-suited if we don't have prepared data.

Another option is to use <a href="https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn">tf.estimator.inputs.numpy_input_fn</a>.  This functions returns ```features``` and ```labels``` like a Dataset Iterator so we can feed them into an Estimator.

<span class="example">The code below is one example of construction a simple ```input_fn``` to use with model we have.  You can test the code in a file ```test.py``` in the ```model_dir``` of your saved model.  Also, use <a href="images/test.jpg">this test image</a>.</span>

```
######################################
###   Inside <model_dir>/test.py   ###
######################################

import tensorflow as tf
from model import model_fn, params

# Load a sample image
image = imageio.imread('test.jpg')
image = image[0:200,0:300,0:3] # a little manual crop

model_dir = ...

# Load the model
estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, params=params)

# Create the input_fn
input_fn = tf.estimator.inputs.numpy_input_fn(x={'image' : image}, num_epochs=1, shuffle=False)

# Predict!
print(estimator.predict(input_fn=input_fn))

```

## Deploying
Deploying a TensorFlow model in production is something I haven't had to do yet.  Hopefully more on this later but for now you should be able to find good documentation in the <a href="https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators" target="_blank">Programmer's Guide</a> and under <a href="https://www.tensorflow.org/deploy/" target="_blank">TensorFlow Deploy</a>.

<hr>
## Continue Reading

<button onclick="location.href='summary'" class='continue-links'>Continue to Part 7</button>
Part 7 summarizes all the pieces so far and how to put them all together.

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