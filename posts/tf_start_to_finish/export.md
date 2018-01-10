---
layout: default
title: Exporting, Testing, and Deploying

---

# Part 6: Exporting, Testing, and Deploying

## Exporting
One of the nice things about using <a href="https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator" target="_blank">Estimator</a> is that it handles the basic export for us.  Estimator saves checkpoints of all learnable weights so we can load them back in later.  The clunkiest part of saving/restoring models in TensorFlow is keeping the graph structure and graph weights (checkpoints) together.

An Estimator can't be restored from just a checkpoint file (since it does not contain the structure of the model!) so we also need to save the structure.  TensorFlow does save a ..., but ...  The best way I've found to load back in the model for simple test cases or visualization is to import the ```model_fn``` an create an Estimator like we did during training.  This time, however, we pass the ```output_dir``` with the path to our saved checkpoints.

Exporting your model in production adds another level of complexity and is something I haven't had to do yet.  See <a href="https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators" target="_blank">here</a> for details on exporting for production or wait for the next update to this blog.

## Testing

## Deploying
Deploying a TensorFlow model in production is something I haven't had to do yet.  Hopefully more on this later but for now you should be able to find good documentation <a href="https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators" target="_blank">here</a> and <a href="https://www.tensorflow.org/deploy/" target="_blank">here</a>.

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