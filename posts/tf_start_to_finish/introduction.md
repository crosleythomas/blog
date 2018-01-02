---
layout: default
title: TensorFlow from Start to Finish

---

# Start to Finish Guide for Developing TensorFlow Models

Implementation and engineering are becoming increasingly important parts of machine learning as data gets bigger and the competition between researchers and companies heats up.  Being able to quickly iterate on ideas is both necessary to keep up and makes machine learning a lot more fun.

This guide focuses on how to implement projects in [TensorFlow](tensorflow.org).  TensorFlow is Google's "open-source software library for Machine Intelligence".

TensorFlow is a great tool for everyone in machine learning but I often myself 
1. Repeating a lot of boiler plate code.
2. Searching for undocumented features and best practices across several blogs.
3. Finding an article for the correct topic but it is already out of date.
4. Not knowing which option to pick when there are several ways of implementing something.

Here, I share a start-to-finish guide for taking your project from a messy set of data to a trained production model.  Additionally, I'm planning on updating in place - when one of my practices changes because I improved my approach or TensorFlow has released a new version, I'll update the guide here.

There are also templates from my related project [TensorPlates](https://github.com/crosleythomas/tensorplates) to get a new project going faster, pro-tips from little tricks I've learned along the way, and explanations for un-or-under-documented features.

The explanations here assume prior knowledge of TensorFlow basics, python, and machine learning.

### * [Part 1 -- Setup](setup)
Configuring your development environment for TensorFlow development.

### * [Part 2 -- Preparing Data](dataprep)
Taking your 20GB worth of jpeg images saved in 50 different directories and converting them into TensorFlow's preferred data format - [TFRecords](https://www.tensorflow.org/versions/master/api_docs/python/tf/data/TFRecordDataset).

### * [Part 3 -- Consuming Data](dataload)
Creating a data [Iterator](https://www.tensorflow.org/api_docs/python/tf/data/Iterator) that reads TFRecords, performs any pre-processing, and feeds data into your compute graph.

### * [Part 4 -- Defining a Model](model)
Defining your model as a TensorFlow [Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) -- part of the new high-level API that handles some of the details of train/eval/test logic, logging results, and exporting your saved model.

### * [Part 5 -- Training and Evaluating](traineval)
This section covers the train/eval loop with Estimators as well as watching training with [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).  I also share tips on how to do this when training remotely.

### * [Part 6 -- Exporting, Testing, and Deploying](deploy)
Exporting your trained model, testing on held-out data, and deploying that model for a service.

### * [Part 7 -- All Together Now](summary)
Now that you understand every step of the process, let's review and show how to put them all together quickly using templates that handle most of the boiler plate code.

### * [Part 8 -- Furthur Reading and References](references)
Links to articles I've found most useful, TensorFlow documentation, video guides, etc.
<br>
You can find all the associated templates <a href="https://github.com/crosleythomas/tensorplates" target="_blank">here</a>.
<button onclick="location.href='https://github.com/crosleythomas/tensorplates'" class='continue-links' target="_blank">TensorPlates Templates Project</button>
	
<hr>
<div style='text-align: center'><b>Thank you for reading!  Comments and questions are appreciated <a href="https://github.com/crosleythomas/blog/issues">here</a>.</b></div>
<hr>
<div style='text-align: center'><b>Contributors:</b><a href="http://github.com/crosleythomas"> Thomas Crosley</a></div>

