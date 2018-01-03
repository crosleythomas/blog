---
layout: default
title: Data Preparation

---

# Part 3: Consuming Data

<div style="text-align: center">
	<a href='https://www.tensorflow.org/programmers_guide/datasets' target="_blank">Relevant TensorFlow Documentation</a><br>
	<a href="https://github.com/crosleythomas/tensorplates/blob/master/templates/high_level_api.ipynb" target="_blank">Associated TensorPlates Template</a><br>
	<a href=".zip" >Sample TFRecord Files</a>
</div>

Great, now we have data formatted as TFRecord files and ready to be ingested by a TensorFlow graph.

To feed data into the compute graph we are making next, we will use the <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset" target="_blank">Dataset</a>
 and <a href="https://www.tensorflow.org/api_docs/python/tf/data/Iterator" target="_blank">Iterator</a>
 APIs.  <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset" target="_blank">Dataset</a>
 makes creating input pipelines much simpler (loading, shuffling, transforming, etc.) and <a href="https://www.tensorflow.org/api_docs/python/tf/data/Iterator" target="_blank">Iterator</a> handles feeding data from a Dataset into a graph.

When developing a new model, and in the <a href="https://github.com/crosleythomas/tensorplates/blob/master/templates/high_level_api.ipynb" target="_blank">associated template</a> for training a model, I go through the following steps:
* Gathering saved data
* Create a dataset_input_fn
* Constructing the Dataset
* Define the parse function

## Imports needed for the rest of Part 3
```
import tensorflow as tf
import glob
```

## Gathering Saved Data

Building off our previous example, let's assume we have one directory of training TFRecords and one directory of validation TFRecords.

Your code will vary slightly here depending on exactly how you stored data.

```
train_dir = 'data/train'
valid_dir = 'data/valid'

train_files = glob.glob(train_dir + '/*.tfrecord')
valid_files = glob.glob(valid_dir + '/*.tfrecord')

# Now make all these files accessible depending on whether we
#    are in training ('train') or validation ('valid') mode.
data_files = {'train' : train_files, 'valid' : valid_files}

```

## Creating a dataset_input_fn
Since we are constrained to using a <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#make_one_shot_iterator" target="_blank">one_shot_iterator</a> with Estimators instead of an <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#make_initializable_iterator" target="_blank">initializable_iterator</a>, which is useful for switching between train, validation, and test, I abstract away the dataset creating one level.  As we will see later, this is useful when switching from training to evaluation.  This function pretty much stays the same across projects.

```
'''
Constructs a Dataset, Iterator, and returns handles that will be called when
	Estimator requires a new batch of data. This function will be passed into 
	Estimator as the input_fn argument.

	Inputs:
		mode: string specifying whether to take the inputs from training or validation data

	Outputs:
		features: the columns of feature input returned from a dataset iterator
		labels: the columns of labels for training return from a dataset iterator
'''
def dataset_input_fn(mode):
	# Function that does the heavy lifting for constructing a Dataset
	#    depending on the current mode of execution
	dataset = load_dataset(mode)

	# Making an iterator that runs from start to finish once
	#    (the preferred type for Estimators)
	iterator = dataset.make_one_shot_iterator()

	# Consuming the next batch of data from our iterator
	features, labels = iterator.get_next()

	return features, labels
```

## Constructing the Dataset

Now it's time to create the <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset" target="_blank">Dataset</a> object by filling in the ```load_dataset``` function we just called.

```
'''
Loads and does all processing for a portion of the dataset specified by 'mode'.

	Inputs:
		mode: string specifying whether to take the inputs from training or validation data

	Outputs:
		dataset: the Dataset object constructed for this mode and pre-processed
'''
def load_dataset(mode):
	# Taking either the train or validation files from the dictionary we constructed above
	files = data_files[mode]

	# Created a Dataset from our list of TFRecord files
	dataset = tf.data.TFRecordDataset(files)

	# Apply any processing we need on each example in our dataset.  We
	#    will define parse next.  num_parallel_calls decides how many records
	#    to apply the parse function to at a time (change this based on your
	#    machine).
	dataset = dataset.map(parse, num_parallel_calls=2)

	# Shuffle the data if training, for validation it is not necessary
	# buffer_size determines how large the buffer of records we will shuffle
	#    is, (larger is better!) but be wary of your memory capacity.
	if mode == 'train':
		dataset = dataset.shuffle(buffer_size=1000)

	# Batch the data - you can pick a batch size, maybe 32, and later
	#    we will include this in a dictionary of other hyper parameters.
	dataset = dataset.batch(hyper_params.batch_size)

```

<span class='waiting'><b>Awaiting Explanation: </b>It's not clear to me if the order of the function calls in parse matters, i.e. map, batch, shuffle.  Calling a map function that loads jpeg images from a string to a full image on all, say 10,000, images would certainly be a bad idea.  It might seem like you want to call batch first to get a small batch and then call map.  However, the TensorFlow examples all show map being called right after the Dataset object is first made and I don't see a massive slowdown from overloaded memory if I call map first with images.</span>


## Defining the Parse Function
The last thing we need to do is fill in the parse function we called.  There are several reasons to need a parse function including:

* Extracting features from the TFRecord
* Loading the raw image data if your dataset contains file paths to images
* Casting data to types required by TensorFlow (i.e. from uint8 to float32 for convolutional layers)
* Performing data normalization
* Performing data augmentation
* etc.


In general you need to define a function with the following signature:

```
'''
The function that will be call by the map function of a dataset.

	Inputs:
		record: a single record from the Dataset.

	Outputs:
		inputs: a dictionary of input features indexed by the string name of the feature
		label: the label for this record
'''
def parser(record):
	
	###########################
	###    Your Code Here   ###
	###########################

	return inputs, label

```

When using a TFRecordDataset it will typically look like this:

```
def parser(record):
	# Define the features to be parsed out of each example.
	features={
		'feature_0_name': tf.FixedLenFeature([], tf.int64), # data type will vary
		'feature_1_name': tf.FixedLenFeature([], tf.int64),
		...
		'feature_N_name': tf.FixedLenFeature([], tf.int64),
	}

	# Parse the features out of this one record we were passed
	parsed = tf.parse_single_example(record, features)

	###########################################################
	###   Your pre-processing and/or formatting code here   ###
	###########################################################

	return inputs, label
```

We will use the following for our running example:
```
def parser(record):
	# Define the features to be parsed out of each example.
	#    You should recognize this from when we wrote the TFRecord files!
	features={
		'height': tf.FixedLenFeature([], tf.int64),
		'width': tf.FixedLenFeature([], tf.int64),
		'depth': tf.FixedLenFeature([], tf.int64),
		'image': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64)
	}

	# Parse the features out of this one record we were passed
	parsed = tf.parse_single_example(record, features)

	# Format the data
	H = tf.cast(parsed['height'], tf.int32)
	W = tf.cast(parsed['width'], tf.int32)
	D = tf.cast(parsed['width'], tf.int32)
	image = tf.cast(tf.reshape(tf.image.decode_jpeg(parsed["image"]), [H, W, D]), tf.float32)
	label = tf.cast(parsed['label'], tf.int32)
	
	# TODO: Check if this works or do we need to hardcode shape?
	image.set_shape([H, W, D])

	return {'image': image}, label
```

Awesome, we now have the whole data pipeline set up and are ready to make the TensorFlow model.

<hr>
## Continue Reading

<button onclick="location.href='model'" class='continue-links'>Continue to Part 4</button>
In Part 4 we will build a model to use this data!

<hr>
<div style="text-align: center;">
	<button onclick="location.href='introduction'" class='continue-links'>Introduction</button>
	<button onclick="location.href='setup'" class='continue-links'>Part 1: Setup</button>
	<button onclick="location.href='dataprep'" class='continue-links'>Part 2: Preparing Data</button>
	<button onclick="location.href='dataload'" class='continue-links'>Part 3: Consuming Data</button>
	<button onclick="location.href='model'" class='continue-links'>Part 4: Defining a Model</button>
	<button onclick="location.href='traineval'" class='continue-links'>Part 5: Training and Evaluating</button>
	<button onclick="location.href='deploy'" class='continue-links'>Part 6: Exporting, Testing, and Deploying</button>
	<button onclick="location.href='summary'" class='continue-links'>Part 7: All Together Now</button>
	<button onclick="location.href='references'" class='continue-links'>Part 8: Furthur Reading and References</button>
</div>

