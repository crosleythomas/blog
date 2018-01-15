---
layout: default
title: Data Preparation

---

# Part 2: Preparing Data

<div style="text-align: center">
    <a href="data/Caltech50.zip" target="_blank">Sample Data</a><br>
    <a href='https://github.com/crosleythomas/tensorplates/blob/master/templates/prepare_tfrecord.ipynb' target="_blank">Associated Template</a><br>
</div>

Now that you're all set up, it's time to get your project off the ground.  The first step is getting data ready to be processed by the model you will make.  Preparing your data and defining your model go hand-in-hand so you may have to go back-and-forth between Parts [2](dataprep), [3](dataload), and [4](model) before you see how it all fits together.

The preferred, yet least documented, format for storing data is in TFRecord files.  TFRecord files (\*.tfrecord) store data as records in a binary format.

Let's see how to take data and write it as TFRecord files.

## Writing Data to TFRecords

To convert your data to TFRecords, you will follow these steps:
1. Gather a datapoint (e.g. an image and corresponding label)
2. Create a [TFRecordWriter](https://www.tensorflow.org/api_docs/python/tf/python_io/TFRecordWriter)
3. Create an [Example](https://www.tensorflow.org/api_docs/python/tf/train/Example) out of [Features](https://www.tensorflow.org/api_docs/python/tf/train/Feature) constructed from your datapoint
4. Serialize the [Example](https://www.tensorflow.org/api_docs/python/tf/train/Example)
5. Write the [Example](https://www.tensorflow.org/api_docs/python/tf/train/Example) to your TFRecord

<span class="example"><b>Running Example: </b> to keep up with the example download <a href="data/Caltech50.zip"> this data</a> and copy/unzip it in your data folder.</span>

### Imports you will need for this example
```
###########################################
###   Inside data/create_tfrecords.py   ###
###########################################
import tensorflow as tf
import glob, imageio, shutil, os
```

### Gathering Data
This is the most variable part of the conversion pipeline.  You may have a small batch of image (png/jpeg) files you can load into memory at once, a large dataset you need to load in piece-by-piece, or you may be generating data (i.e. interacting with a simulator) and writing to TFRecords dynamically.

Let's handle the simple case of loading in a single set of jpeg images.  The data provided is a subset of the [Caltech 101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/).

```
###########################################
###   Inside data/create_tfrecords.py   ###
###########################################

# Gather file paths to all iamges
data_dir = 'Caltech50'
object_dirs = glob.glob(data_dir + '/*')

objects = {}
for d in object_dirs:
    objects[d.split('/')[1]] = glob.glob(d + '/*.jpg')

# Create an integer label for each object category
categories = list(objects.keys())
category_labels = {}
for i in range(len(categories)):
    category_labels[categories[i]] = i
```

That was easy enough.  Now we have the file paths for all the images we want to store in our TFRecords and .

### Creating TFRecord Writer

The TFRecordWriter is what we will use write each Example once it's been constructed.

```
# Say we want one of the files to be named 'gerenuk.tfrecord' for one of the categories
tfrecord_filename = 'gerenuk.tfrecord'
writer = tf.python_io.TFRecordWriter(tfrecord_filename)
```

We will be creating one tfrecord file for every category below.

### Creating an Example

An [Example](https://www.tensorflow.org/api_docs/python/tf/train/Example) represents one datapoint in our dataset -- e.g. an image/label pair for image classification or state/value for reinforcement learning.

Each example is constructed from [Features](https://www.tensorflow.org/api_docs/python/tf/train/Feature).  One Feature is created for each subpiece of the datapoint - i.e. the image is one Feature and the label is another Feature.

The following helper functions create Features for different data types.

```
##########################################
###   Inside data/create_tfrecord.py   ###
##########################################

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))
```

To write a single image to our TFRecord file we would do the following:

```
# Load the image into memory and grab its label
category = list(objects.keys())[0]              # choose the first category -- 'gerenuk'
image = imageio.imread(objects[category][0])    # load the first gerenuk image as a numpy array
shape = image.shape                             # grab the shape so we can add it as meta-data
label = category_labels[category]

# Create a dictionary of this example's features
features = {
    'height' : int64_feature(shape[0]),
    'width' : int64_feature(shape[1]),
    'depth' : int64_feature(shape[2]),
    'image' : bytes_feature(image.tostring()),
    'label' : int64_feature(int(label))
}

# Now we both construct tf.train.Features from our feature dictionary and
#    construct a tf.train.Example from those features.
example = tf.train.Example(features=tf.train.Features(feature=features))

# To write it to our TFRecord file, we serialize the example and call write
#    from our TFRecordWriter handle.
writer.write(example.SerializeToString())
```

<span class='warning'><b>Warning: </b>be mindful of the data type of your image before converting it to a feature.  I sometimes find my numpy data getting converted to float64 when we would much rather store it as uint8!  loading a png/jpeg with imageio.imread will return data as uint8.</span>

### Creating a Dataset
The specifics of what you need for training/validation/test will determine exactly how you throw this into a loop and continue calling ```writer.write(...)``` on all examples.  See the <a href='https://github.com/crosleythomas/tensorplates/blob/master/templates/prepare_tfrecord.ipynb' target='_blank'>TFRecord template</a> for how I reuse some boilerplate code for this process.

<span class="example"><b>Running Example: </b>to complete the code (in addition to all code blocks above that say "Inside data/create_tfrecords.py") for our example, add the following.  This code creates a separate TFRecord file for each object class with a 80/20 split between training and validation data.</span>
```
##########################################
###   Inside data/create_tfrecord.py   ###
##########################################

# Create train/valid directories to store our TFRecords
if not os.path.exists('tfrecords') and not os.path.isdir('tfrecords'):
    os.mkdir('tfrecords')

if not os.path.exists('tfrecords/train') and not os.path.isdir('tfrecords/train'):
    os.mkdir('tfrecords/train')

if not os.path.exists('tfrecords/valid') and not os.path.isdir('tfrecords/valid'):
    os.mkdir('tfrecords/valid')

object_names = list(objects.keys())
# Create a separate TFRecord file for each object category
for o in object_names:
    print(o)
    # Create this object's TFRecord file
    train_writer = tf.python_io.TFRecordWriter('tfrecords/train/' + o + '.tfrecord')
    valid_writer = tf.python_io.TFRecordWriter('tfrecords/valid/' + o + '.tfrecord')
    # Write each image of the object into that file
    num_images = len(objects[o])
    for index in range(num_images):
        i = objects[o][index]
        # Let's make 80% train and leave 20% for validation
        if index < num_images * 0.8:
            writer = train_writer
        else:
            writer = valid_writer
        image = imageio.imread(i)
        shape = image.shape
        label = category_labels[o]
        # Create features dict for this image
        features = {
            'height' : int64_feature(shape[0]),
            'width' : int64_feature(shape[1]),
            'depth' : int64_feature(shape[2]),
            'image' : bytes_feature(image.tostring()),
            'label' : int64_feature(int(label))
        }
        # Create Example out of this image and write it to the TFRecord
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    train_writer.close()
    valid_writer.close()
```

<span class="example"><b>Running Example: </b>the complete create_tfrecords.py file can be found <a href="code/create_tfrecords.py">here</a>.</span>

### Advanced Formatting
There are several extensions of this basic process you may want to use when creating your records.

* Splitting into train/validation/test records
* Splitting data into multiple TFRecords to avoid giant files.
* Loading small chunks of your data at one time
* Interleaving generating data (e.g. interacting with the OpenAI Gym simulator) and then writing the data to a TFRecord

<span class='protip'><b>Pro tip: </b>The size of the TFRecord files you create should depend on... (memory limits of the computer you will run on?).  You can use the following to determine a good file size: </span>

<hr>
## Continue Reading

<button onclick="location.href='dataload'" class='continue-links'>Continue to Part 3</button>
In Part 3 we will see how to load these TFRecord files and prepare them to be used in a model we will build in [Part 4](model).

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

