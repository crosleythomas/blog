import tensorflow as tf
import glob, imageio, shutil, os

#############################
###   Gather Data Files   ###
#############################

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

############################
###   Helper Functions   ###
############################

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

###########################
###   Write TFRecords   ###
###########################

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
