---
layout: default
title: Environment Setup

---

# Part 1: Environment Setup

<div style="text-align: center">
    <a href="http://hn.premii.com/#/article/15940724" target="_blank">GPU Benchmarking Article</a><br>
    <a href="https://virtualenv.pypa.io/en/stable/" target="_blank">Virtualenv</a>
</div>

## tl;dr
```
# Make sure you have virtualenv
pip install virtualenv

# Create an environment for TensorFlow development
virtualenv env --python=python3

# Activate the environment
source env/bin/activate

# Install TensorFlow
pip3 install tensorflow # without GPU -- you probably want to start with this one
pip3 install tensorflow-gpu # with GPU -- additional setup steps need to be taken for GPU support (coming soon)

# Install Jupyter
pip3 install jupyter

# Create a project
mkdir tf_tutorial
cd tf_tutorial
mkdir code
touch code/train.py
touch code/model.py
touch code/util.py
mkdir data
touch data/prepare_tfrecords.py
```

<div style="text-align: center">
    <img src='images/tf_structure.png'><br>
    <b>High level description of the project code.</b>
</div>

## Machine

Using the right machine can be the difference between an exciting, but unrecognized idea, and the next hit project.  Below are some recommendations for how to get the best setup for developing models in TensorFlow.

#### Software
There is official support for Mac, Linux, and Windows machines.  However, almost everyone uses either Mac or Linux.  Linux (Ubuntu) is the best choice since you can find a lot of Linux machines for sale with NVIDIA GPUs (the only GPUs that work with TensorFlow).  Mac laptops don't have NVIDIA GPUs but you can technically set up external GPUs.  Also, if you have the money to get a Mac Pro, NVIDIA launched the [NVIDIA TITAN Xp](https://blogs.nvidia.com/blog/2017/04/06/titan-xp/) in May of 2017 that is Mac Pro compatible.

#### Hardware
In general, the following hardware setups are roughly ordered both in their benefit and the difficulty of set up:

1. <b>Worst:</b> Local CPU Only Machine
2. Your Organization's CPU Only Machines
3. <b>Recommended:</b> Local CPU+GPU Machine
4. Your Organization's CPU+GPU Machines
5. <b>Best:</b> Distributed GPUs on a cloud service (Google Cloud, AWS, Azure)

Note that TensorFlow only supports NVIDIA GPUs.

<span class='protip'><b>Tip:</b> if buying a decent desktop with a good GPU (NVIDIA GeForce GTX 1060-1080) is close to within your budget, do it now.  You are going to do it eventually and you don't want to give up on machine learning because you got frustrated with your lack of computing power.</span>

Having some, any, kind of GPU will help every deep learning project you do immensely.  [This benchmarking article](http://hn.premii.com/#/article/15940724) does a really nice job of comparing both the speed and cost of CPUs and different GPUs.  

Check out the article for a detailed breakdown but here is the best diagram for understanding how much GPUs help.
![Benchmarking](images/benchmarking_cnn_for_mnist.png)

That's the difference between trying out a new project one night and seeing it work and running the same project, checking back in two days, forgetting what parameters you used, getting frustrated, and giving up.

Or, for another way of looking at this, enjoy Adam and Jamie from Mythbusters explain the power of GPUs. (Interesting to note this was made in 2009 and sponsored by NVIDIA.)

<div style="text-align: center;">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/-P28LKWTzrI?rel=0" frameborder="0" gesture="media" allow="encrypted-media" allowfullscreen align="center"></iframe>
</div>

Hardware is a big topic I've only just gotten interested in recently.  This post is mostly about TensorFlow best practices.  In a future post I will share many more details on what hardware is best and why.

Now that you have your hardware ready, let's get up and running with the software.

## Virtual Environments
Despite many people trying to get me to switch to [Conda environments](https://conda.io/docs/user-guide/tasks/manage-environments.html), I still find virtualenv the easiest tool to use for managing python dependencies.

[Virtualenv](https://virtualenv.pypa.io/en/stable/) creates isolated Python environments so we can install different dependencies for different projects.

First, make sure you have virtualenv on your machine.

```
pip install virtualenv
```

Next, create an environment for TensorFlow development.  This command will create a directory called <em>env</em> where all our dependencies will be stored.

```
virtualenv env --python=python3
```

To start using this environment, 'activate' it with the following command:
```
source env/bin/activate
```

Notice you should now see (env) appended to your terminal message.  For example mine changed from<br>
```
tcrosley: ...
```

to 

```
(env) tcrosley: ...
```

That means you are good to go.  Now, when you pip install something, it will install the new package to this virtual environment.  This may be important if you have different projects using different versions of TensorFlow.  One of your old projects may use an environment with TensorFlow Version 1.X for backwards compatibility and a newer project may use TensorFlow Version 1.Y for all the latest features.

<span class="example"><b>Running Example: </b>the best thing about virtual environments is how easy it is to install dependencies.  For the running example I will be doing in this guide you see all the dependencies needed in <a href="data/requirements.txt" download>this file</a>.  That file is called a <em>requirements</em> file.  It can be generated from a virtualenv with ```pip freeze > requirements.txt ```.  After downloading the file, install all the dependencies by activating your virtualenv and running ```pip install -r requirements.txt```</span>



## Installing TensorFlow


##### CPU Version
This guide only requires the CPU installation of TensorFlow.  Fortunately, installing this is as easy as:
```
pip3 install tensorflow
```

You should now be able to import TensorFlow.
```
python3
>>> import tensorflow as tf
>>> tf.__version__
# '1.4.0' as of this writing
```

##### GPU Version
Installing TensorFlow with GPU support is more difficult and requires installing cuda and cuDNN.  More to come on this later...
```
Coming Soon
```

## Jupyter Notebooks
If you aren't using [Jupyter Notebooks](http://jupyter.org/) yet, start now.  I put off making the switch for far too long and research has gotten immeasurably more enjoyable since I did.  "The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text." - jupyter.org<br>

For the running example in this post I use python files for simplicity but I almost always use Jupyter Notebooks when prototyping new ideas.

Jupyter is a program you run locally that pops open a tab in your web browser that looks like this (from the Jupyter website)...
![Jupyter Preview](images/jupyterpreview.png)

It is the perfect environment for prototyping in machine learning and data science.  You can test out snippets of code, visualize images (such as your dataset to make sure you are loading it correctly), run training, etc.

<span class="protip"><b>Tip:</b> If you will eventually run the training on a server, prototype using Jupyter and then export the .ipynb notebook file as a .py with File --> Download as --> Python (.py) from inside the notebook.
</span>
<br><br>
![Exporting Notebook](images/exporting_notebook.png)

<span class='sidenote'><b>Side note:</b> Google now has something called [Collaboratory](https://colab.research.google.com/) which takes Jupyter Notebooks and integrates them with Google Drive.  Collaboratory picks up a lot of the features from Google Docs such as saving in Drive, sharing a file with otheres who can live edit/run, and leave comments/feedback like you would any other Google Docs file.  I'm just starting to use these but think they are really exciting for projects with a lot of informal collaboration.</span>
<br><br>
![Collaboratory Preview](images/collaboratory.png)

## Creating a New Project
<span class="example"><b>Running Example: </b>Create a directory with the following structure to follow along with the example project we will create.</span>
```
|<project name>
    |--- code
        |--- train.py
        |--- model.py
        |--- util.py
    |--- data
        |--- Caltech50
        |--- prepare_tfrecords.py
```

You can copy the following code into your command line:
```
mkdir tf_tutorial
cd tf_tutorial
mkdir code
touch code/train.py
touch code/model.py
touch code/util.py
mkdir data
touch data/prepare_tfrecords.py
```

Download the barebones structure here:
[Project Structure](data/tf_tutorial.zip)

Or download the complete project here if you just want to follow along:
[Complete Tutorial](data/tf_tutorial.zip)

<hr>
## Continue Reading

<button onclick="location.href='dataprep'" class='continue-links'>Continue to Part 2</button>
Now we're all set up and ready to get started on a real project.  In Part 2 we will see how to prepare a dataset in TensorFlow's TFRecord format.

<hr>

<div style="text-align: center;">
    <button onclick="location.href='https://crosleythomas.github.io/blog/'" class='continue-links' target="_blank">Blog</button>
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