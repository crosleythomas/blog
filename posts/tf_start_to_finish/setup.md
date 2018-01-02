---
layout: default
title: TensorFlow Environment Setup

---

# Part 1: Environment Setup

## tl;dr
```
# Make sure you have virtualenv
pip install virtualenv

# Create an environment for TensorFlow development
virtualenv env --python=python3

# Start the environment
source env/bin/activate

# Install TensorFlow
pip3 install tensorflow-gpu # with GPU -- additional setup steps need to be taken for GPU support (coming soon).
pip3 install tensorflow # without GPU

# Install Jupyter
pip3 install jupyter

# Create a project
mkdir my_new_project
cd my_new_project

# Start jupyter
jupyter notebook

# Create a new notebook
# New (button) --> Python [env]

```

## Machine

## Virtual Environments
[virtualenv]()

## Installing TensorFlow

##### CPU Version

##### GPU Version

## Jupyter Notebooks
If you aren't using [Jupyter Notebooks](http://jupyter.org/) yet, start now.  I put off making the switch for far too long and research has gotten immeasurably since I did.  "The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text." - jupyter.org<br>

Jupyter is a program you run locally that pops open a tab in your web browser that looks like this...
![Jupyter Preview](images/jupyterpreview.png)

It is the perfect environment for prototyping in machine learning and data science.  You can test out snippets of code, visualize images (such as your dataset to make sure you are loading it correctly), run training, etc.

<span class="protip"><b>Protip:</b> If you will eventually run the training on a server, prototype using Jupyter and then export the .ipynb notebook file as a .py with File --> Download as --> Python (.py) from inside the notebook.
</span>
<br><br>
![Exporting Notebook](images/exporting_notebook.png)

<span class='sidenote'><b>Sidenote:</b> Google has something now called [Collaboratory](https://colab.research.google.com/) which takes Jupyter Notebooks and integrates them with Google Drive.  Collaboratory picks up a lot of the features from Google Docs such as saving in Drive, sharing a file with otheres who can live edit/run, and leave comments/feedback like you would any other Google Docs file.  I'm just starting to use these but think they are really exciting for projects with a lot of informal collaboration.</span>
<br><br>
![Collaboratory Preview](images/collaboratory.png)

<hr>
## Continue Reading

<button onclick="location.href='dataprep'" class='continue-links'>Continue to Part 2</button>
In Part 2 we will see how to prepare a dataset in TensorFlow's TFRecord format.

<hr>

<div style="text-align: center;">
	<button onclick="location.href='setup'" class='continue-links'>Part 1: Setup</button>
	<button onclick="location.href='dataprep'" class='continue-links'>Part 2: Preparing Data</button>
	<button onclick="location.href='dataload'" class='continue-links'>Part 3: Consuming Data</button>
	<button onclick="location.href='model'" class='continue-links'>Part 4: Defining a Model</button>
	<button onclick="location.href='traineval'" class='continue-links'>Part 5: Training and Evaluating</button>
	<button onclick="location.href='deploy'" class='continue-links'>Part 6: Exporting, Testing, and Deploying</button>
	<button onclick="location.href='summary'" class='continue-links'>Part 7: All Together Now</button>
	<button onclick="location.href='references'" class='continue-links'>Part 8: Furthur Reading and References</button>
</div>