---
layout: default
title: Summary

---

# Part 7: Summary

Now you know how to prepare training data, create a model, train that model, and load it back in for testing.  To get set up as quickly as possible on new projects, I've created a full project template in <a href="https://github.com/crosleythomas/tensorplates" target="_blank">TensorPlates</a>.

## Creating a New Project
To start a new project simply do the following:
```
git clone git@github.com:crosleythomas/tensorplates.git
cp -r tensorplates/project <your project directory>
```

<hr>

<span class="protip"><b>Tip: </b>After creating a new project you can make your life easier by creating an alias inside your ```.bash_profile``` to activate the associate virtual environment and cd to that directory.  For example, say you have a project called flowers and have created a virtual environment called ```env``` inside the project directory.</span>

```
##################################
###   Inside ~/.bash_profile   ###
##################################
alias flowers='cd ~/Documents/projects/flowers && source env/bin/activate'
```

## Project Implementation Checklist

Now, you can step through completing your project with this checklist:
* <b>Setup</b><br>
<input type="checkbox"> Copy project structure<br>
<input type="checkbox"> Gather new dataset<br>
<input type="checkbox"> Create or activate virtual environment<br>
* <b>Data Preparation</b><br>
<input type="checkbox"> Load data<br>
<input type="checkbox"> Define output file(s) structure<br>
<input type="checkbox"> Define parse_fn<br>
* <b>Data Loading</b><br>
<input type="checkbox"> Get paths to TFRecords<br>
<input type="checkbox"> Define Dataset's map parse function<br>
* <b>Model</b><br>
<input type="checkbox"> Define any additional ```config``` parameters<br>
<input type="checkbox"> Make sure runtime parameters make sense for your current system (i.e. ```num_parallel_calls``` in ```dataset.map```, ```buffer_size``` in ```dataset.shuffle``` )<br>
<input type="checkbox"> Implement model_fn<br>
<input type="checkbox"> Define ```params```<br>
* <b>Training</b><br>
    * Training/Evaluation loop is already defined is the template<br>
    <input type="checkbox"> You should only need to define ```params.train_epochs```<br>
* <b>Evaluation</b><br>
<input type="checkbox"> Mount ```model_dir``` if running remotely<br>
<input type="checkbox"> Open TensorBoard and Watch<br>
* <b>Deploy</b><br>
<input type="checkbox"> This one is completely up to you!<br>

<hr>
## Continue Reading

<button onclick="location.href='references'" class='continue-links'>Continue to Part 8</button>
Part 8 cites the tutorials I found most helpful and other resources you may want to use.

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