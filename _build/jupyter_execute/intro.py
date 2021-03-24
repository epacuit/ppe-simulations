# Course Overview

Computer Simulations for Philosophy, Politics and Economics<br/>
Spring 2021<br/>
Eric Pacuit <br/>
epacuit@umd.edu <br/>
University of Maryland<br/>


Computer simulations have been used for almost as long as there have been computers.  The earliest scientific use of computer simulations were for physics and engineering. Increasingly people are also employing computer simulation to understand social behavior both in humans and other animals. This class is about the nuts and bolts of computer simulation of social behavior and about the underlying theory behind this work. The course also discusses the philosophical issues that arise when evaluating the use of computational and mathematical models in economics and philosophy. During the course, students will gain hands-on experience developing simulations in Python. Although some previous programming experience will be helpful, the course will be completely self-contained.

```{warning}
This class website is under active development, and content will be added throughout the duration of the course. 
```



## Topics (tentative)

* Schelling's Model of Segregation
* Iterated Prisoner's Dilemma
* Opinion Dynamics: Hegselmann-Krause Model 
* Deliberation and Polarization
* Epistemic Democracy and the Condorcet Jury Theorem
* Simulating Elections 


##  Computational Tools 

* Python 3
* Jupyter Notebooks ([https://jupyter.org/](https://jupyter.org/))
    * Literate programming: Interactive documents (written in markdown/LaTeX) explaining the model and the code. 
* Data science toolkit: scipy, numpy, pandas, ...
* Visualization: matplotlib, seaborn, altair, etc.
* Sharing models: [Colab](https://colab.research.google.com/), [Streamlit](https://www.streamlit.io/), [Voila](https://voila.readthedocs.io/en/stable/index.html),...
* Probabilistic programming: [lea](https://bitbucket.org/piedenis/lea/wiki/Home), [pyro](http://pyro.ai/)
* Special packages: [mesa](https://mesa.readthedocs.io/en/master/#), [axelrod](https://axelrod.readthedocs.io/en/stable/), [preflib tools](https://www.preflib.org/)

## Mathematical Background

* Networks (Graph Theory)
* Basic Game Theory
* Basic Probability Theory
* Voting Theory

## Agent-Based Modeling

Agent-based simulations are used in many different areas: 

* Ross A. Hammond, [Considerations and Best Practices in Agent-Based Modeling to Inform Policy](https://www.ncbi.nlm.nih.gov/books/NBK305917/), NCBI, 2015.

In this coure, we are focused on simulations used in Philosophy (especially Social Epistemology), Politics  and Economics: 

* Conor Mayo-Wilson and Kevin J.S. Zollman, The Computational Philosophy: Simulation as a Core Philosophical Method. [Preprint](http://philsci-archive.pitt.edu/18100/), 2020.
* J. McKenzie Alexander, The Structural Evolution of Morality. Cambridge University Press, 2007.
* Mostapha Diss and Vincent Merlin (eds.), Evaluating Voting Systems with Probability Models: Essays by and in Honor of William Gehrlein and Dominique Lepelley. Springer, 2021.


## Weekly Schedule

0. Mathematical background
1. Explain the model
2. Learn how to code the model
3. Explore the model
4. Evaluate the model


## NetLogo

NetLogo ([https://ccl.northwestern.edu/netlogo/](https://ccl.northwestern.edu/netlogo/)) is a powerful framework for creating agent-based simulations.   It is a programming environment created in Java (although you do not need to know Java to use NetLogo).   

* There is a web version of NetLogo: [http://www.netlogoweb.org/](http://www.netlogoweb.org/)
* With NetLogo you can quickly create an agent-based model with animation (e.g., see the [Schelling Model of Segregation](http://www.netlogoweb.org/launch#http://ccl.northwestern.edu/netlogo/models/models/Sample%20Models/Social%20Science/Segregation.nlogo)) 
* There are a number of books written about NetLogo:
    * Uri Wilensky and William Rand, [An Introduction to Agent-Based Modeling: Modeling Natural, Social, and Engineered Complex Systems with NetLogo](https://www.amazon.com/Introduction-Agent-Based-Modeling-Natural-Engineered/dp/0262731894/ref=sr_1_4?dchild=1&keywords=agent+based+modeling&qid=1611489846&sr=8-4), The MIT Press, 2015.
    * Steven F. Railsback and Volker Grimm, [Agent-Based and Individual-Based Modeling: A Practical Introduction](https://www.amazon.com/Agent-Based-Individual-Based-Modeling-Practical-Introduction/dp/0691190836/ref=sr_1_3?dchild=1&keywords=agent+based+modeling&qid=1611489846&sr=8-3), Princeton University Press; 2nd edition (March 26, 2019).


## Why not NetLogo? 

* Netlogo is its own programming language with its own quirks
* Python is a standard tool used in AI, machine learning, and data science 
* Access to powerful packages for scientific and mathematical computing (scipy, numpy, etc.)
* Many different options for visualizing and sharing your simulations
* While NetLogo makes animating the model easier, we do not always need to visualize the dynamics of an agent-based model


## A Brief Introduction to Python (as needed)

> Python is an easy to learn, powerful programming language. It has efficient high-level data structures and a simple but effective approach to object-oriented programming. Python’s elegant syntax and dynamic typing, together with its interpreted nature, make it an ideal language for scripting and rapid application development in many areas on most platforms. ([https://docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/))

* You don't need to understand everything at this stage!   
* Python is an interpretted language. 
* Python is dynamically typed.


x = "hello"
print(x)
print(x + " world")

x = 7
print(x)
print(str(x) + " world")

x = [5, 10, 15]
print(x)
print(x + [20])

## Tutorials on Python

1. [https://developers.google.com/edu/python/](https://developers.google.com/edu/python/)

2. [http://pythontutor.com/](http://pythontutor.com/)





```{toctree}
:hidden:
:titlesonly:
:caption: Python Basics

Overview <python-basics/overview>
Introduction to Python <python-basics/introduction>
Classes <python-basics/classes>
Scientific Computing <python-basics/scientific-computing>
Networks <python-basics/networks>
Mesa <python-basics/mesa>
```


```{toctree}
:hidden:
:titlesonly:
:caption: Schelling Segregation Model

Overview <schelling/01-schelling>
Implementation <schelling/01a-schelling-implementation>
```


```{toctree}
:hidden:
:titlesonly:
:caption: Game Theory

Introduciton <game-theory/02a-intro-game-theory>
Games on a Network <game-theory/02b-games-local-interaction>
Justice and Bargaining <game-theory/02c-divide-the-dollar>
```


```{toctree}
:hidden:
:titlesonly:
:caption: Belief Dynamics

belief-dynamics/overview
belief-dynamics/hk-model
belief-dynamics/cjt
belief-dynamics/testimony
belief-dynamics/epistemic-networks
belief-dynamics/polarization
belief-dynamics/deliberation
```


```{toctree}
:hidden:
:titlesonly:
:caption: Preference Dynamics and Voting

Overview <preference-aggregation/overview>
```
