# Mesa

Mesa is a Python package that can be used to quickly create and analyze agent-based models. Its goal is to be a Python 3  alternative to other frameworks for creating agent-based simulations, such as [NetLogo](https://ccl.northwestern.edu/netlogo/), [Repast](https://repast.github.io/), and [MASON]. (https://cs.gmu.edu/~eclab/projects/mason/).

Consult the documentation for more information about Mesa: [https://mesa.readthedocs.io/en/master/](https://mesa.readthedocs.io/en/master/).

```{note}
To install mesa in a Jupyter notebook run the following command: 

`!pip install mesa`

(Remove the ! to install Mesa from the command line). 
```

When creating an agent-based simualtion in Mesa, 

1. Define your agent as a subclass of the Mesa `Agent` class.
2. Define your model as a subclass of the Mesa `Model` class.

from mesa import Agent, Model

class MyAgent(Agent): 
    
    def __init__(self, unique_id, model = None):
        super().__init__(unique_id, model)
        # initialize an agent 
        
    ### add other methods that are useful for your agent ###
    
    def step(self):
        # what does the agent do each step of the simulation?
        pass
    
class MyModel(Model): 
    
    def __init__(self):
        super().__init__()
        # initialize a model and create agents 
                
    ### add other methods that are useful for your simulation ###
    
    def step(self):
        # executes a step of the simulation
        pass

```{note}

Your agent class or model class is not *required* to have a step method.  Many of the Mesa agent-based models do use these functions, so it is good practice to define these functions.    Note also that the default value of the model in MyAgent is `None`.  This makes it easier for testing purposes so you can create an instance of MyAgent without creating an instance of MyModel. 

```



from mesa.time import RandomActivation


Mesa has two main types of grids: SingleGrid and MultiGrid. SingleGrid enforces at most one agent per cell; MultiGrid allows multiple agents to be in the same cell. Since we want agents to be able to share a cell, we use MultiGrid.



