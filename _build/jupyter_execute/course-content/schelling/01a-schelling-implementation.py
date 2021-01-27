#!/usr/bin/env python
# coding: utf-8

# # Implementation of Schelling's Segregation Model
# 
# This notebook contains an implementation of Schelling's segregation model in mesa.  See [01-schelling.ipynb](01-schelling.ipynb) for an overview of the model. 

# ## Imports

# In[1]:


from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ipywidgets import widgets, interact, interact_manual
import seaborn as sns
sns.set()


# ## The agents and the model

# In[2]:


class SchellingAgent(Agent):
    '''
    Schelling segregation agent
    '''
    def __init__(self, unique_id, pos, model, agent_type):
        '''
         Create a new Schelling agent.
         Args:
            pos: Agent initial location.
            agent_type: Indicator for the agent's type (minority=1, majority=0)
        '''
        super().__init__(unique_id, model)
        self.pos = pos
        self.type = agent_type

    def step(self):
        similar = 0
        neighbors = self.model.grid.neighbor_iter(self.pos)
        for neighbor in neighbors:
            if neighbor.type == self.type:
                similar += 1

        # If unhappy, move:
        if similar < self.model.homophily:
            self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1


class SchellingModel(Model):
    '''
    Model class for the Schelling segregation model.
    '''
    def __init__(self, height, width, density, minority_percent, homophily):
        
        self.height = height
        self.width = width
        self.density = density
        self.minority_percent = minority_percent
        self.homophily = homophily

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(height, width, torus=True)

        self.happy = 0
        self.datacollector = DataCollector(
            {"happy": lambda m: m.happy},  # Model-level count of happy agents
            # For testing purposes, agent's individual x and y
            {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]})
        self.running = True
        # Set up agents
        agent_id = 0
        for cell in self.grid.coord_iter():
            _,x,y = cell
            if random.random() < self.density:
                if random.random() < self.minority_percent:
                    agent_type = 1
                else:
                    agent_type = 0
                agent = SchellingAgent(agent_id, (x, y), self, agent_type)
                agent_id += 1
                self.grid.position_agent(agent, x=x, y=y)
                self.schedule.add(agent)
        
    def step(self):
        '''
        Run one step of the model. If All agents are happy, halt the model.
        '''
        self.happy = 0  # Reset counter of happy agents
        self.schedule.step()
        self.datacollector.collect(self)
        if self.happy == self.schedule.get_agent_count():
            self.running = False


# ## Visualization using Jupyter widgets

# In[3]:



max_rounds = 100

def value(cell):
    if cell is None: return 0
    elif cell.type == 1: return 1
    elif cell.type == 0: return 2

def run_schelling_sim(height, width, density, minority_percent, homophily):
    fig, ax = plt.subplots()
    
    # initialize the model
    model = SchellingModel(height, width, density, minority_percent, homophily)
    num_rounds = 0
    while model.running and num_rounds < max_rounds:
        num_rounds += 1
        model.step()
        data = np.array([[value(c) for c in row] for row in model.grid.grid])
        df = pd.DataFrame(data)
        sns.heatmap(df, cbar=False, linecolor='white', cmap=['white', 'blue', 'red'])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        clear_output(wait=True)
        display(fig);
    plt.close()
    print(f"The simulation ran for {num_rounds} rounds.")
    print(f"Out of {model.schedule.get_agent_count()}, there are {model.happy} happy agents.")

interact_manual(run_schelling_sim, 
                height = widgets.IntSlider(
                    value=50,
                    min=0,
                    max=100,
                    step=1,
                    description='height:',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d'
                ), 
                width = widgets.IntSlider(
                    value=50,
                    min=0,
                    max=100,
                    step=1,
                    description='width:',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d'
                ),  
                density = widgets.FloatSlider(
                    value=0.7,
                    min=0,
                    max=1.0,
                    step=0.1,
                    description='density:',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.1f'
                ),
                minority_percent = widgets.FloatSlider(
                    value=0.2,
                    min=0,
                    max=1.0,
                    step=0.1,
                    description='minority %:',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.1f'
                ), 
                homophily = widgets.IntSlider(
                    value=4,
                    min=0,
                    max=8,
                    step=1,
                    description='homophily:',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d'
                ), 
               );


# In[ ]:




