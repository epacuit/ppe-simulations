#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mesa import Agent

class MyAgent(Agent):
    def __init__(self, param):
        self.param = param
    def step(self):
        print("The agent parameter is ", self.param)
a = MyAgent("test")
print(a.param)


# In[2]:


from mesa import Agent

class MyAgent(Agent):
    def __init__(self, param):
        self.param = param
    def step(self):
        print("The agent parameter is", self.param)
a = MyAgent("test")
a.step()


# In[3]:


from mesa import Agent

class MyAgent(Agent):
    def __init__(self, param):
        self.param = param
        super().__init__(7, None)
    def step(self):
        print("The agent parameter is ", self.param)

a = MyAgent("test")

print(a.unique_id)


# In[4]:


class A1():
    def __init__(self, param):
        self.param1 = param

class A2(A1):
    def __init__(self, param):
        self.param2 = param

a = A2("test")
print(a.param1)


# In[14]:


class A1():
    def __init__(self, param):
        self.param1 = param

class A2(A1):
    def __init__(self, param):
        self.param2 = param
        super().__init__("another parameter")

a = A2("test")
print(a.param1)


# # Modification to the Schelling Model
# 
# 
# 
# * Limit the size of groupings. 
# 
# * Modify the move strategy: When an unhappy agent moves, let the agent move to the *closest* happy spot. If there is no such a spot available in an agent's turn, let the agent stay at the original position. 
# 
# * Add  a value to the spots might simulate how some areas are more expensive and desirable to others. Some people might understand that they are living around people they may not like, but the area they live in can compensate for that.
# 
# * A "neighborhood" could be more than just immediate neighbors and extend maybe another row and column. 
# 
# * Alter the 2D grid structure into a 3D cube with more locations for agents to reside. 
# 
# * Instead of a strict tolerance, make the moving probability a random distribution based on the homophilly number (possibly exponential distribution). 
# 
# 

# # Modification to the Schelling Model
# 
# * Agents will get increasingly upset if they are unable to move when they want to, causing their homophily to grow. Perhaps this could emulate the beginnings of resentment towards other groups in real life.
# 
# * Agents can are only willing to move a certain distance: this is often the case in real life, considering transportation costs.
# 
# * More than 2 type of agents and/or let the agents have different characteristics.  Whit is the highest threshold possible while still ensuring that 100 percent of the agents are happy. 
# 
# * Introduce more agent types and rank how much one agent type "likes" the others. This is more reflective of the real world, where there are many different groups of people that get along with other groups to varying degrees. 
# 
# * Examine the effect of a change in homophily: Set the original distribution with a certain extent of segregation. For example, the original distribution could be a stable one in which everyone has at least three similar neighbors. Now, examine what could happen if everyone increases their homophily requirement; for example, everyone wants at least four similar neighbors instead of two. 
# 
# 
# 

# In[ ]:




