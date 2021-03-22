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

Your agent class or model class is not *required* to have a step method.  Many of the Mesa agent-based models do use these functions, so it is good practice to define these functions.   

Note also that the default value of the model in MyAgent is `None`.  This makes it easier for testing purposes so you can create an instance of MyAgent without creating an instance of MyModel. 

```

To illustrate the main features of the Mesa framework, consider the following simulation: 

1. There are some number of agents
2. Each agent is initially assigned 10 units of money.
3. At each step, an agent $i$ is matched up with another agent $j$ (chosen at random).  Both agents flip a fair coin.   If the coins match (both land heads or both land tails) then $i$ gives $j$ 1 unit of money, and if they mismatch then $j$ gives $i$ one unit of money.  
4. When an agent runs out of money they no longer participate in any game. 

The Python `random` package will be used in two ways: 

1. To simulate flipping a fair coin (recall the implementation of the `CoinFlipper` class).   
2. Use `random.choice` to randomly choose an opponet to play the coin game: `random.choice` choses a random element from any list of objects.      

Rather than using the Python random package, we will use Mesa's wrapper for this function. This is accessible as an attribute of both the Agent and the Model class.

The final piece needed to implement the above simulation is a way of *scheduling* the agents.   Mesa has multiple schudlers from the [Mesa Time Moduel](https://mesa.readthedocs.io/en/master/apis/time.html). The most useful is the `RandomActivation` scheduler.  During each step of the simulation, the agents are shuffled, then each agent's  `step` function is called.  The agents are added to the schedule using the `add` method. 

from mesa.time import RandomActivation


class Player(Agent): 
    
    def __init__(self, unique_id, model = None, init_payout = 10):
        super().__init__(unique_id, model)
        self.payout = init_payout
        self.current_flip = None # the current flip of the coin
        self.model = model
        
    def flip(self): 
        return 'H' if self.random.random() < 0.5 else 'T'

    def step(self): 
        
        if self.payout > 0: 
            
            # find possible opponents (any other player with a positive payout)
            possible_opponents = [_p for _p in self.model.schedule.agents if _p != self if _p.payout > 0]
            if len(possible_opponents) > 0: 
                # choose a random opponent
                opponent = self.random.choice(possible_opponents)
                
                # play the game
                if self.flip() == opponent.flip(): 
                    # outcomes match, so gain 1 and opponent loses 1  
                    self.payout += 1
                    opponent.payout -= 1

                else: 
                    # outcomes mismatch, so lose 1 and opponent gains 1
                    self.payout -= 1
                    opponent.payout += 1

class CoinGame(Model): 
    
    def __init__(self, num_players, seed = None):
        
        self.num_players = num_players
        
        self.schedule = RandomActivation(self) # use random activation
        
        # creat the players and add them to the schedule
        for p_id in range(num_players): 
            
            p = Player(p_id, self)
            self.schedule.add(p)
            
        # keep track of when the model should stop running
        self.running = True
        
    def step(self): 
        
        # call each players step function after randomly shuffling the agents
        self.schedule.step()
        
        # if there is only one player with a non-zero amount of money, then stop running
        if len([_p for _p in self.schedule.agents if _p.payout > 0]) == 1:
            self.running = False


# create a CoinGame with 10 players
g = CoinGame(10)

num_rounds = 1000

print("The players initial payouts:")
print([p.payout for p in g.schedule.agents], "\n")

# run the simulation for a maximu of num_rounds
for r in range(num_rounds): 
    g.step()
    # break out of the loop if the model stops running
    if not g.running: 
        break
    
print(f"After {r} rounds the final payouts: ")
print([p.payout for p in g.schedule.agents])

```{note}
The implementation uses  Mesa's own wrapper for Python `random` package.This allows one to set the random seed to reproduce the results (see [https://mesa.readthedocs.io/en/stable/best-practices.html#randomization](https://mesa.readthedocs.io/en/stable/best-practices.html#randomization) for a discussion). 
```

# create a CoinGame with 10 players
g = CoinGame(10, seed = 1)

num_rounds = 1000

print("The players initial payouts:")
print([p.payout for p in g.schedule.agents], "\n")

# run the simulation for a maximu of num_rounds
for r in range(num_rounds): 
    g.step()
    # break out of the loop if the model stops running
    if not g.running: 
        break
    
print(f"After {r} rounds the final payouts: ")
print([p.payout for p in g.schedule.agents])

print("\nRun the simulation again with the same seed.\n")
g = CoinGame(10, seed = 1)

num_rounds = 1000

print("The players initial payouts:")
print([p.payout for p in g.schedule.agents], "\n")

# run the simulation for a maximu of num_rounds
for r in range(num_rounds): 
    g.step()
    # break out of the loop if the model stops running
    if not g.running: 
        break
    
print(f"After {r} rounds the final payouts: ")
print([p.payout for p in g.schedule.agents])

We will introduce other features of the Mesa package, such as the [Mesa Space Module](https://mesa.readthedocs.io/en/master/tutorials/intro_tutorial.html#adding-space), the [DataCollector](https://mesa.readthedocs.io/en/master/tutorials/intro_tutorial.html#collecting-data) and [Batch Running](https://mesa.readthedocs.io/en/master/tutorials/intro_tutorial.html#batch-run), later in the text. 


