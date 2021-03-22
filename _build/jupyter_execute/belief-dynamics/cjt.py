# Condorcet Jury Theorem

import random
import pylab
import matplotlib.mlab as mlab
import functools
import itertools
from __future__ import print_function
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from functools import reduce
from collections import Counter
from tqdm.notebook import tqdm  

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
from IPython.display import display, Math, Latex
sns.set()

```{warning}
This notebook uses Jupyter widgets that will only work if the notebook is run locally. 
```

Suppose that $V=\{1, 2, 3, \ldots, n\}$ is a set of voters or experts, and consider a  set of two alternatives.  E.g., $\{\mbox{convict}, \mbox{acquit}\}$, $\{\mbox{abolish}, \mbox{keep}\}$, $\{0,1\}$, $\ldots$

Let  $\mathbf{x}$ be a random variable (called the **state**)  whose values range over the two alternatives. 

In addition, let $\mathbf{v}_1, \mathbf{v}_2, \ldots$ be random variables represeting the votes for individuals $1, 2, \ldots, n$

Let $R_i$ be the event that $i$ votes correctly: it is  the event that $v_i$ coincides with the state. 

**Unconditional independence (UI)**: The correctness events $R_1, R_2, \ldots, R_n$ are (unconditionally) independent.

**Unconditional competence (UC)**: The (unconditional) correctness probability
$p = Pr(R_i)$, the (unconditional) competence, (i) exceeds $\frac{1}{2}$ and (ii) is the same for
each voter $i$.

**Condorcet Jury Theorem**. Assume UI and UC. As the group size increases, the probability of a
correct majority (i) increases (growing reliability), and (ii) tends to one (infallibility).

The Condorcet Jury Theorem has two main theses: 
    
**The growing-reliability thesis**: Larger groups are better truth-trackers. That
is, they are more likely to select the correct alternative (by majority) than
smaller groups or single individuals.

**The infallibility thesis**: Huge groups are infallible truth-trackers. That is, the
likelihood of a correct (majority) decision tends to full certainty as the group
becomes larger and larger.

The probability of at least $m$ voters being correct is: 

$$\sum_{h=m}^n \frac{n!}{h!(n-h)!} * p^h*(1-p)^{n-h}$$


import operator as op
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return float(numer//denom)

def probability_majority_is_correct(num_voters=100,prob=0.51):
    return sum([ncr(num_voters,k)*(prob**k)*(1-prob)**(num_voters-k) 
                for k in range(int(num_voters/2+1),num_voters+1)])

def make_maj_prob_graphs():
    probs = np.linspace(0,1,num=100)

    number_of_voters = [ 1, 3,  11, 51,  201, 501, 1001]
    sns.set(rc={'figure.figsize':(10,5)})
    
    plt.subplot(121)
    for num_voters in number_of_voters:
        maj_probs = [probability_majority_is_correct(num_voters=num_voters,prob=p)  for p in probs]
        plt.plot(list(probs),maj_probs, label="$n=" + str(num_voters) + "$")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Probability of voting correctly')
    plt.ylabel('Probability the majority is correct')
    
    plt.subplot(122)
    for num_voters in number_of_voters:
        maj_probs = [probability_majority_is_correct(num_voters=num_voters,prob=p) - p  for p in probs]
        plt.plot(list(probs),maj_probs, label="$n=" + str(num_voters) + "$")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Probability of voting correctly')
    plt.ylabel('$Pr(M_n) - p$')

    plt.plot([0.0,1.0],[0.0,0.0],color='black',alpha=0.6)
    
    sns.set()
    plt.subplots_adjust(bottom=0.1, right=1.5, top=0.9, wspace = 0.75)

    plt.savefig('cjtplots.png')


make_maj_prob_graphs()

**Theoreom**.   For any (odd) number of voters, each with a probability $p>1/2$ of choosing correctly, then majority rule is preferred to the expert rule.  

**Theorem**.    Assume $p_1\ge p_2>p_3>1/2$, then the simple majority rule is preferred to the expert rule.  


def probability_majority_is_correct_diff_probs(p1=0.55, p2=0.6, p3=0.8):
    
    maj_prob = p1*p2*p3 + p1*p2*(1-p3) + + p2*p3*(1-p1) + + p1*p3*(1-p2) 
    expert_prob = 1.0/3.0 * p1 + 1.0/3.0 * p2 + 1.0/3.0 * p3
    print(f"Majority probability: {round(maj_prob,3)}\nExpert Probability: {round(expert_prob,3)}")
    if maj_prob > expert_prob: 
        print(f" Majority rule is better than the expert rule")
    else: 
        print(f"\n The expert rule is better than majority rule")

    
maxprob = interact_manual(probability_majority_is_correct_diff_probs,p1=(0.5,1,0.01),p2=(0.5,1,0.01),p3=(0.5,1,0.01))

evidence = [2,3,4,5,6,7,8,10, 12, 14]

class Agent():
    
    def __init__(self, comp=0.501):
        self.comp = comp
        
    def vote(self, ev):
        #vote on whether the event is true or false
        #need the actual truth value in order to know which direction to be biased
        if ev:
            #ev is true
            return int(random.random() < self.comp)
        else:
            return 1 - int(random.random() < self.comp)


def maj_vote(the_votes):
    votes_true = len([v for v in the_votes if v == 1])
    votes_false = len([v for v in the_votes if v == 0])

    if votes_true > votes_false:
        return 1
    elif votes_false > votes_true:
        return 0
    else:
        return -1  #tied

def generate_competences(n, mu=0.51, sigma=0.2):
    competences = list()
    for i in range(0,n):
        #sample a comp until you find one between 1 and 0
        comp=np.random.normal(mu, sigma)
    
        while comp > 1.0 or comp < 0.0:
            comp=np.random.normal(mu, sigma)
        competences.append(comp)
    return competences

import pandas as pd
NUM_ROUNDS = 500
from tqdm import notebook 

def make_plots(max_voters=201, 
               comp_mu=0.501, 
               comp_sigma=0.1):
    P=True
    max_num_voters = max_voters
    total_num_voters = range(1,max_num_voters)

    competences = generate_competences(max_num_voters,
                                       mu=comp_mu, 
                                       sigma=comp_sigma)
    maj_probs = list()
    expert_probs = list()
    for num_voters in notebook.tqdm(total_num_voters, desc='voting'):
        experts = list()

        experts = [Agent(comp=competences[num-1]) for num in range(0,num_voters)]
    
        maj_votes = list()
        expert_votes = list()
        for r in range(0,NUM_ROUNDS):
            # everyone votes
            votes = [a.vote(P) for a in experts]
            maj_votes.append(maj_vote(votes))
        
            expert_votes.append(random.choice(experts).vote(P))
    
        maj_probs.append(float(float(len([v for v in maj_votes if v==1]))/float(len(maj_votes))))
        expert_probs.append(float(len([v for v in expert_votes if v==1]))/float(len(expert_votes)))
    
    sns.set(rc={'figure.figsize':(11,5)})
    plt.subplot(121)

    data = {" ": range(0,max_num_voters), "competence": competences}
    plt.ylim(0,1.05)
    plt.title("Competences")
    df = pd.DataFrame(data=data)
    sns.regplot(x=" ", y="competence", data=df, color=sns.xkcd_rgb["pale red"])


    plt.subplot(122)
    plt.title("Majority vs. Experts")
    plt.plot(list(total_num_voters), maj_probs, label="majority ")
    plt.plot(list(total_num_voters), expert_probs, label="expert ")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Number of experts')
    plt.ylabel('Probability')
    plt.ylim(0,1.05)
    plt.subplots_adjust(bottom=0.1, right=1.5, top=0.9, wspace = 0.75)

    sns.set()
    plt.savefig("cjt_simulation.png")




p = interact_manual(make_plots,max_voters=(1,501,1),comp_mu=(0,1,0.01),comp_sigma=(0,2,0.1))

## Further Reading 

D. Austen-Smith and J. Banks, Aggregation, Rationality and the Condorcet Jury Theorem, The American Political Science Review, 90, 1, pgs. 34 - 45, 1996
 
D. Estlund, Opinion Leaders, Independence and Condorcet's Jury Theorem, Theory and Decision, 36, pgs. 131 - 162, 1994
 
F. Dietrich, The premises of Condorcet's Jury Theorem are not simultaneously justified, Episteme, Episteme - a Journal of Social Epistemology 5(1): 56-73, 2008

R. Goodin and K. Spiekermann, *An Epistemic Theory of Democracy*, Oxford University Press, 2018
 

What happens if there are more than two options? 


C. List and R. Goodin. Epistemic democracy: Generalizing the condorcet jury theorem. Journal of
political philosophy, 9(3):277–306, 2001.

def display_probs(cjt_model, num_options):
    '''display the probabilities of the agents'''
    _probs = list()
    for a in cjt_model.schedule.agents: 
        _probs.append(np.array(a.probs))
    probs = np.array(_probs)
    prs = probs.transpose()

    for opt in range(num_options):
        plt.barh(range(num_agents), prs[opt], 1,
                 left=sum([np.array([0.0]*num_agents)] + [prs[i] for i in range(opt)]),
                  lw = 0.01)
    plt.show()
    plt.clf()

from dataclasses import dataclass, field
from typing import List

@dataclass
class Options:
    '''Options is a list with one option identified as the "correct" one '''
    num: int = 2
    correct_idx: int = 0 # index of the correct option
    names:  List[str] = field(default_factory=list) # names of the options
        
    def __post_init__(self):
        self.names = [f"P{p+1}" for p in self.props]
    
    @property
    def props(self) -> List[int]:
        '''the list of all options'''
        return list(range(self.num))
    
    @property
    def C(self) -> int:
        return self.props[self.correct_idx]
    
    @property
    def C_as_list(self) -> List[int]:
        return [self.props[self.correct_idx]]
    
    @property
    def W(self) -> List[int]: 
        return list(self.props[self.correct_idx + 1::])
    
    def name(self, opt): 
        return self.names[opt]
    
    def set_names(self, names):
        assert len(names) == self.num, f"You need {self.num} names, but only provided {len(names)} names: {names}"
        self.names = names
        
    # make options iterable
    def __iter__(self):
        return iter(self.props)


from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


def gen_option_probability_normal(mu,sigma):
    '''return single p'''
    pr=np.random.normal(mu, sigma)
    while pr > 1.0 or pr < 0.0:
        pr=np.random.normal(mu, sigma)
    return [pr, 1-pr]

def gen_option_probability_beta(a,b, num=1):
    pr=np.random.beta(a,b, num)[0]
    return [pr, 1-pr]

def gen_options_probability_dirichlet(params, num=1):
    return np.random.dirichlet(params, num)
    

init_probs = {'1_opt_fixed_probs1': lambda : [0.51, 0.49],
              '1_opt_fixed_probs2': lambda : [0.75, 0.25],
              '1_opt_fixed_probs3': lambda : [0.49, 0.51],
              '4_opt_fixed_probs': lambda : [0.40, 0.20, 0.20, 0.20],
              '7_opt_fixed_probs': lambda : [0.30, 0.10, 0.20, 0.05, 0.05, 0.15, 0.15],
              '2_opt_normal1': lambda : gen_option_probability_normal(0.51, 0.1),
              '2_opt_normal2': lambda : gen_option_probability_normal(0.6, 0.25),
              '2_opt_normal3': lambda : gen_option_probability_normal(0.6, 0.1),
              '2_opt_beta1': lambda : gen_option_probability_beta(20,20),
              '2_opt_beta2': lambda : gen_option_probability_beta(21,20),
              '2_opt_beta3': lambda : gen_option_probability_beta(15,20),
              '4_opt_dirichlet1': lambda : gen_options_probability_dirichlet((2, 1, 1, 1))[0],
              '4_opt_dirichlet2': lambda : gen_options_probability_dirichlet((1.15, 1, 1, 1))[0],
              '4_opt_dirichlet3': lambda : gen_options_probability_dirichlet((4,3,2,1))[0],
             }


def plurality_vote(votes):
    tally  = Counter(votes)
    max_plurality_score = max(tally.values())
    winners = [o for o in tally.keys() if tally[o] == max_plurality_score]
    return winners 

def percent_plurality_vote_correct(model):
    num_correct = 0
    for r in range(model.num_rounds):
        winners = plurality_vote([a.vote() for a in model.schedule.agents])
        if len(winners) == 1 and model.options.C == winners[0]:
            num_correct += 1
    return float(num_correct) / model.num_rounds

def percent_expert_correct(model):
    num_correct = 0
    for r in range(model.num_rounds):
        expert = random.choice(model.schedule.agents)
        if  model.options.C == expert.vote():
            num_correct += 1
    return float(num_correct) / model.num_rounds


class Expert(Agent):
    """Expert to vote on a single proposition.
    competence: float between 0 and 1"""
    def __init__(self, unique_id, model, options, probs):
        super().__init__(unique_id, model)
        self.options = options
        self.probs = probs
        
    def vote(self):
        return np.random.choice(self.options, 1, p=self.probs)[0]
    
    def step(self):
        self.vote()
        #print(self.unique_id, self.selected_option)
    
class CJTModel(Model):
    """A model with some number of experts."""
    def __init__(self, N, num_rounds, gen_prob, num_options=2):
        self.num_experts = N
        self.options = Options(num_options)
        self.schedule = RandomActivation(self)
        self.num_rounds = num_rounds
        
        # Create experts
        for i in range(self.num_experts):
            a = Expert(i, self, self.options.props, gen_prob())
            self.schedule.add(a) 
        
        self.datacollector = DataCollector(
            model_reporters={"PercentPluralityCorrect": percent_plurality_vote_correct,
                             "PercentExpertCorrect": percent_expert_correct})
    
    def run(self):
        '''run simulation.'''
        
        self.schedule.step()
        self.datacollector.collect(self)



def display_plots(plot_type):
    
    max_num_agents = 101
    
    num_options = int(plot_type.split("_")[0]) 
    num_rounds = 500

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 6)
    perc_plurality_correct = list()
    perc_expert_correct = list()
    for num_agents in tqdm(range(1,max_num_agents+1)): 
        cjt_model = CJTModel(num_agents, 
                             num_rounds, 
                             init_probs[plot_type], 
                             num_options=num_options)
        cjt_model.run()
        winners = cjt_model.datacollector.get_model_vars_dataframe()
        perc_plurality_correct.append(winners.PercentPluralityCorrect.values[0])
        perc_expert_correct.append(winners.PercentExpertCorrect.values[0])
        
    _probs = list()
    for a in cjt_model.schedule.agents: 
        _probs.append(np.array(a.probs))
    probs = np.array(_probs)
    prs = probs.transpose()

    for opt in range(num_options):
        ax1.barh(range(num_agents), prs[opt], 1,
                 left=sum([np.array([0.0]*num_agents)] + [prs[i] for i in range(opt)]),
                  lw = 0.01)
    ax2.plot(range(1,max_num_agents+1), perc_plurality_correct, label="Plurality")
    ax2.plot(range(1,max_num_agents+1), perc_expert_correct, label="Expert")
    
    plt.legend(bbox_to_anchor=(1.25,0.5))
    plt.savefig("plurality_example.pdf")
    #plt.show()
    


p=interact_manual(display_plots,plot_type=widgets.Dropdown(
    options=[
        ('4 Options', '4_opt_fixed_probs'), 
        ('7 Options', '7_opt_fixed_probs'), 
        ('4 Options Random Competence 1', '4_opt_dirichlet1'),
        ('4 Options Random Competence 2', '4_opt_dirichlet2'),
        ('4 Options Random Competence 3', '4_opt_dirichlet3'), 
        ('1 Option Random Competence 1', '2_opt_beta1'),
        ('1 Option Random Competence 2', '2_opt_beta2'),
        ('1 Option Random Competence 3', '2_opt_beta3'),
        ('1 Option Random Competence 4', '2_opt_normal1'),
        ('1 Option Random Competence 5', '2_opt_normal2'),
        ('1 Option Random Competence 6', '2_opt_normal3'),

    ],
    value='4_opt_fixed_probs',
    description='Simulation:'))