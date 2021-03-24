'''
    File: voting_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: December 12, 2020
    
    Implementations of voting methods
'''

from voting.profiles import Profile, _borda_score, _find_updated_profile
from voting.generate_profiles import generate_profile
from itertools import permutations, product
import networkx as nx
import numpy as np
from numba import jit


'''TODO: 
   * implement tie-breaking and PUT for Hare and other iterative methods
   * to optimize the iterative methods, I am currently using the private compiled methods
     _borda_score and _find_updated_profiles. We should think of a better way to deal with this issue. 
   * implement other voting methods: e.g., Dodgson
   * implement the linear programming version of Ranked Pairs: https://arxiv.org/pdf/1805.06992.pdf
   * implement a Voting Method class?
'''

######
# Helper functions
######

# decorator that adds a "name" attribute to a voting method function
def vm_name(vm_name):
    def wrapper(f):
        f.name = vm_name
        return f
    return wrapper


@jit(fastmath=True)
def isin(arr, val):
    """compiled function testing if the value val is in the array arr
    """
    
    for i in range(arr.shape[0]):
        if (arr[i]==val):
            return True
    return False

@jit(nopython=True)
def _num_rank_first(rankings, rcounts, cands_to_ignore, cand):
    """The number of voters that rank candidate cand first after ignoring the candidates in 
    cands_to_ignore
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:  1d numpy array
        list of numbers of voters with the rankings  
    cands_to_ignore:   1d numpy array
        list of the candidates to ignore
    cand: int
        a candidate
    
    Key assumptions: 
        * the candidates are named 0...num_cands - 1, and c1 and c2 are 
          numbers between 0 and num_cands - 1
        * voters submit linear orders over the candidate        
    """
    
    num_voters = len(rankings)
    
    top_cands_indices = np.zeros(num_voters, dtype=np.int32)
    
    for vidx in range(num_voters): 
        for level in range(0, len(rankings[vidx])):
            if not isin(cands_to_ignore, rankings[vidx][level]):
                top_cands_indices[vidx] = level
                break                
    top_cands = np.array([rankings[vidx][top_cands_indices[vidx]] for vidx in range(num_voters)])
    is_cand = top_cands == cand # set to 0 each candidate not equal to cand
    return np.sum(is_cand * rcounts) 


@jit(nopython=True)
def _num_rank_last(rankings, rcounts, cands_to_ignore, cand):
    """The number of voters that rank candidate cand first after ignoring the candidates in 
    cands_to_ignore
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:  1d numpy array
        list of numbers of voters with the rankings  
    cands_to_ignore:   1d numpy array
        list of the candidates to ignore
    cand: int
        a candidate
    
    Key assumptions: 
        * the candidates are named 0...num_cands - 1, and c1 and c2 are 
          numbers between 0 and num_cands - 1
        * voters submit linear orders over the candidate        
    """
    
    num_voters = len(rankings)
    
    last_cands_indices = np.zeros(num_voters, dtype=np.int32)
    
    for vidx in range(num_voters): 
        for level in range(len(rankings[vidx]) - 1,-1,-1):
            if not isin(cands_to_ignore, rankings[vidx][level]):
                last_cands_indices[vidx] = level
                break                
    bottom_cands = np.array([rankings[vidx][last_cands_indices[vidx]] for vidx in range(num_voters)])
    is_cand = bottom_cands  == cand
    return np.sum(is_cand * rcounts) 

@vm_name("Majority")
def majority(profile):
    '''returns the majority winner is the candidate with a strict majority  of first place votes.   
    Returns an empty list if there is no candidate with a strict majority of first place votes.
    '''
    
    maj_size = profile.strict_maj_size()
    plurality_scores = profile.plurality_scores()
    maj_winner = [c for c in profile.candidates if  plurality_scores[c] >= maj_size]
    return sorted(maj_winner)


######
# Scoring rules
#####

@vm_name("Plurality")
def plurality(profile):
    """A plurality winnner a candidate that is ranked first by the most voters
    """

    plurality_scores = profile.plurality_scores()
    max_plurality_score = max(plurality_scores.values())
    
    return sorted([c for c in profile.candidates if plurality_scores[c] == max_plurality_score])

@vm_name("Borda")
def borda(profile):
    """A Borda winner is a candidate with the larget Borda score. 
    
    The Borda score of the candidates is calculated as follows: If there are $m$ candidates, then 
    the Borda score of candidate $c$ is \sum_{r=1}^{m (m - r) * Rank(c,r)$ where $Rank(c,r)$ is the 
    number of voters that rank candidate $c$ in position $r$. 
    """
    
    candidates = profile.candidates
    borda_scores = profile.borda_scores()
    max_borda_score = max(borda_scores.values())
    
    return sorted([c for c in candidates if borda_scores[c] == max_borda_score])

@vm_name("Anti-Plurality")
def anti_plurality(profile):
    """An anti-plurality winnner is a candidate that is ranked last by the fewest voters"""
    
    candidates, num_candidates = profile.candidates, profile.num_cands
    last_place_scores = {c: profile.num_rank(c,level=num_candidates) for c in candidates}
    min_last_place_score = min(list(last_place_scores.values()))
    
    return sorted([c for c in candidates if last_place_scores[c] == min_last_place_score])
    
######
# Iterative Methods
#####

@vm_name("Ranked Choice")
def hare(profile):
    """If there is a majority winner then that candidate is the ranked choice winner
    If there is no majority winner, then remove all candidates that are ranked first by the fewest 
    number of voters.  Continue removing candidates with the fewest number first-place votes until 
    there is a candidate with a majority of first place votes.  
    
    Note: If there is  more than one candidate with the fewest number of first-place votes, then *all*
    such candidates are removed from the profile. 
    
    Note: We typically refer to this method as "Ranked Choice", but it also known as "Hare" or "IRV"
    """
    
    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates 
                            if  not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if  plurality_scores[c] == min_plurality_score])

        # remove cands with lowest plurality winners
        cands_to_ignore = np.concatenate((cands_to_ignore, lowest_first_place_votes), axis=None)
        if len(cands_to_ignore) == num_cands: #all the candidates where removed
            winners = sorted(lowest_first_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]
     
    return sorted(winners)

@vm_name("PluralityWRunoff")
def plurality_with_runoff(profile):
    """If there is a majority winner then that candidate is the plurality with runoff winner
    If there is no majority winner, then hold a runoff with  the top two candidates: 
    either two (or more candidates)  with the most first place votes or the candidate with 
    the most first place votes and the candidate with the 2nd highest first place votes 
    are ranked first by the fewest number of voters.    
    
    Note: If the candidates are all tied for the most first place votes, then all candidates are winners. 
    """
    
    candidates = profile.candidates
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)
    plurality_scores = profile.plurality_scores()  

    max_plurality_score = max(plurality_scores.values())
    
    first = [c for c in candidates if plurality_scores[c] == max_plurality_score]
    
    if len(first) > 1:
        runoff_candidates = first
    else:
        # find the 2nd highest plurality score
        second_plurality_score = list(reversed(sorted(plurality_scores.values())))[1]
        second = [c for c in candidates if plurality_scores[c] == second_plurality_score]
        runoff_candidates = first + second
        
    runoff_candidates = np.array(runoff_candidates)
    candidates_to_ignore = np.array([c for c in candidates if not isin(runoff_candidates,c)])

    runoff_plurality_scores = {c: _num_rank_first(rs, rcounts, candidates_to_ignore, c) for c in candidates 
                               if isin(runoff_candidates,c)} 
    
    runoff_max_plurality_score = max(runoff_plurality_scores.values())
    
    return sorted([c for c in runoff_plurality_scores.keys() 
                   if runoff_plurality_scores[c] == runoff_max_plurality_score])

@vm_name("Coombs")
def coombs(profile):
    """If there is a majority winner then that candidate is the Coombs winner
    If there is no majority winner, then remove all candidates that are ranked last by the greatest 
    number of voters.  Continue removing candidates with the most last-place votes until 
    there is a candidate with a majority of first place votes.  
    
    Note: If there is  more than one candidate with the most number of last-place votes, then *all*
    such candidates are removed from the profile. 
    """
    
    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        
        last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates 
                             if not isin(cands_to_ignore,c)}  
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = np.array([c for c in last_place_scores.keys() 
                                              if  last_place_scores[c] == max_last_place_score])

        # remove candidates ranked last by the greatest number of voters
        cands_to_ignore = np.concatenate((cands_to_ignore, greatest_last_place_votes), axis=None)
        
        if len(cands_to_ignore) == num_cands:
            winners = list(greatest_last_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    return sorted(winners)

###
# Variations of Coombs with tie-breaking, parallel universe tie-breaking
##

@vm_name("Coombs TB")
def coombs_tb(profile, tie_breaker=None):
    """Coombs with a fixed tie-breaking rule:  If there is a majority winner then that candidate 
    is the Coombs winner.  If there is no majority winner, then remove all candidates that 
    are ranked last by the greatest  number of voters.   If there are ties, then choose the candidate
    according to a fixed tie-breaking rule (given below). Continue removing candidates with the 
    most last-place votes until     there is a candidate with a majority of first place votes.  
    
    The tie-breaking rule is any linear order (i.e., list) of the candidates.  The default rule 
    is to order the candidates as follows: 0,....,num_cands-1

    """
    
    # the tie_breaker is any linear order (i.e., list) of the candidates
    tb = tie_breaker if tie_breaker is not None else list(range(profile.num_cands))

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        
        last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates 
                             if not isin(cands_to_ignore,c)}  
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = [c for c in last_place_scores.keys() 
                                     if  last_place_scores[c] == max_last_place_score]

        # select the candidate to remove using the tie-breaking rule (a linear order over the candidates)
        cand_to_remove = greatest_last_place_votes[0]
        for c in greatest_last_place_votes[1:]: 
            if tb.index(c) < tb.index(cand_to_remove):
                cand_to_remove = c
        
        # remove candidates ranked last by the greatest number of voters
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array([cand_to_remove])), axis=None)
        
        if len(cands_to_ignore) == num_cands:
            winners = list(greatest_last_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    return sorted(winners)

@vm_name("Coombs PUT")
def coombs_put(profile):
    """Coombs with parallel universe tie-breaking (PUT).  Apply the Coombs method with a tie-breaker
    for each possible linear order over the candidates. 
    
    Warning: This will take a long time on profiles with many candidates."""
    
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, np.empty(0), c) >= strict_maj_size]

    if len(winners) == 0:
        # run Coombs with tie-breaker for each permulation of candidates
        for tb in permutations(candidates):
            winners += coombs_tb(profile, tie_breaker = tb) 

    return sorted(list(set(winners)))

###

@vm_name("Baldwin")
def baldwin(profile):
    """Iteratively remove all candidates with the lowest Borda score until a single 
    candidate remains.   If, at any stage, all  candidates have the same Borda score, 
    then all (remaining) candidates are winners.
    """

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    borda_scores = {c: _borda_score(rs, rcounts, num_cands, c) for c in candidates}

    min_borda_score = min(list(borda_scores.values()))
    
    last_place_borda_scores = [c for c in candidates 
                               if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
      
    cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] ==  num_cands: # call candidates have lowest Borda score
        winners = sorted(last_place_borda_scores)
    else: # remove the candidates with lowest Borda score
        updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    while len(winners) == 0:
        borda_scores = {c: _borda_score(updated_profile, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore,c)}
                
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in borda_scores.keys() if borda_scores[c] == min_borda_score]
        
        cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
                
        if cands_to_ignore.shape[0] == num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    return sorted(winners)

@vm_name("Strict Nanson")
def strict_nanson(profile):
    """Iteratively remove all candidates with the  Borda score strictly below the average Borda score
    until one candidate remains.   If, at any stage, all  candidates have the same Borda score, 
    then all (remaining) candidates are winners.
    """

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    borda_scores = {c: _borda_score(rs, rcounts, num_cands, c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = np.array([c for c in borda_scores.keys() if borda_scores[c] < avg_borda_score])
    
    cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
    
    if cands_to_ignore.shape[0] == num_cands:  # all candidates have same Borda score
        winners = sorted(candidates)
    else: 
        winners = list()
        updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_profile, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore,c)}
        avg_borda_scores = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                               if borda_scores[c] < avg_borda_scores])
        
        cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
                
        if (below_borda_avg_candidates.shape[0] == 0) or ((num_cands - cands_to_ignore.shape[0]) == 1):
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else:
            updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
            
    return winners


@vm_name("Weak Nanson")
def weak_nanson(profile):
    """Iteratively remove all candidates with the  Borda score less than or equal to the average Borda score
    until one candidate remains.   If, at any stage, all  candidates have the same Borda score, 
    then all (remaining) candidates are winners.
    """

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    borda_scores = {c: _borda_score(rs, rcounts, num_cands, c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                           if borda_scores[c] <= avg_borda_score])
    
    cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
    
    if cands_to_ignore.shape[0] == num_cands:  # all candidates have same Borda score
        winners = sorted(candidates)
    else: 
        winners = list()
        updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_profile, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore,c)}
        avg_borda_scores = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                               if borda_scores[c] <= avg_borda_scores])
        
        cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
                
        if (num_cands - cands_to_ignore.shape[0]) == 0:
            winners = sorted(below_borda_avg_candidates)
        elif (num_cands - cands_to_ignore.shape[0]) == 1:
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else:
            updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
            
    return winners

####
# Majority Graph Invariant Methods
#
# For each method, there is a second version appended with "_mg" that assumes that the 
# input is a margin graph (represented as a networkx graph)
####


####
# Helper functions for reasoning about margin graphs
####

def generate_weak_margin_graph(profile):
    '''generate the weak weighted margin graph, where there is an edge if the margin is greater than or 
    equal to 0.'''
    mg = nx.DiGraph()
    candidates = profile.candidates
    mg.add_nodes_from(candidates)

    mg.add_weighted_edges_from([(c1,c2,profile.margin(c1,c2))  
           for c1 in candidates 
           for c2 in candidates if c1 != c2 if profile.margin(c1, c2) >= 0])
    return mg

# flatten a 2d list - turn a 2d list into a single list of items
flatten = lambda l: [item for sublist in l for item in sublist]

def has_cycle(mg):
    """true if the margin graph mg has a cycle"""
    try:
        cycles =  nx.find_cycle(mg)
    except:
        cycles = list()
    return len(cycles) != 0

def unbeaten_candidates(mg): 
    """the set of candidates with no incoming arrows, i.e., the 
    candidates that are unbeaten"""
    return [n for n in mg.nodes if mg.in_degree(n) == 0]

def find_condorcet_winner(mg): 
    """the Condorcet winner is the candidate with an edge to every other candidate"""
    return [n for n in mg.nodes if mg.out_degree(n) == len(mg.nodes) -  1]

def find_weak_condorcet_winners(mg):
    """weak condorcet winners are candidates with no incoming edges"""
    return unbeaten_candidates(mg)

def find_condorcet_losers(mg):
    """A Condorcet loser is the candidate with incoming edges from every other candidate"""
    
    # edge case: there is no Condorcet loser if there is only one node
    if len(mg.nodes) == 1:
        return []
    return [n for n in mg.nodes if mg.in_degree(n) == len(mg.nodes) - 1]

def is_majority_preferred(mg, c1, c2): 
    """true if c1 is majority preferred to c2, i.e., there is an edge from c1 to c2"""
    return mg.has_edge(c1, c2)

def is_tied(mg, c1, c2): 
    """true if there is no edge between c1 and c2"""    
    return not mg.has_edge(c1, c2) and not mg.has_edge(c2, c1)

@vm_name("Condorcet")
def condorcet(profile):
    """Return the Condorcet winner if one exists, otherwise return all the candidates"""
    
    cond_winner = profile.condorcet_winner()
    return [cond_winner] if cond_winner is not None else sorted(profile.candidates)

@vm_name("Condorcet")
def condorcet_mg(mg):
    """Return the Condorcet winner if one exists, otherwise return all the candidates"""
    
    cond_winner = find_condorcet_winner(mg)
    return cond_winner if len(cond_winner) > 0 else sorted(mg.nodes)


@vm_name("Copeland")
def copeland(profile): 
    """The Copeland score for c is the number of candidates that c is majority preferred to 
    minus the number of candidates majority preferred to c.   The Copeland winners are the candidates
    with the max Copeland score."""
    
    candidates = profile.candidates
    copeland_scores = {c:len([1 for c2 in candidates if profile.margin(c,c2) > 0]) - 
                       len([1 for c2 in candidates if profile.margin(c,c2) < 0]) 
                       for c in candidates}
    max_copeland_score = max(copeland_scores.values())
    return sorted([c for c in candidates if copeland_scores[c] == max_copeland_score])

@vm_name("Llull")
def llull(profile): 
    """The Llull score for a candidate c is the number of candidates that c is weakly majority 
    preferred to.   The Llull winners are the candidates with the greatest Llull score."""
    
    candidates = profile.candidates
    llull_scores = {c:len([1 for c2 in candidates if profile.margin(c,c2) >= 0])
                    for c in candidates}
    max_llull_score = max(llull_scores.values())
    return sorted([c for c in candidates if llull_scores[c] == max_llull_score])


######
# Copeland/Llull on margin graphs
######
def copeland_scores(mg, alpha=0.5):
    """Copeland alpha score of candidate c is: 1 point for every candidate c2 that c is majority 
    preferred to and alpha points for every candidate that c is tied with."""
    c_scores = {c: 0.0 for c in mg.nodes}
    for c in mg.nodes:
        for c2 in mg.nodes:
            if c != c2 and is_majority_preferred(mg, c, c2):
                c_scores[c] += 1.0
            if c != c2 and is_tied(mg, c, c2): 
                c_scores[c] += alpha
    return c_scores

@vm_name("Copeland")    
def copeland_mg(mg): 
    """Copeland winners are the candidates with maximum Copeland_alpha socre with alpha=0.5"""
    c_scores = copeland_scores(mg)
    max_score = max(c_scores.values())
    return sorted([c for c in c_scores.keys() if c_scores[c] == max_score])
      
@vm_name("Llull")    
def llull_mg(mg): 
    """Llull winners are the candidates with maximum Copeland_alpha socre with alpha=1"""
    c_scores = copeland_scores(mg, alpha=1)
    max_score = max(c_scores.values())
    return sorted([c for c in c_scores.keys() if c_scores[c] == max_score])


## Uncovered Set

def left_covers(dom, c1, c2):
    # c1 left covers c2 when all the candidates that are majority preferred to c1
    # are also majority preferred to c2. 
    #
    # dom is a dictionary listing for each candidate, the set of candidates majority 
    # prefeerred to that candidate. 
    return dom[c1].issubset(dom[c2])

@vm_name("Uncovered Set")
def uc_gill_mg(mg): 
    """(Gillies version)   Given candidates a and b, say that a defeats b in the profile P, a defeats b 
    if a is majority preferred to b and a left covers b: i.e., for all c, if c is majority preferred to a, 
    then c majority preferred to b. Then the winners are the set of  candidates who are undefeated in P. 
    """
    
    dom = {n: set(mg.predecessors(n)) for n in mg.nodes}
    uc_set = list()
    for c1 in mg.nodes:
        is_in_ucs = True
        for c2 in mg.predecessors(c1): # consider only c2 predecessors
            if c1 != c2:
                # check if c2 left covers  c1 
                if left_covers(dom, c2, c1):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))

@vm_name("Uncovered Set")
def uc_gill(profile): 
    '''See the explanation for uc_gill_mg'''
    
    mg = profile.margin_graph() 
    return uc_gill_mg(mg)

@vm_name("Uncovered Set - Fishburn")
def uc_fish_mg(mg): 
    """(Fishburn version)  Given candidates a and b, say that a defeats b in the profile P
    if a left covers b: i.e., for all c, if c is majority preferred to a, then c majority preferred to b, and
    b does not left cover a. Then the winners are the set of candidates who are undefeated."""
    
    dom = {n: set(mg.predecessors(n)) for n in mg.nodes}
    uc_set = list()
    for c1 in mg.nodes:
        is_in_ucs = True
        for c2 in mg.nodes:
            if c1 != c2:
                # check if c2 left covers  c1 but c1 does not left cover c2
                if left_covers(dom, c2, c1)  and not left_covers(dom, c1, c2):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))
                
@vm_name("Uncovered Set - Fishburn")
def uc_fish(profile): 
    """See the explaination of uc_fish_mg"""
    
    mg = profile.margin_graph() 
    return uc_fish_mg(mg)

@vm_name("GETCHA")
def getcha_mg(mg):
    """The smallest set of candidates such that every candidate inside the set 
    is majority preferred to every candidate outside the set.  Also known as the Smith set.
    """
    min_indegree = min([max([mg.in_degree(n) for n in comp]) for comp in nx.strongly_connected_components(mg)])
    smith = [comp for comp in nx.strongly_connected_components(mg) if max([mg.in_degree(n) for n in comp]) == min_indegree][0]
    return sorted(list(smith))

@vm_name("GETCHA")
def getcha(profile):
    """See the explanation of getcha_mg"""
    mg = generate_weak_margin_graph(profile)
    return getcha_mg(mg)


@vm_name("GOCHA")
def gocha_mg(mg):
    """The GOCHA set (also known as the Schwartz set) is the smallest set of candidates with the property
    that every candidate inside the set is not majority preferred by every candidate outside the set. 
    """
    transitive_closure =  nx.algorithms.dag.transitive_closure(mg)
    schwartz = set()
    for ssc in nx.strongly_connected_components(transitive_closure):
        if not any([transitive_closure.has_edge(c2,c1) 
                    for c1 in ssc for c2 in transitive_closure.nodes if c2 not in ssc]):
            schwartz =  schwartz.union(ssc)
    return sorted(list(schwartz))

@vm_name("GOCHA")
def gocha(profile):
    """See the explanation of gocha_mg""" 
    mg = profile.margin_graph()
    return gocha_mg(mg)


#####
# (Qualitative) Margin Graph invariant methods
#####

@vm_name("Minimax")
def minimax(profile, score_method="winning"):
    """Return the candidates with the smallest maximum pairwise defeat.  That is, for each 
    candidate c determine the biggest margin of a candidate c1 over c, then select 
    the candidates with the smallest such loss. Alson known as the Simpson-Kramer Rule.
    """
    
    candidates = profile.candidates
    
    if len(candidates) == 1:
        return candidates
    
    # there are different scoring functions that can be used to measure the worse loss for each 
    # candidate. These all produce the same set of winners when voters submit linear orders. 
    score_functions = {
        "winning": lambda c1,c2: profile.support(c1,c2) if profile.support(c1,c2) > profile.support(c2,c1) else 0,
        "margins": lambda c1,c2: profile.support(c1,c2)   -  profile.support(c2,c1),
        "pairwise_opposition": lambda c1,c2: profile.support(c1,c2)
    } 
    scores = {c: max([score_functions[score_method](_c,c) for _c in candidates if _c != c]) 
              for c in candidates}
    min_score = min(scores.values())
    return sorted([c for c in candidates if scores[c] == min_score])

@vm_name("Split Cycle") 
def split_cycle(profile):
    """A *majority cycle in a profile P is a sequence x_1,...,x_n of distinct candidates in 
    P with x_1=x_n such that for 1 <= k <= n-1,  x_k is majority preferred to x_{k+1}.
    The *strength of* a majority is the minimal margin in the cycle.  
    Say that a defeats b in P if the margin of a over b is positive and greater than 
    the strength of the strongest majority cycle containing a and b. The Split Cycle winners
    are the undefeated candidates.
    """
    
    candidates = profile.candidates 
    
    # create the margin graph
    mg = profile.margin_graph()
    
    # find the cycle number for each candidate
    cycle_number = {cs:0 for cs in permutations(candidates,2)}
    for cycle in nx.simple_cycles(mg): # for each cycle in the margin graph

        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(mg[c1][c2]['weight'])
            
        split_number = min(margins) # the split number of the cycle is the minimal margin
        for c1,c2 in cycle_number.keys():
            c1_index = cycle.index(c1) if c1 in cycle else -1
            c2_index = cycle.index(c2) if c2 in cycle else -1

            # only need to check cycles with an edge from c1 to c2
            if (c1_index != -1 and c2_index != -1) and ((c2_index == c1_index + 1) or (c1_index == len(cycle)-1 and c2_index == 0)):
                cycle_number[(c1,c2)] = split_number if split_number > cycle_number[(c1,c2)] else cycle_number[(c1,c2)]        

    # construct the defeat relation, where a defeats b if margin(a,b) > cycle_number(a,b) (see Lemma 3.13)
    defeat = nx.DiGraph()
    defeat.add_nodes_from(candidates)
    defeat.add_edges_from([(c1,c2)  
           for c1 in candidates 
           for c2 in candidates if c1 != c2 if profile.margin(c1,c2) > cycle_number[(c1,c2)]])

    # the winners are candidates not defeated by any other candidate
    winners = unbeaten_candidates(defeat)
    
    return sorted(list(set(winners)))

@vm_name("Split Cycle")
def split_cycle_faster(profile):   
    """Implementation of Split Cycle using a variation of the Floyd Warshall-Algorithm  
    """
    candidates = profile.candidates
    weak_condorcet_winners = {c:True for c in candidates}
    mg = [[-np.inf for _ in candidates] for _ in candidates]
    
    # Weak Condorcet winners are Split Cycle winners
    for c1 in candidates:
        for c2 in candidates:
            if (profile.support(c1,c2) > profile.support(c2,c1) or c1 == c2):
                mg[c1][c2] = profile.support(c1,c2) - profile.support(c2,c1)
                weak_condorcet_winners[c2] = weak_condorcet_winners[c2] and (c1 == c2)
    
    strength = list(map(lambda i : list(map(lambda j : j , i)) , mg))
    for i in candidates:         
        for j in candidates: 
            if i!= j:
                if not weak_condorcet_winners[j]: # weak Condorcet winners are Split Cycle winners
                    for k in candidates: 
                        if i!= k and j != k:
                            strength[j][k] = max(strength[j][k], min(strength[j][i],strength[i][k]))
    winners = {i:True for i in candidates}
    for i in candidates: 
        for j in candidates:
            if i!=j:
                if mg[j][i] > strength[i][j]: # the main difference with Beat Path
                    winners[i] = False
    return sorted([c for c in candidates if winners[c]])


@vm_name("Iterated Split Cycle")
def iterated_splitcycle(prof):
    '''Iteratively calculate the split cycle winners until there is a
    unique winner or all remaining candidates are split cycle winners'''
    

    sc_winners = split_cycle_faster(prof)
    orig_cnames = {c:c for c in prof.candidates}
    
    reduced_prof = prof
    
    while len(sc_winners) != 1 and sc_winners != list(reduced_prof.candidates): 
        reduced_prof, orig_cnames = prof.remove_candidates([c for c in prof.candidates if c not in sc_winners])
        sc_winners = split_cycle_faster(reduced_prof)
        
    return sorted([orig_cnames[c] for c in sc_winners])

@vm_name("Beat Path")
def beat_path(profile): 
    """For candidates a and b, a *path from a to b in P* is a sequence 
    x_1,...,x_n of distinct candidates in P with  x_1=a and x_n=b such that 
    for 1 <= k <= n-1$, x_k is majority preferred to x_{k+1}.  The *strength of a path* 
    is the minimal margin along that path.  Say that a defeats b in P if 
    the strength of the strongest path from a to b is greater than the strength of 
    the strongest path from b to a. Then Beat Path winners are the undefeated candidates. 
    Also known as the Schulze Rule.
    """
    
    #1. calculate vote_graph, edge from c1 to c2 of c1 beats c2, weighted by support for c1 over c2
    #2. For all pairs c1, c2, find all paths from c1 to c2, for each path find the minimum weight.  
    #   beatpath[c1,c2] = max(weight(p) all p's from c1 to c2)
    #3. winner is the candidates that beat every other candidate 

    candidates = profile.candidates
    mg = profile.margin_graph()
    beat_paths_weights = {c: {c2:0 for c2 in candidates if c2 != c} for c in candidates}
    for c in candidates: 
        for other_c in beat_paths_weights[c].keys():
            all_paths =  list(nx.all_simple_paths(mg, c, other_c))
            if len(all_paths) > 0:
                beat_paths_weights[c][other_c] = max([min([mg[p[i]][p[i+1]]['weight'] 
                                                           for i in range(0,len(p)-1)]) 
                                                      for p in all_paths])
    
    winners = list()
    for c in candidates: 
        if all([beat_paths_weights[c][c2] >= beat_paths_weights[c2][c] 
                for c2 in candidates  if c2 != c]):
            winners.append(c)

    return sorted(list(winners))

@vm_name("Beat Path")
def beat_path_faster(profile):   
    """Implementation of Beat Path using a variation of the Floyd Warshall-Algorithm
    See https://en.wikipedia.org/wiki/Schulze_method#Implementation
    """
    
    candidates = profile.candidates
    
    mg = [[-np.inf for _ in candidates] for _ in candidates]
    for c1 in candidates:
        for c2 in candidates:
            if (profile.support(c1,c2) > profile.support(c2,c1) or c1 == c2):
                mg[c1][c2] = profile.support(c1,c2) - profile.support(c2,c1)
    strength = list(map(lambda i : list(map(lambda j : j , i)) , mg))
    for i in candidates:         
        for j in candidates: 
            if i!= j:
                for k in candidates: 
                    if i!= k and j != k:
                        strength[j][k] = max(strength[j][k], min(strength[j][i],strength[i][k]))
    winners = {i:True for i in candidates}
    for i in candidates: 
        for j in candidates:
            if i!=j:
                if strength[j][i] > strength[i][j]:
                    winners[i] = False
    return sorted([c for c in candidates if winners[c]])

@vm_name("Ranked Pairs")
def ranked_pairs(profile):
    """Order the edges in the weak margin graph from largest to smallest and lock them 
    in in that order, skipping edges that create a cycle.   Break ties using a tie-breaking 
    linear ordering over the edges.   A candidate is a Ranked Pairs winner if it wins according 
    to some tie-breaking rule. Also known as Tideman's Rule.
    """
    wmg = generate_weak_margin_graph(profile)
    cw = profile.condorcet_winner()
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        winners = list()            
        margins = sorted(list(set([e[2]['weight'] for e in wmg.edges(data=True)])), reverse=True)
        sorted_edges = [[e for e in wmg.edges(data=True) if e[2]['weight'] == w] for w in margins]
        tbs = product(*[permutations(edges) for edges in sorted_edges])
        for tb in tbs:
            edges = flatten(tb)
            new_ranking = nx.DiGraph() 
            for e in edges: 
                new_ranking.add_edge(e[0], e[1], weight=e[2]['weight'])
                if  has_cycle(new_ranking):
                    new_ranking.remove_edge(e[0], e[1])
            winners.append(unbeaten_candidates(new_ranking)[0])
    return sorted(list(set(winners)))


#####
# Other methods
#####

@vm_name("Iterated Removal Condorcet Loser")
def iterated_remove_cl(profile):
    """The winners are the candidates that survive iterated removal of 
    Condorcet losers
    """
    
    condorcet_loser = profile.condorcet_loser()  
    
    updated_profile = profile
    orig_cnames = {c:c for c in profile.candidates}
    while len(updated_profile.candidates) > 1 and  condorcet_loser is not None:    
        updated_profile, _orig_cnames = updated_profile.remove_candidates([condorcet_loser])
        orig_cnames = {c:orig_cnames[cn] for c,cn in _orig_cnames.items()}
        condorcet_loser = updated_profile.condorcet_loser()
            
    return sorted([orig_cnames[c] for c in updated_profile.candidates])


@vm_name("Daunou")
def daunou(profile):
    """Implementaiton of Daunou's voting method as described in the paper: 
    https://link.springer.com/article/10.1007/s00355-020-01276-w
    
    If there is a Condorcet winner, then that candidate is the winner.  Otherwise, 
    iteratively remove all Condorcet losers then select the plurality winner from among 
    the remaining conadidates
    """

    cw = profile.condorcet_winner()
    
    if cw is not None: 
        updated_profile = profile
        orig_cnames = {c:c for c in profile.candidates}
        winners = [cw]
    else: 
        cands_survive_it_rem_cl = iterated_remove_cl(profile)
        updated_profile, orig_cnames = profile.remove_candidates([_c for _c in profile.candidates 
                                                                  if _c not in cands_survive_it_rem_cl])
        winners = plurality(updated_profile)
        
    return sorted([orig_cnames[c] for c in winners])


@vm_name("Blacks")
def blacks(profile):
    """Blacks method returns the Condorcet winner if one exists, otherwise return the Borda winners.
    """
    
    cw = profile.condorcet_winner()
    
    if cw is not None:
        winners = [cw]
    else:
        winners = borda(profile)
        
    return winners



all_vms = [
    plurality,
    borda, 
    anti_plurality,
    hare, 
    plurality_with_runoff,
    coombs,
    coombs_tb,
    coombs_put,
    baldwin,
    strict_nanson, 
    weak_nanson,
    condorcet,
    copeland,
    llull,
    uc_gill,
    uc_fish,
    getcha,
    gocha,
    minimax, 
    split_cycle,
    split_cycle_faster,
    beat_path,
    beat_path_faster,
    ranked_pairs,
    iterated_remove_cl,
    daunou,
    blacks 
]

all_vms_mg = [
    condorcet_mg,
    copeland_mg,
    llull_mg,
    uc_gill_mg,
    uc_fish_mg,
    getcha_mg,
    gocha_mg
]

    
    