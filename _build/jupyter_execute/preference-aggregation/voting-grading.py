# Voting by Grading

# import the Profile class
from voting.profiles import Profile
from voting.generate_profiles import *
from voting.voting_methods import *
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from itertools import combinations, permutations

from tqdm.notebook import tqdm

Steven J. Brams and Richard F. Potthoff (2015). [The paradox of grading systems](https://link.springer.com/article/10.1007/s11127-015-0303-6), Public Choice volume 165,  pp. 193–210.

class wRanking(object):
    
    def __init__(self, cmap, ranks = None, r_str = None): 

        self._cand_to_cnum = cmap
        self._cnum_to_cand = {cnum: c for c, cnum in cmap.items()} 
        self._ranks = ranks  
        self._r_str = r_str 
        if self._ranks is None: 
            self.set_rank_list_from_str(r_str, cmap)
        if self._r_str is None: 
            self.set_rank_str_from_rank_list(ranks, self._cnum_to_cand)
    @property
    def rank_list(self): 
        return self._ranks
    
    @property
    def rank_dict(self): 
        return {self._cnum_to_cand[cnum]:  self._ranks[cnum] for cnum in range(len(self._ranks))}
        
    @property
    def alts(self): 
        return sorted(self._cand_to_cnum.keys())
    
    @property
    def num_alts(self): 
        return len(self._cand_to_cnum.keys())
        
    @property
    def max_rank(self): 
        return self.num_alts
    
    def rank(self, c):
        
        return self._ranks[self._cand_to_cnum[c]]
    
    def cands_at_rank(self, rank): 
        
        return [self._cnum_to_cand[cnum] for cnum,r in enumerate(self._ranks) if r == rank]
    
    def R(self, c1, c2): 
        
        c1_rank = self.rank(c1)
        c2_rank = self.rank(c2)
        return (c2_rank is None) or (c1_rank is not None and (c1_rank <= c2_rank))
                
    def P(self, c1, c2): 
        
        c1_rank = self.rank(c1)
        c2_rank = self.rank(c2)
        return (c2_rank is None and c1_rank is not None) or (c1_rank is not None and (c1_rank < c2_rank))
    
    def I(self, c1, c2): 
        
        return self.rank(c1) == self.rank(c2)
    
    def lift(self, cand, rank):
        """assumes that the ballot is complete and the only cands with no rank are at the bottom of the ranking"""
        c_rank =  self._ranks[self._cand_to_cnum[cand]]
        for _c in self._cand_to_cnum.keys():
            if self._ranks[self._cand_to_cnum[_c]] is not None and self._ranks[self._cand_to_cnum[_c]] <= c_rank:
                self._ranks[self._cand_to_cnum[_c]] += 1
        self._ranks[self._cand_to_cnum[cand]] = rank
        self.set_rank_str_from_rank_list(self._ranks, self._cnum_to_cand)        
        
    def judgement_set(self): 
        j_set = dict()
        for c in self.alts: 
            for d in self.alts:
                if c != d: 
                    if self.P(c,d): 
                        j_set[(c,d)] = True
                        j_set[(d,c)] = False
                    elif self.P(d,c): 
                        j_set[(c,d)] = False
                        j_set[(d,c)] = True  
                    elif self.I(c,d): 
                        j_set[(c,d)] = True
                        j_set[(d,c)] = True
        return j_set
        
    def set_rank_list_from_str(self, r_str, cmap):
        
        cname_type = type(list(cmap.keys())[0])
        bits = r_str.strip().split(",")
        ranks = [None] * len(cmap.keys())
        pos  = 0
        rank = 1
        partial = False
        while pos < len(bits): 
            if bits[pos].find("{") != -1:
                partial = True
                ranks[cmap[cname_type(bits[pos].replace("{",""))]] = rank
                pos += 1
            elif bits[pos].find("}") != -1:
                ranks[cmap[cname_type(bits[pos].replace("}",""))]] = rank
                rank += 1
                partial = False
                pos += 1
            else: 
                ranks[cmap[cname_type(bits[pos])]] = rank
                if not partial: 
                    rank += 1
                pos += 1    
        self._ranks = ranks
        
    
    def set_rank_str_from_rank_list(self, r_list, cnum_to_cand):
        
        r_str = ''
        for r in range(1,len(cnum_to_cand.keys()) + 1): 
            cs_at_rank = [str(cnum_to_cand[c]) for c,cr in enumerate(r_list) if cr == r]
            if len(cs_at_rank) == 1: 
                r_str += f"{cs_at_rank[0]},"
            elif len(cs_at_rank) > 1: 
                r_str  += "{" + ",".join(cs_at_rank) + "},"
        self._r_str = r_str[0:-1] # remove last ,

    def __str__(self): 
        return self._r_str



from tabulate  import tabulate
class wProfile(object):
    """
    A profile of strict weak orderings

    """
    def __init__(self, rankings, num_cands, rcounts=None, cmap=None, ranking_strings = True):
        """
        Create a profile
                
        """
        
        self.num_cands = num_cands
        self.cand_nums = range(0, num_cands) 
        
        self._cand_to_cnum = cmap if cmap else {c:c for c in range(num_cands)}
        self._cnum_to_cand = {cnum: c for c,cnum in self._cand_to_cnum.items()}
        
        if ranking_strings:
            self._rankings = [wRanking(self._cand_to_cnum, r_str = r_str) for r_str in rankings]
        else: 
            self._rankings = rankings
        self._rcounts = rcounts if rcounts is not None else [1] * len(rankings)

        # total number of voters
        self.num_voters = np.sum(self._rcounts)

    @property
    def rankings_counts(self):
        # getter function to get the rankings and rcounts
        return self._rankings, self._rcounts
    
    @property
    def rankings(self): 
        # get the list of rankings
        return [r for ridx,r in enumerate(self._rankings) for n in range(self._rcounts[ridx])]
    
    @property 
    def candidates(self): 
        return sorted(list(self._cand_to_cnum.keys()))
    
    def support(self, c1, c2):
        # the number of voters that rank c1 strictly above c2 
        # wrapper function that calls the compiled _support function

        return sum([rcount if r.P(c1,c2) else 0.0 for r,rcount in zip(self._rankings, self._rcounts)])
    
    def margin(self, c1, c2):
        # the number of voters that rank c1 over c2 minus the number
        #   that rank c2 over c2.
        # wrapper function that calls the compiled _margin function

        return self.support(c1, c2) - self.support(c2, c1)
        
    def majority_prefers(self, c1, c2): 
        # return True if more voters rank c1 over c2 than c2 over c1

        return self.margin(c1, c2) > 0

    def condorcet_winner(self):
        # return the Condorcet winner --- a candidate that is majority preferred to every other candidate
        # if a Condorcet winner doesn't exist, return None
        
        cw = None
        for c in self.candidates: 
            if all([self.majority_prefers(c,c2) for c2 in self.candidates if c != c2]): 
                cw = c
                break # if a Condorcet winner exists, then it is unique
        return cw

    def weak_condorcet_winner(self):
        # return the set of Weak Condorcet winner --- candidate c is a weak Condorcet winner if there 
        # is no other candidate majority preferred to c. Note that unlike with Condorcet winners, there 
        # may be more than one weak Condorcet winner.
        # if a weak Condorcet winner doesn't exist, return None
        
        weak_cw = list()
        for c in self.candidates: 
            if not any([self.majority_prefers(c2,c) for c2 in self.candidates if c != c2]): 
                weak_cw.append(c)
        return sorted(weak_cw) if len(weak_cw) > 0 else None

    def condorcet_loser(self):
        # return the Condorcet loser --- a candidate that is majority preferred by every other candidate
        # if a Condorcet loser doesn't exist, return None
        
        cl = None
        for c in self.candidates: 
            if all([self.majority_prefers(c2,c) for c2 in self.candidates if c != c2]): 
                cl = c
                break # if a Condorcet loser exists, then it is unique
        return cl
    
    def top_cycle(self):
        """The smallest set of candidates such that every candidate inside the set 
        is majority preferred to every candidate outside the set.  Also known as the Smith set.
        """
        mg = self.weak_margin_graph()
        min_indegree = min([max([mg.in_degree(n) for n in comp]) for comp in nx.strongly_connected_components(mg)])
        smith = [comp for comp in nx.strongly_connected_components(mg) if max([mg.in_degree(n) for n in comp]) == min_indegree][0]
        return sorted(list(smith))
    
    def strict_maj_size(self):
        # return the size of  strictly more than 50% of the voters
        
        return int(self.num_voters/2 + 1 if self.num_voters % 2 == 0 else int(ceil(float(self.num_voters)/2)))

    def weak_margin_graph(self, cmap=None): 
        # generate the margin graph (i.e., the weighted majority graph)
    
        mg = nx.DiGraph()
        mg.add_nodes_from(self.candidates)
        mg.add_weighted_edges_from([(c1, c2, self.margin(c1,c2))
                                    for c1 in self.candidates 
                                    for c2 in self.candidates if c1 != c2 if self.majority_prefers(c1, c2) or self.margin(c1,c2)== 0])
        return mg

    def margin_graph(self, cmap=None): 
        # generate the margin graph (i.e., the weighted majority graph)
    
        mg = nx.DiGraph()
        mg.add_nodes_from(self.candidates)
        mg.add_weighted_edges_from([(c1, c2, self.margin(c1,c2))
                                    for c1 in self.candidates 
                                    for c2 in self.candidates if c1 != c2 if self.majority_prefers(c1, c2)])
        return mg

    
    def display(self, cmap=None, style="pretty"):
        # display a profile
        # style defaults to "pretty" (the PrettyTable formatting)
        # other stype options is "latex" or "fancy_grid" (or any style option for tabulate)
        
        
        possible_ranks = range(1, self.num_cands + 1)
        
        tbl = list()
        
        for r in possible_ranks: 
            tbl.append([",".join(map(str,wr.cands_at_rank(r))) for wr in self._rankings])
        print(tabulate(tbl,
                       self._rcounts, 
                       tablefmt=style))        
        
    def display_margin_graph(self, cmap=None):
        # display the margin graph
        
        # create the margin graph.   The reason not to call the above method margin_graph 
        # is that we may want to apply the cmap to the names of the candidates
        
        
        mg = nx.DiGraph()
        mg.add_nodes_from(self.candidates)
        mg.add_weighted_edges_from([(c1, c2, self.margin(c1,c2))
                                    for c1 in self.candidates 
                                    for c2 in self.candidates if c1 != c2 if self.majority_prefers(c1, c2)])

        pos = nx.circular_layout(mg)
        nx.draw(mg, pos, 
                font_size=20, node_color='blue', font_color='white', node_size=700, 
                width=1, lw=1.5, with_labels=True)
        labels = nx.get_edge_attributes(mg,'weight')
        nx.draw_networkx_edge_labels(mg,pos,edge_labels=labels, label_pos=0.3)
        plt.show()

num_cands = 4
rankings = ["3,0,1,2", "0,3,{1,2}", "{0,1,2}", "2,3,{0,1}"]

prf = wProfile(rankings, num_cands)

prf.display()
print(prf.support(0,1))
print(prf.support(0,2))
print(prf.support(1,2))

print("prf.margin(0,1)", prf.margin(0,1))
print("prf.margin(1,0)", prf.margin(1,0))
print("prf.margin(0,2)", prf.margin(0,2))
print("prf.margin(2,0)", prf.margin(2,0))
print("prf.margin(1,2)", prf.margin(1,2))
print("prf.margin(2,1)", prf.margin(2,1))

print(prf.condorcet_winner())

prf.display_margin_graph()


class Grade(object):
    
    def __init__(self, grades, possible_grades = None):
        
        self.grades = grades
        self.candidates = sorted(self.grades.keys())
        self.possible_grades = possible_grades if possible_grades is not None else range(min(grades.values()), 
                                                                                         max(grades.values()) + 1)
        
    def grade(self,c): 
        return self.grades[c]
        
    def candidates_with_grade(self,g): 
        return [c for c in self.candidates if self.grades[c] == g]
        
    def ranking(self): 
            
        cmap = {c:cidx for cidx,c in enumerate(self.candidates)}
            
        grades = sorted(list(set(self.grades.values())), reverse=True)
            
        r_str = ''
        for g in grades: 
                
            cs_at_grade = self.candidates_with_grade(g)
            if len(cs_at_grade) == 1: 
                r_str += f"{cs_at_grade[0]},"
            elif len(cs_at_grade) > 1: 
                r_str  += "{" + ",".join(cs_at_grade) + "},"
        return wRanking(cmap, r_str = r_str[0:-1])
        
    def __str__(self): 
            
        return ",".join([f" {str(c)}: {str(self.grades[c])}" for c in self.candidates])

class gProfile(object):
    
    def __init__(self,  all_grades, candidates, grades, gcounts = None):
        
        self.candidates = candidates
        self.all_grades = all_grades
        self._grades = grades
        self._gcounts = gcounts if gcounts is not None else [1 for _ in grades]
        
    def grades_for_candidate(self, c):
        
        gs = [[g.grade(c)] * gnum for g,gnum in zip(self._grades, self._gcounts)]
        return [_g for _gs in gs for _g in _gs ]
        
    def average_grades(self): 
        
        return {c:np.average(self.grades_for_candidate(c)) for c in self.candidates}
    
    def average_grade_winner(self): 
        
        avg_grades = self.average_grades()
        
        max_avg_grade = max(avg_grades.values())
        return sorted([c for c in self.candidates if avg_grades[c] == max_avg_grade])

    def wr_profile(self): 
        return wProfile([g.ranking() for g in self._grades], 
                        len(self.candidates), 
                        rcounts=self._gcounts, 
                        cmap={c:cidx for cidx,c in enumerate(self.candidates)},
                        ranking_strings = False)
        
    def superior_grade_winner(self):
        
        wprof = wProfile([g.ranking() for g in self._grades], 
                         len(self.candidates), 
                         rcounts=self._gcounts, 
                         cmap={c:cidx for cidx,c in enumerate(self.candidates)},
                         ranking_strings = False)
        
        return sorted(wprof.top_cycle())
        
    def display(self,style="pretty"):
        
        tbl = list()
        
        for c in self.candidates: 
            tbl.append([str(c)] + [g.grade(c) for g in self._grades])
        print(tabulate(tbl,
                       self._gcounts, 
                       tablefmt=style))        

    
A="A"
B="B"
C="C"

voters = [Grade({A:2, B:1, C:0})] * 2

print(list(map(str,voters)))

print([str(g.ranking()) for g in voters])

print("\n")
grades = [
    Grade({A:2, B:1, C:0}),
    Grade({A:0, B:2, C:1}),
    Grade({A:1, B:0, C:2})
]
gcounts = [2, 3, 4]
all_grades = [0, 1, 2]
candidates = [A, B, C]
prof = gProfile(all_grades, candidates,grades, gcounts = gcounts)

prof.display()
print()
print(f"Grades for {A}", prof.grades_for_candidate(A))
print(f"Grades for {B}",prof.grades_for_candidate(B))
print(f"Grades for {C}",prof.grades_for_candidate(C))

print("Average grades", prof.average_grades())


print("AG Winner", prof.average_grade_winner())
print("SG Winner", prof.superior_grade_winner())

wprof = prof.wr_profile()

wprof.display()
wprof.display_margin_graph()



A="A"
B="B"
C="C"

all_grades = [0, 1, 2, 3]

candidates = [A, B, C]

grades = [
    Grade({A:3, B:0, C:0}),
    Grade({A:2, B:3, C:3}),
    Grade({A:0, B:1, C:1})
]
gcounts = [1, 1, 1]
prof = gProfile(all_grades, candidates,grades, gcounts = gcounts)

prof.display()
print()

for c in candidates:
    print(f"Grades for {c}", prof.grades_for_candidate(c))

print("")    
print("Average grades", prof.average_grades())


print("AG Winner", prof.average_grade_winner())
print("SG Winner", prof.superior_grade_winner())

wprof = prof.wr_profile()

wprof.display()
wprof.display_margin_graph()



all_grades = [0, 1, 2, 3, 4, 5]
candidates = [A, B, C]

grades = [
    Grade({A:5, B:0, C:0}),
    Grade({A:0, B:1, C:1}),
]
gcounts = [1, 4]
prof = gProfile(all_grades, candidates,grades, gcounts = gcounts)

prof.display()
print()
print(f"Grades for {A}", prof.grades_for_candidate(A))
print(f"Grades for {B}",prof.grades_for_candidate(B))
print(f"Grades for {C}",prof.grades_for_candidate(C))

print("Average grades", prof.average_grades())


print("AG Winner", prof.average_grade_winner())
print("SG Winner", prof.superior_grade_winner())

wprof = prof.wr_profile()

wprof.display()
wprof.display_margin_graph()



all_grades = [0, 1, 2]
candidates = [A, B, C]

grades = [
    Grade({A:2, B:1, C:0}),
    Grade({A:0, B:2, C:0}),
    Grade({A:0, B:0, C:2}),
]
gcounts = [1, 1, 1]
prof = gProfile(all_grades, candidates,grades, gcounts = gcounts)

prof.display()
print()
print(f"Grades for {A}", prof.grades_for_candidate(A))
print(f"Grades for {B}",prof.grades_for_candidate(B))
print(f"Grades for {C}",prof.grades_for_candidate(C))

print("Average grades", prof.average_grades())


print("AG Winner", prof.average_grade_winner())
print("SG Winner", prof.superior_grade_winner())

wprof = prof.wr_profile()

wprof.display()
wprof.display_margin_graph()



A = "A"
B = "B"
C = "C"
D = "D"

all_grades = [0, 1, 2, 3]
candidates = [A, B, C, D]

grades = [
    Grade({A:2, B:0, C:1, D:3}),
    Grade({A:1, B:1, C:1, D:2}),
    Grade({A:3, B:0, C:0, D:0}),
    Grade({A:3, B:1, C:0, D:0}),
]
gcounts = [1, 1, 1, 1]
prof = gProfile(all_grades, candidates,grades, gcounts = gcounts)

prof.display()
print()
print(f"Grades for {A}", prof.grades_for_candidate(A))
print(f"Grades for {B}",prof.grades_for_candidate(B))
print(f"Grades for {C}",prof.grades_for_candidate(C))

print("Average grades", prof.average_grades())


print("AG Winner", prof.average_grade_winner())
print("SG Winner", prof.superior_grade_winner())

wprof = prof.wr_profile()

wprof.display()
wprof.display_margin_graph()



import random

num_voters = 5
candidates = [A, B, C]
possible_grades = [0,1,2,3]

def generate_grade_profile(candidates, num_voters, grades):
    
    v_grades = list()
    for v in range(num_voters):
        
        v_grades.append(Grade({c: random.choice(grades) for c in candidates}))
 
    return gProfile(grades, candidates, v_grades)
    
                        
prof = generate_grade_profile(candidates, num_voters, possible_grades)
prof.display()

import random

num_voters = 5
candidates = [A, B, C, D]
possible_grades = [0,1,2,3]

def generate_grade_profile_strategic(candidates, num_voters, grades):
    
    v_grades = list()
    for v in range(num_voters):
        
        _grades = {c: random.choice(grades) for c in candidates}
        while not ((min(grades)  in list(_grades.values())) and (max(grades) in list(_grades.values()))):
            _grades = {c: random.choice(grades) for c in candidates}

        v_grades.append(Grade(_grades))
 
    return gProfile(grades, candidates, v_grades)
    
                        
prof = generate_grade_profile_strategic(candidates, num_voters, possible_grades)
prof.display()

%%time

num_trials = 10000

all_num_voters = [3, 4, 5, 6, 7, 8, 9, 25, 50, 100]
candidates = [A, B, C]

possible_grades = [0,1,2]

print("Sincere Voters")
for num_voters in tqdm(all_num_voters): 
    num_diff = 0
    for t in tqdm(range(num_trials), leave=False): 

        prof = generate_grade_profile(candidates, num_voters, possible_grades)
        #print()
        if prof.average_grade_winner() != split_cycle(prof.wr_profile()):
            num_diff += 1

    print(f"For {num_voters} voters: ", num_diff / num_trials)

%%time

num_trials = 10000

all_num_voters = [3, 4, 5, 6, 7, 8, 9, 25, 50, 100]
candidates = [A, B, C]

possible_grades = [0,1,2]

print("Sincere Voters")
for num_voters in tqdm(all_num_voters): 
    num_diff = 0
    for t in tqdm(range(num_trials), leave=False): 

        prof = generate_grade_profile(candidates, num_voters, possible_grades)

        if prof.average_grade_winner() != prof.superior_grade_winner():
            num_diff += 1

    print(f"For {num_voters} voters: ", num_diff / num_trials)

%%time

num_trials = 25000

all_num_voters = [3, 4, 5, 6, 7, 8, 9, 25, 50, 100]
candidates = [A, B, C]

possible_grades = [0,1,2]

print("Strategic Voters")
for num_voters in tqdm(all_num_voters): 
    num_diff = 0
    for t in tqdm(range(num_trials), leave=False): 

        prof = generate_grade_profile_strategic(candidates, num_voters, possible_grades)

        if prof.average_grade_winner() != prof.superior_grade_winner():
            num_diff += 1

    print(f"For {num_voters} voters: ", num_diff / num_trials)

%%time

num_trials = 10000

all_num_voters = [3, 4, 5, 6, 7, 8, 9, 25, 50, 100]
candidates = [A, B, C, D]

possible_grades = [0, 1, 2]

print("Sincere Voters")
for num_voters in tqdm(all_num_voters): 
    num_diff = 0
    for t in tqdm(range(num_trials), leave=False): 

        prof = generate_grade_profile(candidates, num_voters, possible_grades)

        if prof.average_grade_winner() != prof.superior_grade_winner():
            num_diff += 1

    print(f"For {num_voters} voters: ", num_diff / num_trials)

%%time

num_trials = 25000

all_num_voters = [3, 4, 5, 6, 7, 8, 9, 25, 50, 100]
candidates = [A, B, C, D]

possible_grades = [0, 1, 2]

print("Strategic Voters")
for num_voters in tqdm(all_num_voters): 
    num_diff = 0
    for t in tqdm(range(num_trials), leave=False): 

        prof = generate_grade_profile_strategic(candidates, num_voters, possible_grades)

        if prof.average_grade_winner() != prof.superior_grade_winner():
            num_diff += 1

    print(f"For {num_voters} voters: ", num_diff / num_trials)

%%time 

num_trials = 10000

all_num_voters = [3, 4, 5, 6, 7, 8, 9, 25, 50, 100]
candidates = [A, B, C]

possible_grades = [0, 1, 2, 3, 4]

print("Sincere Voters")
for num_voters in tqdm(all_num_voters): 
    num_diff = 0
    for t in tqdm(range(num_trials), leave=False): 

        prof = generate_grade_profile(candidates, num_voters, possible_grades)

        if prof.average_grade_winner() != prof.superior_grade_winner():
            num_diff += 1

    print(f"For {num_voters} voters: ", num_diff / num_trials)

%%time 

num_trials = 10000

all_num_voters = [3, 4, 5, 6, 7, 8, 9, 25, 50, 100]
candidates = [A, B, C]

possible_grades = [0, 1, 2, 3, 4]

print("Strategic Voters")
for num_voters in tqdm(all_num_voters): 
    num_diff = 0
    for t in tqdm(range(num_trials), leave=False): 

        prof = generate_grade_profile_strategic(candidates, num_voters, possible_grades)

        if prof.average_grade_winner() != prof.superior_grade_winner():
            num_diff += 1

    print(f"For {num_voters} voters: ", num_diff / num_trials)

class Grade(object):
    
    def __init__(self, grades, possible_grades = None):
        
        self.grades = grades
        self.candidates = sorted(self.grades.keys())
        self.possible_grades = possible_grades if possible_grades is not None else range(min(grades.values()), 
                                                                                         max(grades.values()) + 1)
        
    def grade(self,c): 
        return self.grades[c]
        
    def candidates_with_grade(self,g): 
        return [c for c in self.candidates if self.grades[c] == g]
        
    def ranking(self): 
            
        cmap = {c:cidx for cidx,c in enumerate(self.candidates)}
            
        grades = sorted(list(set(self.grades.values())), reverse=True)
            
        r_str = ''
        for g in grades: 
                
            cs_at_grade = self.candidates_with_grade(g)
            if len(cs_at_grade) == 1: 
                r_str += f"{cs_at_grade[0]},"
            elif len(cs_at_grade) > 1: 
                r_str  += "{" + ",".join(cs_at_grade) + "},"
        return wRanking(cmap, r_str = r_str[0:-1])
        
    def __str__(self): 
            
        return ",".join([f" {str(c)}: {str(self.grades[c])}" for c in self.candidates])

class gProfile(object):
    
    def __init__(self,  all_grades, candidates, grades, gcounts = None):
        
        self.candidates = candidates
        self.all_grades = all_grades
        self._grades = grades
        self._gcounts = gcounts if gcounts is not None else [1 for _ in grades]
        
    def grades_for_candidate(self, c):
        
        gs = [[g.grade(c)] * gnum for g,gnum in zip(self._grades, self._gcounts)]
        return [_g for _gs in gs for _g in _gs ]
        
    def average_grades(self): 
        
        return {c:np.average(self.grades_for_candidate(c)) for c in self.candidates}
    def median_grades(self): 
        
        return {c:np.median(self.grades_for_candidate(c)) for c in self.candidates}
    
    def average_grade_winner(self): 
        
        avg_grades = self.average_grades()
        
        max_avg_grade = max(avg_grades.values())
        return sorted([c for c in self.candidates if avg_grades[c] == max_avg_grade])
    def median_grade_winner(self): 
        
        median_grades = self.average_grades()
        
        max_avg_grade = max(avg_grades.values())
        return sorted([c for c in self.candidates if avg_grades[c] == max_avg_grade])

    def wr_profile(self): 
        return wProfile([g.ranking() for g in self._grades], 
                        len(self.candidates), 
                        rcounts=self._gcounts, 
                        cmap={c:cidx for cidx,c in enumerate(self.candidates)},
                        ranking_strings = False)
        
    def superior_grade_winner(self):
        
        wprof = wProfile([g.ranking() for g in self._grades], 
                         len(self.candidates), 
                         rcounts=self._gcounts, 
                         cmap={c:cidx for cidx,c in enumerate(self.candidates)},
                         ranking_strings = False)
        
        return sorted(wprof.top_cycle())
        
    def display(self,style="pretty"):
        
        tbl = list()
        
        for c in self.candidates: 
            tbl.append([str(c)] + [g.grade(c) for g in self._grades])
        print(tabulate(tbl,
                       self._gcounts, 
                       tablefmt=style))        

    
A="A"
B="B"
C="C"

voters = [Grade({A:2, B:1, C:0})] * 2

print(list(map(str,voters)))

print([str(g.ranking()) for g in voters])

print("\n")
grades = [
    Grade({A:2, B:1, C:0}),
    Grade({A:0, B:2, C:1}),
    Grade({A:1, B:0, C:2})
]
gcounts = [2, 3, 4]
all_grades = [0, 1, 2]
candidates = [A, B, C]
prof = gProfile(all_grades, candidates,grades, gcounts = gcounts)

prof.display()
print()
print(f"Grades for {A}", prof.grades_for_candidate(A))
print(f"Grades for {B}",prof.grades_for_candidate(B))
print(f"Grades for {C}",prof.grades_for_candidate(C))

print("Average grades", prof.average_grades())


print("AG Winner", prof.average_grade_winner())
print("SG Winner", prof.superior_grade_winner())

wprof = prof.wr_profile()

wprof.display()
wprof.display_margin_graph()



