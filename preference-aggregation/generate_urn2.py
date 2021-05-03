from typing import List, Any, Dict, Tuple

import math
import random
import time

# Generate votes based on the URN Model..
# we need numvotes with replace replacements.
def gen_urn(numvotes: int, replace: int, alts: List[int]) -> Dict[Tuple, int]:
    voteMap: Dict[Tuple, int] = {}
    ReplaceVotes: Dict[Tuple, int] = {}
    
    ICsize: int = math.factorial(len(alts))
#     print("ICsize ", ICsize)
    ReplaceSize: int = 0

    for x in range(numvotes):
#         print("start voteMap", voteMap)
#         print("start ReplaceVotes", ReplaceVotes)
        flip =  random.randint(1, ICsize+ReplaceSize)
        #print("flip", flip)
#         print("flip", flip)
        if flip <= ICsize:
            #generate an IC vote and make a suitable number of replacements...
            tvote = gen_ic_vote(alts)
            voteMap[tvote] = (voteMap.get(tvote, 0) + 1)
            ReplaceVotes[tvote] = (ReplaceVotes.get(tvote, 0) + replace)
            ReplaceSize += replace
#             print("ReplaceSize", ReplaceSize)
#             print("made " + str(tvote))
        else:
            #iterate over replacement hash and select proper vote.
            flip = flip - ICsize
#             print("checking replacement: flip is  ", flip)
            for vote in ReplaceVotes.keys():
#                 print("testing ", vote)
                flip = flip - ReplaceVotes[vote]
#                 print("flip is now", flip)
                if flip <= 0:
                    voteMap[vote] = (voteMap.get(vote, 0) + 1)
                    ReplaceVotes[vote] = (ReplaceVotes.get(vote, 0) + replace)
                    ReplaceSize += replace
                    break
            else:
                print("We Have a problem... replace fell through....")		
                exit()
#         print("end voteMap", voteMap)
#         print("end ReplaceVotes", ReplaceVotes)
#         print("======\n")
    return voteMap

# Return a TUPLE! IC vote given a vector of alternatives.   

def gen_ic_vote(alts: List[int]):
    options: List[int] = list(alts)
    vote: List[int] = []
    while(len(options) > 0):
        #randomly select an option
        vote.append(options.pop(random.randint(0,len(options)-1)))
    return tuple(vote)


t0 = time.time()
print(gen_urn(10, 5, [0,1,2,3]))
print(time.time() - t0)


