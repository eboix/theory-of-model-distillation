import torch
import random
import copy

import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import graphviz

device = torch.device("cuda")

def comp_and(x,literals):
    with torch.no_grad():
        S,sgns = literals
        if len(S) == 0:
            return torch.ones(x.shape[0]).to(device)
        # x is a B x d pytorch array of +1,-1 -valued variables
        # S is a list of variables in range(d) of length |S|
        # sgns is an array of signs in +1,-1 of dimensions 1 x |S|
        xsub = x[:,S]
        sgn_correct = torch.sum(xsub * sgns > 0,dim=1) # B x |S|
        return 2 * (sgn_correct > len(S) - 0.5) - 1
    
def comp_and_list(x, literals_list):
    n = len(literals_list)
    B = x.shape[0]
    out_tensor = torch.zeros((B,n)).to(device)
    for i, literals in enumerate(literals_list):
        out_tensor[:,i] = comp_and(x, literals)
    return out_tensor

def literals_to_tup(d,literals):
    literal_vec = [0]*d
    S,sgns = literals
    for i in range(len(S)):
        literal_vec[S[i]] = sgns[0,i].item()
    return tuple(literal_vec)

def tup_to_literals(tup):
    S = []
    sgns = []
    for i in range(len(tup)):
        if abs(tup[i]) < 1e-5:
            continue
        elif abs(tup[i] - 1) < 1e-5:
            S.append(i)
            sgns.append(1)
        elif abs(tup[i] + 1) < 1e-5:
            S.append(i)
            sgns.append(-1)
        else:
            assert(False)
    return (S,torch.Tensor(sgns).view(1,-1).to(device))

class AndOfLiterals:
    def __init__(self,tup):
        self.tup = tup
        self.literals = None
    def numvars(self):
        S,sgns = self.get_literals()
        return len(S)
    def get_tup(self):
        return self.tup
    def get_literals(self):
        if self.literals is None:
            self.literals = tup_to_literals(self.tup)
        return self.literals
    def get_possible_children_vars(self):
        d = len(self.tup)
        return [i for i in range(d) if self.tup[i] == 0]

    def compute(self,x):
        # print(x.shape)
        # print(x)
        if len(x.shape) == 1:
            x = x.view(1,-1)
        S,sgns = self.get_literals()
        if len(S) == 0:
            return torch.ones(x.shape[0]).to(device)
        # x is a B x d pytorch array of +1,-1 -valued variables
        # S is a list of variables in range(d) of length |S|
        # sgns is an array of signs in +1,-1 of dimensions 1 x |S|
        xsub = x[:,S]
        sgn_correct = torch.sum(xsub * sgns > 0,dim=1) # B x |S|
        return 2 * (sgn_correct > len(S) - 0.5) - 1
    
    def __str__(self):
        literals = self.get_literals()
        return str(literals)
    def __repr__(self):
        literals = self.get_literals()
        return 'AND(S=' + str(literals[0]) + ', sgns=' + str(literals[1]) + ')'


class DecisionTree:
    def __init__(self,d=None,r=None,root_tup=None,child_dict=None,sgn_dict=None):
        
        if root_tup is None:
            # Random tree of depth r on d variables
            self.children = {}
            self.parent = {}
            self.ands = []
            self.node_label = {}
            self.leaf_weights = []
            self.leaf_indices = []

            self.ands.append(AndOfLiterals(tuple([0]*d))) # Add the root
            i = 0
            while i < len(self.ands):
                currand = self.ands[i]
                if currand.numvars() == r:
                    self.leaf_indices.append(i)
                    self.leaf_weights.append(random.randint(0,1)*2-1)
                    self.node_label[i] = str(self.leaf_weights[-1])
                else:
                    pos_children = currand.get_possible_children_vars()
                    randchild = random.choice(pos_children)
                    self.node_label[i] = f'x_{randchild}'
                    self.children[i] = (len(self.ands),len(self.ands)+1)
                    self.parent[len(self.ands)] = i
                    self.parent[len(self.ands)+1] = i

                    for j in [1,-1]:
                        tup1 = copy.deepcopy(list(currand.tup))
                        tup1[randchild] = j
                        tup1 = tuple(tup1)
                        self.ands.append(AndOfLiterals(tup1))
                i += 1
        else:
            # Random tree of depth r on d variables
            self.children = {}
            self.parent = {}
            self.ands = []
            self.node_label = {}
            self.leaf_weights = []
            self.leaf_indices = []
            
            self.ands.append(AndOfLiterals(root_tup))
            d = len(root_tup)
            i = 0
            while i < len(self.ands):
                currand = self.ands[i]
                currtup = currand.get_tup()
                if child_dict[currtup] is None:
                    self.leaf_indices.append(i)
                    self.leaf_weights.append(sgn_dict[currtup])
                    self.node_label[i] = str(self.leaf_weights[-1])
                else:
                    tup1,tup2 = child_dict[currtup]
                    jlist = [j for j in range(d) if tup1[j] != currtup[j]]
                    assert(len(jlist) == 1)
                    j = jlist[0]
                    self.node_label[i] = f'x_{j}'
                    self.children[i] = (len(self.ands),len(self.ands)+1)
                    self.ands.append(AndOfLiterals(tup1))
                    self.ands.append(AndOfLiterals(tup2))
                i += 1
                    
        
    def compute(self,x):
        # Takes in B x d vector with +1,-1 values
        # Outputs tree on all branches, which is sum of ANDs at all leaves, appropriately weighted
        currout = torch.zeros(x.shape[0]).to(x.device)
        for i in range(len(self.leaf_indices)):
            currand = self.ands[self.leaf_indices[i]]
            currwt = self.leaf_weights[i]
            currbranch = (currand.compute(x)+1)/2
            # print(currand)
            # print(currbranch)
            currout += currbranch * currwt
            # print(currwt)
        return currout
    
    def get_graph(self,zero_ones=False):
        G = nx.DiGraph()
        for i in range(len(self.ands)):
            currlabel = self.node_label[i]
            if zero_ones:
                if str(currlabel) == '1':
                    currlabel = '0'
                elif str(currlabel) == '-1':
                    currlabel = '1'
            G.add_node(i, label=currlabel)
        for i in range(len(self.ands)):
            if i in self.children:
                if not zero_ones:
                    G.add_edge(i, self.children[i][0],label=1)
                    G.add_edge(i, self.children[i][1],label=-1)
                else:
                    G.add_edge(i, self.children[i][0],label=0)
                    G.add_edge(i, self.children[i][1],label=1)
        return G
        
        
    def visualize(self, zero_ones=False):
        G = self.get_graph(zero_ones=zero_ones)
        write_dot(G, "temporary_graph.dot")
        dot = graphviz.Source.from_file("temporary_graph.dot")
        display(dot)
        return dot