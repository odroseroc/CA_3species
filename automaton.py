#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:24:47 2025

@author: oscar
"""

import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
from importlib import reload
plt=reload(plt)
from numba import jit
from statistics import mean

"""
Defines a cellular automaton with 3 types of actors, a top-level predator,
a low-level predator and a herbivore.

The model assumes that the herbivore's food suply is absolutely abundant,
that the herbivore population is insensitive to ovepopulation and can
only die by predation.

Predator death rate is a random process, independent of hunger or age, and 
every member has equal probability of dying each time step.

The herbivore species is the only source of energy for low-level preadators, 
and the latter are the only source of energy for top-level predators.

Sexual reprodution is not assumed. A new herbivore can be born with 
probability herb_birthRate if there is at least one herbivore in the vicinity.
A new predator (low- or top-level) can be born after a successful hunt with 
probability predLow_birthRate and predTop_birthRate.
    
Emtpy cells are filled with 0. Herbivores are identified with 1, low-level
predators with 2, and top-level predators with 3.
"""

# Color map definition for CA plotting
cmap = colors.ListedColormap(['black', 'green', 'orange', 'red'])
bounds = [0, 1, 2, 3]
norm = colors.BoundaryNorm(bounds, cmap.N)
labels = ['empty', 'herbivore', 'low_predator', 'top_predator']  # Custom labels

# Identifiers for herbivore, low-level predator and top-level predator
HBV = 1
LLP = 2
TLP = 3
LLP_pred = 4

class CellAutomaton:
    
    def __init__(self, size,
                 herb_birthRate, predLow_birthRate, predTop_birthRate,
                 herb_deathRate, predLow_deathRate, predTop_deathRate,
                 llp_rndDeathRate,
                 herb_spawnProb = 0.1,
                 predLow_spawnProb = 0.1,
                 predTop_spawnProb = 0.1,
                 neighbourhoodRadius = 1
                 ):
        self.size = size
        empty_spawnProb = 1 - predTop_spawnProb - predLow_spawnProb - herb_spawnProb
        self.state = np.random.choice([0,1,2,3], size*size, 
                                     p=[empty_spawnProb, herb_spawnProb, 
                                        predLow_spawnProb, predTop_spawnProb]).reshape(size,size)
        self.birthRates = [0, herb_birthRate, predLow_birthRate, predTop_birthRate]
        self.deathRates = [0, herb_deathRate, predLow_deathRate, predTop_deathRate, llp_rndDeathRate]
        self.neighbourhoodRadius = neighbourhoodRadius
        self.history = []
        self.populationHistory = []
        self.hbv_popHist = []
        self.llp_popHist = []
        self.tlp_popHist = []
        self.all_popHist = [0, self.hbv_popHist, self.llp_popHist, self.tlp_popHist]
        
    def count_neighbours(self, pos):
        # Counts neighbours of each species with toroidal boundaries
        r = self.neighbourhoodRadius
        neighbours = [0,0,0,0] # Contains number of emtpy, herbivores, low- and top-level predators, respectively
        size = self.size
        state = self.state
        for i in range(-r,r+1):
            for j in range(-r,r+1):
                current_neighbour = state[(pos[0]+i)%size, (pos[1]+j)%size]
                if current_neighbour == HBV:
                    neighbours[HBV] += 1
                elif current_neighbour == LLP:
                    neighbours[LLP] += 1
                elif current_neighbour == TLP:
                    neighbours[TLP] += 1
                else:
                    neighbours[0] += 1
        return neighbours
    
    def rules_prey(self, cell_id, predator_id, neighbours):
        n_predators = neighbours[predator_id]
        r = rng.rand()
        if r < (1 - self.deathRates[cell_id])**(n_predators):
            return cell_id # No predators or hunt failed -> stays the same
        else:
            r = rng.rand()
            if r < self.birthRates[predator_id]:
                return predator_id
            else:
                return 0
            
    def rules_predator(self, cell_id):
        r = rng.rand()
        if r < self.deathRates[cell_id]:
            return 0
        else:
            if cell_id == LLP_pred:
                return LLP
            else:
                return cell_id
        
    def rules_empty(self, neighbours):
        if neighbours[HBV] == 0 or neighbours[LLP] > 0:
            return 0
        else:
            r = rng.rand()
            if r < (1 - self.birthRates[HBV])**neighbours[HBV]:
                return HBV
            else:
                return 0
            
    def evaluate_cell(self, pos):
        cell_id = self.state[pos[0], pos[1]]
        neighbours = self.count_neighbours(pos)
        if cell_id == HBV:
            new_id = self.rules_prey(HBV, LLP, neighbours)
        elif cell_id == LLP:
            new_id = self.rules_prey(LLP, TLP, neighbours)
            if new_id == LLP: # Hunt by TLP failed and cell is still alive
                new_id = self.rules_predator(LLP_pred)
        elif cell_id == TLP:
            new_id = self.rules_predator(TLP)
        else:
            new_id = self.rules_empty(neighbours)
        return new_id
    
    def get_speciesPopHistory(self, species, stabilizationSteps=0, countEmpties=0):
        sp_popHist = [array[species]/((countEmpties*array[0])+array[HBV]+array[LLP]+array[TLP]) for array in self.populationHistory[stabilizationSteps:]]
        return sp_popHist
    
    def evolveStep(self):
        new_state = np.zeros([self.size, self.size] ,dtype=int)
        population = np.zeros(4, dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                pos = (i, j)
                new_state[i, j] = self.evaluate_cell(pos)
                population[new_state[i,j]] += 1
        return new_state, population
        
    def evolve(self, n_steps=50):
        self.history.append(self.state.copy())
        population0 = self.count_total(step=0)
        self.populationHistory.append(population0)
        totalPop = population0[0]+population0[HBV]+population0[LLP]+population0[TLP]
        self.hbv_popHist.append(population0[HBV]/totalPop)
        self.llp_popHist.append(population0[LLP]/totalPop)
        self.tlp_popHist.append(population0[TLP]/totalPop)
        for step in range(n_steps):
            state, population = self.evolveStep()
            self.state = state
            self.populationHistory.append(population)
            totalPop = population[0]+population[HBV]+population[LLP]+population[TLP]
            self.hbv_popHist.append(population[HBV]/totalPop)
            self.llp_popHist.append(population[LLP]/totalPop)
            self.tlp_popHist.append(population[TLP]/totalPop)
            self.history.append(state)
            
    def snapshot(self, step):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(self.history[step], cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
        popHBV = str(self.populationHistory[step][HBV])
        popLLP = str(self.populationHistory[step][LLP])
        popTLP = str(self.populationHistory[step][TLP])
        title = "Step: " + str(step) + " | Pop: HBV = " + popHBV + ", LLP = " + popLLP + ", TLP = " + popTLP
        ax.set_title(title)
        # cbar = fig.colorbar(im, ticks=[0.375, 1.125, 1.875, 2.625])  # Set tick positions
        # cbar.ax.set_yticklabels(labels)  # Set custom tick labels
        plt.show()
        
    def show_history(self, interval=1, initial=0):
        # labels = ['empty', 'herbivore', 'low_predator', 'top_predator']  # Custom labels
        if initial > len(self.history):
            raise ValueError("Initial step exceeds length of the history of the automaton")
        step = initial
        while step < len(self.history):
            if step % interval == 0:
                self.snapshot(step)
            step += 1
        return None
    
    def count_total(self, step):
        state = self.history[step]
        counts = np.zeros(4, dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                counts[state[i,j]] += 1
        return counts
    
    def plot_popProps(self, plot_HBV = True, plot_LLP = True, plot_TLP = True, stabilizationSteps = 0, writeTitle = True):
        # countEmpties must be 1 for the population proportion to take empty spaces into account, and 0 for it to not do that
        timeSteps = range(stabilizationSteps, len(self.populationHistory))
        plt.figure()
        if plot_HBV:
            plt.plot(timeSteps, self.hbv_popHist[stabilizationSteps:], label="Herbivores", color="green", linewidth=1)
        if plot_LLP:
            plt.plot(timeSteps, self.llp_popHist[stabilizationSteps:], label="Low-level predators", color="orange", linewidth=1)
        if plot_TLP:
            plt.plot(timeSteps, self.tlp_popHist[stabilizationSteps:], label="Top-level predators", color="red", linewidth=1)
        if writeTitle:
            title = "birthRates: " + str(self.birthRates[1:]) + ", deathRates: " + str(self.deathRates[1:])
        else:
            title = ""
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Proportion of total population")
        plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
        plt.show()
        
    def plot_phaseDiag(self, species1, species2, stabilizationSteps=0, plot_means=False):
        species_name = ["Empty", "Herbivore", "Low-level predator", "Top-level predator"]
        if plot_means:
            mean1 = mean(self.all_popHist[species1])
            mean2 = mean(self.all_popHist[species2])
        sp1_pop = self.all_popHist[species1]
        sp2_pop = self.all_popHist[species2]
        plt.scatter(sp1_pop, sp2_pop, s=5)
        plt.scatter(mean1, mean2, marker="D", color="red")
        plt.xlabel(species_name[species1])
        plt.ylabel(species_name[species2])
        plt.show()
        
    
        
class TinyCA(CellAutomaton):
    
    def __init__(self,
                 herb_birthRate, predLow_birthRate, predTop_birthRate,
                 herb_deathRate, predLow_deathRate, predTop_deathRate,
                 llp_rndDeathRate,
                 state0 = np.zeros([3,3],dtype=int),
                 size=3,
                 neighbourhoodRadius = 1
                 ):
        self.size = size
        self.state = state0
        self.birthRates = [0, herb_birthRate, predLow_birthRate, predTop_birthRate]
        self.deathRates = [0, herb_deathRate, predLow_deathRate, predTop_deathRate, llp_rndDeathRate]
        self.neighbourhoodRadius = neighbourhoodRadius
        self.history = []
        self.populationHistory = []
        self.hbv_popHist = []
        self.llp_popHist = []
        self.tlp_popHist = []
        self.all_popHist = [0, self.hbv_popHist, self.llp_popHist, self.tlp_popHist]
        self.history.append(self.state)
        
    def snapshot(self, step):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(self.history[step], cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
        # popHBV = str(self.populationHistory[step][HBV])
        # popLLP = str(self.populationHistory[step][LLP])
        # popTLP = str(self.populationHistory[step][TLP])
        # title = "Step: " + str(step) + " | Pop: HBV = " + popHBV + ", LLP = " + popLLP + ", TLP = " + popTLP
        # ax.set_title(title)
        cbar = fig.colorbar(im, ticks=[0.375, 1.125, 1.875, 2.625])  # Set tick positions
        cbar.ax.set_yticklabels(labels)  # Set custom tick labels
        plt.show()
    
###############################################################################

def test1():
    hbv_spawn = 0.1 # Probability of spawning a herbivore predator in the initial grid
    llp_spawn = 0.1 # Probability of spawning a low-level predator in the initial grid
    tlp_spawn = 0.1 # Probability of spawning a top-level predator in the initial grid
    
    base = [0.3,0.8,0.7,0.8,0.4,0.1,0.1]
    hbv_b, llp_b, tlp_b, hbv_d, llp_d, tlp_d, llp_rd = base
    
    size = 50
    
    # hbv_b = 0.3
    # llp_b = 0.7
    # tlp_b = 0.65

    # hbv_d = 0.8
    # llp_d = 0.7
    # tlp_d = 0.1
    # llp_rd = 0.1
    
    n_simSteps = 350
    stbl_steps = 50
    
    ca = CellAutomaton(size,hbv_b, llp_b, tlp_b, hbv_d, llp_d, tlp_d, llp_rd, hbv_spawn, llp_spawn, tlp_spawn)
    ca.evolve(n_steps=n_simSteps)
    ca.show_history(interval=50)
    ca.plot_popProps(stabilizationSteps = stbl_steps, plot_HBV=True, plot_LLP=True, plot_TLP=True,writeTitle=True)
    ca.plot_phaseDiag(HBV,LLP,stabilizationSteps=stbl_steps,plot_means=True)
    ca.plot_phaseDiag(LLP,TLP,stabilizationSteps=stbl_steps,plot_means=True)
    ca.plot_phaseDiag(HBV,TLP,stabilizationSteps=stbl_steps,plot_means=True)
    return 0

def survivalProb(species):
    hbv_b = 0.2
    llp_b = 0.5
    tlp_b = 0.5

    hbv_d = 0.9
    llp_d = 0.25
    tlp_d = 0.1
    llp_rd = 0.1
    n_sims = 10000
    
    if species == 0:
        plt.figure()
        for hbv_b in [0.2, 0.35, 0.5, 0.65, 0.8]:
            probabilities = [0 for i in range(8)]
            state0 = np.array([[0,0,0,],
                               [0,0,0],
                               [0,0,0]])
            n_hbv = 0
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    state0[i,j] = HBV
                    n_hbv += 1
                    for sim in range(n_sims):
                        ca = TinyCA(hbv_b, llp_b, tlp_b, hbv_d, llp_d, tlp_d, llp_rd, state0)
                        cell_state = ca.evaluate_cell((1,1))
                        probabilities[n_hbv-1] += cell_state/(HBV*n_sims)
                        # print(probabilities)
            plt.plot(range(1,9), probabilities, label=r'$b_h = $'+str(hbv_b))
        plt.xlabel("Number of herbivores")
        plt.ylabel("Probability of birth")
        plt.legend()
        plt.show()
        
    if species == HBV:
        plt.figure()
        for hbv_d in [0.1, 0.25, 0.5, 0.75, 0.9]:
            probabilities = [0 for i in range(9)]
            state0 = np.array([[0,0,0,],
                               [0,HBV,0],
                               [0,0,0]])
            n_llp = 0
            probabilities[0] = 1
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    state0[i,j] = LLP
                    n_llp += 1
                    for sim in range(n_sims):
                        ca = TinyCA(hbv_b, llp_b, tlp_b, hbv_d, llp_d, tlp_d, llp_rd, state0)
                        cell_state = ca.evaluate_cell((1,1))
                        if cell_state != HBV:
                            continue
                        else:
                            probabilities[n_llp] += 1/n_sims
            plt.plot(range(9), probabilities, label=r'$d_h = $'+str(hbv_d))
        plt.xlabel("Number of low-level predators")
        plt.ylabel("Probability of survival")
        plt.legend()
        plt.show()
        
    if species == LLP:
        plt.figure()
        for llp_rd in [0.1, 0.25, 0.5, 0.75, 0.9]:
            for llp_d in [0.1, 0.25, 0.5, 0.75, 0.9]:
                probabilities = [0 for z in range(9)]
                state0 = np.array([[0,0,0,],
                                   [0,LLP,0],
                                   [0,0,0]])
                n_tlp = 0
                for sim in range(n_sims):
                    ca = TinyCA(hbv_b, llp_b, tlp_b, hbv_d, llp_d, tlp_d, llp_rd, state0)
                    cell_state = ca.evaluate_cell((1,1))
                    if cell_state != LLP:
                        continue
                    else:
                        probabilities[n_tlp] += 1/n_sims
                for i in range(3):
                    for j in range(3):
                        if i == 1 and j == 1:
                            continue
                        state0[i,j] = TLP
                        n_tlp += 1
                        for sim in range(n_sims):
                            ca = TinyCA(hbv_b, llp_b, tlp_b, hbv_d, llp_d, tlp_d, llp_rd, state0)
                            cell_state = ca.evaluate_cell((1,1))
                            if cell_state != LLP:
                                continue
                            else:
                                probabilities[n_tlp] += 1/n_sims
                plt.plot(range(9), probabilities, label=r'$d_m = $'+str(llp_d)+"\t"+r'$d_m^\prime = $'+str(llp_rd))
            plt.xlabel("Number of top-level predators")
            plt.ylabel("Probability of survival")
            plt.legend()
            plt.show()

          
def test_survival():
    hbv_spawn = 0.1 # Probability of spawning a herbivore predator in the initial grid
    llp_spawn = 0.1 # Probability of spawning a low-level predator in the initial grid
    tlp_spawn = 0.1 # Probability of spawning a top-level predator in the initial grid
    
    size = 50
    sim_steps = 350
    n_sims = 5
    
    
    base = [0.3,0.8,0.7,0.8,0.4,0.1,0.1]
    params = [r'$b_h$', r'$b_m$', r'$b_p$', r'$d_h$', r'$d_m$', r'$d_p$', r'$d_m^\prime$']
    hbv_b, llp_b, tlp_b, hbv_d, llp_d, tlp_d, llp_rd = base
    values = [0.1, 0.25, 0.35, 0.5, 0.75, 0.85, 0.9]
    avg_n_tlp = [[0 for a in range(len(values))] for b in range(7)] # Contains one list of len(values) elements for each parameter of the CA, containing averages of each simulation
    
    for v_id,value in enumerate(values):
        print("----------Evaluating value "+str(v_id+1)+" / "+str(len(values))+": "+str(value)+"-----------------------------------")
        for sim in range(n_sims):
            print("..........Simulation "+str(sim+1)+" / "+str(n_sims)+"..........")
            cas = [CellAutomaton(size, value, llp_b, tlp_b, hbv_d, llp_d, tlp_d, llp_rd, hbv_spawn, llp_spawn, tlp_spawn),
                    CellAutomaton(size, hbv_b, value, tlp_b, hbv_d, llp_d, tlp_d, llp_rd, hbv_spawn, llp_spawn, tlp_spawn),
                    CellAutomaton(size, hbv_b, llp_b, value, hbv_d, llp_d, tlp_d, llp_rd, hbv_spawn, llp_spawn, tlp_spawn),
                    CellAutomaton(size, hbv_b, llp_b, tlp_b, value, llp_d, tlp_d, llp_rd, hbv_spawn, llp_spawn, tlp_spawn),
                    CellAutomaton(size, hbv_b, llp_b, tlp_b, hbv_d, value, tlp_d, llp_rd, hbv_spawn, llp_spawn, tlp_spawn),
                    CellAutomaton(size, hbv_b, llp_b, tlp_b, hbv_d, llp_d, value, llp_rd, hbv_spawn, llp_spawn, tlp_spawn),
                    CellAutomaton(size, hbv_b, llp_b, tlp_b, hbv_d, llp_d, tlp_d, value, hbv_spawn, llp_spawn, tlp_spawn)]
            for idx, ca in enumerate(cas):
                print("*****Varying parameter "+str(params[idx])+"*****")
                ca.evolve(sim_steps)
                print("CA evolved "+str(sim_steps)+" steps")
                contribution = ca.hbv_popHist[-1]/n_sims # Change hbv_ llp_ or tlp_popHist to analyze single parameter variation of different species
                avg_n_tlp[idx][v_id] += contribution
                print("Contribution to average population at the end: "+str(contribution))
            print("\n Averages so far: "+str(avg_n_tlp)+"\n")
    print("\n Simulation complete. Averages: "+str(avg_n_tlp))           
    plt.figure()
    for idx, param in enumerate(params):
        plt.plot(values, avg_n_tlp[idx], label=param, marker=".")
    plt.xlabel("Parameter values")
    plt.ylabel("Proportion of total population")
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.show()
    
                
                
        
        
        
        

if __name__ == "__main__" :
    # test1()
    # survivalProb(LLP)
    test_survival()
    
    
