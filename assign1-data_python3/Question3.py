import pandas as pd
import numpy as np
import random

#These are hyperparameter
initialPopulation = 1000
initalWeightRange = 1
tournement_size = 10

copy_rate = 0.1
crossover_rate = 0.8
mutation_rate = 0.1
mutation_strength = 0.1

max_iteration = 100
accuracy_threshold = 95



class GPSystem():
    def __init__(self,initialPopulation,initalWeightRange,tournement_size,copy_rate,crossover_rate,mutation_rate,mutation_strength,max_iteration,accuracy_threshold):
        #These are hyperparameter.again
        self.population = initialPopulation     
        self.tournement_size = tournement_size 
        
        
        self.copy_rate = copy_rate              
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

        self.max_iteration = max_iteration
        self.accuracy_threshold = accuracy_threshold

        #Dataset, list of agent, list of fitness
        self.dataset = (pd.read_csv('gp-training-set.csv', header=None)).values #Load the training dataset to the model
        self.agent_list = [Agent(np.random.uniform(-initalWeightRange,initalWeightRange,size = (9 + 1,))) for i in range(initialPopulation)] #initialize a list of agents
        self.fitness_list = np.zeros(initialPopulation); self.all_evaluate_fitness()
        
        self.iteration = 0 #current iteration

    def all_evaluate_fitness(self): #Evaluate the fitness for all agent
        for i in range(self.population):
            self.fitness_list[i] = self.agent_list[i].evaluate_fitness(self.dataset)

    def tournement_selection(self): #Tournement Selection(Select the best only)
        candidate_index = random.sample(range(self.population),self.tournement_size)
        candidate = []
        candidate_fitness = [] 

        for i in candidate_index:
            this_agenet = self.agent_list[i]
            candidate.append(this_agenet)
            candidate_fitness.append(this_agenet.fitness)
        
        return candidate[np.argmax(candidate_fitness)]
    

    def copy(self): #Copy to reproduce child
        return [self.tournement_selection() for i in range(int(self.population*copy_rate))]

    def crossover(self): #Crossover: mix the first half of father gene with mother's later half of gene
        crossover_list = []

        for i in range(int(self.population*crossover_rate)):
            father = self.tournement_selection()
            mother = self.tournement_selection()

            crossover_list.append(Agent(np.concatenate((father.gene[0:5],mother.gene[5:])))) #concate father and mother gene to produce child

        return crossover_list
        
    def mutation(self): #Mutation: Add offest to the gene of some agents 
        candidate_index = random.sample(range(self.population),int(self.mutation_rate * self.population))        
        return [Agent(self.agent_list[i].gene + np.random.uniform(-self.mutation_strength,self.mutation_strength,size = (9 + 1,))) for i in candidate_index]

            
    def evolve(self): #Start Evolving the system
        for i in range(self.max_iteration):
            self.iteration+=1 #update current_iteration

            self.agent_list = self.copy() + self.crossover() + self.mutation() #generate child from copy,crossover, mutation
            self.all_evaluate_fitness() #evaluate their fitness


            if(np.max(self.fitness_list) >= self.accuracy_threshold): #return when the current generation meet the accuracy threshold
                return self.agent_list[np.argmax(self.fitness_list)] 
        
        return self.agent_list[np.argmax(self.fitness_list)] #Return the list of agent 

    def printAgents(self): #Helper function for debugging
        for i in range(self.population):
            this_agent = self.agent_list[i]
            print(this_agent.gene, f'Fitness : {this_agent.fitness}')
        

class Agent():
    def __init__(self,gene): 
        self.gene = gene
        self.fitness = 0 #(w1,w9,....,bias)
        

    def evaluate_fitness(self,dataset): #Test the agent with training dataset to count how many labels are matched
        self.fitness = np.sum((dataset[:,:-1] @ self.gene[:-1] >= self.gene[-1]) == dataset[:,-1])
        return self.fitness
        

    

#Main
gps = GPSystem(initialPopulation,initalWeightRange,tournement_size,copy_rate,crossover_rate,mutation_rate,mutation_strength,max_iteration,accuracy_threshold) #initialize the GP-system
best_agent = gps.evolve()
print(f'Optimal weight and threshold = {best_agent.gene}, with accuracy = {best_agent.fitness} at {gps.iteration}th generation')



"""
Self: Note
Training Dataset [Shape = (100,10)]
=> 100 instances
=> x1-x9 = input (real number)
=> l     = output (0 or 1)

Target: Train the program (=agent) 
=> (w1,...,w9,theta) -->  we call it gene
"""


