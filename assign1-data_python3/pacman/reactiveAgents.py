# reactiveAgents.py
# ---------------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC
# Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search


import numpy as np

class NaiveAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        sense = state.getPacmanSensor()
        if sense[7]:
            action = Directions.STOP
        else:
            action = Directions.WEST
        return action

class PSAgent(Agent):
    "An agent that follows the boundary using production system."

    def getAction(self, state):
        sense = state.getPacmanSensor()
        x = [sense[1] or sense[2] , sense[3] or sense[4] ,
        sense[5] or sense[6] , sense[7] or sense[0]]
        if x[0] and not x[1]:
            action = Directions.EAST
        elif x[1] and not x[2]:
            action = Directions.SOUTH
        elif x[2] and not x[3]:
            action = Directions.WEST
        elif x[3] and not x[0]:
            action = Directions.NORTH
        else:
            action = Directions.NORTH
        return action

class ECAgent(Agent):
    "An agent that follows the boundary using error-correction."

    def getAction(self, state):
        ''' @TODO: Your code goes here! '''
        sense = state.getPacmanSensor()
        
        inputs = np.array([sense[0], sense[1], sense[2], sense[3], sense[4],sense[5],sense[6],sense[7]])
        weights = np.array([[ 1, -2, -2,  0,  0,  0,  0,  1, -1], #Please refer to the Question4Perceptron.py
        [ 0,  1,  1, -2, -2,  0,  0,  0, -1],
        [ 0,  0,  0,  1,  1, -2, -2,  0, -1],
        [-2,  0,  0,  0,  0,  1,  1, -2, -1]])


        result = weights[:,:-1]@inputs >= (weights[:,-1]*-1)

        if(result[0]):
            return Directions.NORTH
        
        if(result[1]):
            return Directions.EAST
        
        if(result[2]):
            return Directions.SOUTH
        
        if(result[3]):
            return Directions.WEST

        return Directions.NORTH
    
    

class SMAgent(Agent):
    "An sensory-impaired agent that follows the boundary using state machine."
    def registerInitialState(self,state):
        "The agent receives the initial GameState (defined in pacman.py)."
        sense = state.getPacmanImpairedSensor() 
        self.prevAction = Directions.STOP
        self.prevSense = sense

    def getAction(self, state):
        '''@TODO: Your code goes here! ''' 

        #Get the current sense and the previous sense
        [w2,w4,w6,w8] = state.getPacmanImpairedSensor() 
        [w1,w3,w5,w7] = self.prevSense

        #Use the production system same as the lecture notes
        if(w2 and not w4):
            return_direction = Directions.EAST

        elif(w4 and not w6):
            return_direction = Directions.SOUTH

        elif(w6 and not w8):
            return_direction = Directions.WEST

        elif(w8 and not w2):
            return_direction = Directions.NORTH

        elif(w1):
            return_direction = Directions.NORTH

        elif(w3):
            return_direction = Directions.EAST

        elif(w5):
            return_direction = Directions.SOUTH

        elif(w7):
            return_direction = Directions.WEST
        else:
            return_direction = Directions.NORTH


        #udapte the previous sense
        (d1,d2,d3,d4) = (0,0,0,0)
        if((return_direction == Directions.EAST) and (w2 == 1)):
            d1 = 1
        if((return_direction == Directions.SOUTH) and (w4 == 1)):
            d2 = 1
        if((return_direction == Directions.WEST) and (w6 == 1)):
            d3 = 1
        if((return_direction == Directions.NORTH) and (w8 == 1)):
            d4 = 1
        

        #save previous action and sense
        self.prevSense = [w2*d1,w4*d2,w6*d3,w8*d4]
        self.prevAction = return_direction

        return return_direction


