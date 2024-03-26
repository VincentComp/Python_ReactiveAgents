import pandas as pd
import numpy as np

df1 = pd.read_csv("north.csv",header=None)
df2 = pd.read_csv("east.csv",header=None)
df3 = pd.read_csv("south.csv",header=None)
df4 = pd.read_csv("west.csv",header=None)
training_dataset1 = df1.values
training_dataset2 = df2.values
training_dataset3 = df3.values
training_dataset4 = df4.values
init_weight = 0.0
lr = 1.0
epoch = 100

class Perceptron():
    def __init__(self, training_dataset,init_weight, lr,epoch):
        self.training_dataset = training_dataset
        self.weight = np.full(shape=(8 + 1,), fill_value = init_weight) #Also include threshold( = -w9)
        self.lr = lr
        self.epoch = epoch

    def printDetails(self): #Helper function for debug
        print(f'This perceptron with weight = {self.weight} \n lr= {lr} \n training_dataset = {self.training_dataset}' )


    def forward_propagation(self, debug = False):
        
        
        for i in range(self.epoch):
            num_error = 0

            for training_instance in self.training_dataset:
                this_input = np.hstack((training_instance[:-1],np.ones(1))) #Append threshold as a column of 1
                this_actual_label = training_instance[-1]
                
                this_predicted_label = ((this_input @ self.weight)  >= 0)
            
                if((this_actual_label != this_predicted_label)):
                    self.weight = self.weight + self.lr * (this_actual_label - this_predicted_label)* this_input
                    num_error+=1;
                
            if(debug):
                print(f'Iteration {i}, Number_error = {num_error}')

            if(num_error == 0):
                break

    def predict(self,test_dataset): #help function for debug
        return (test_dataset @ self.weight[:-1]) >= self.weight[-1] * -1 #Biased is -ve of weight
            
            

            


#Main function
perceptron1 = Perceptron(training_dataset1,init_weight,lr,epoch)
perceptron1.forward_propagation()
print(f'Weight and biasd for North Action = {perceptron1.weight}')
#print(perceptron1.predict(training_dataset1[:,:-1]))

perceptron2 = Perceptron(training_dataset2,init_weight,lr,epoch)
perceptron2.forward_propagation()
print(f'Weight and biasd for East Action = {perceptron2.weight}')
#print(perceptron2.predict(training_dataset2[:,:-1]))

perceptron3 = Perceptron(training_dataset3,init_weight,lr,epoch)
perceptron3.forward_propagation()
print(f'Weight and biasd for South Action = {perceptron3.weight}')
#print(perceptron3.predict(training_dataset3[:,:-1]))

perceptron4 = Perceptron(training_dataset4,init_weight,lr,epoch)
perceptron4.forward_propagation()
print(f'Weight and biasd for West Action = {perceptron4.weight}')
#print(perceptron4.predict(training_dataset4[:,:-1]))





"""
Self Note:
Training dataset
=> Shape: (7,9)
=> 8 sensors + 1 labels
"""