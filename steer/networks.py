import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=2): 
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(64, activation='tanh') 
        self.fc3 = Dense(64, activation='tanh') 
        self.fc5 = Dense(n_actions, activation='softmax')
    
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc5(x)

        return x
    
class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=2):
        super(CriticNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(64, activation='relu')
        self.fc3 = Dense(64, activation='relu')
        self.fc5 = Dense(1, activation='linear')
    
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc5(x)

        return x
