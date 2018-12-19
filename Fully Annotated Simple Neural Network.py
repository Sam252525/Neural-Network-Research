'''This is Sam Stazinski's annotation of https://bit.ly/2s2S7i4'''

#  Imports numpy, numpy adds a lot of math functions and tools
import numpy as np


#  This sigmoid function can take in a number and then spit out a
#  value on a curve between 0 and 1
def sigmoid(a):
    return 1.0/(1 + np.exp(-a))

#  The derivative of a sigmoid comes out to a nice f(x)(1-f(x))
#  The derivative of a sigmoid gives the slope the sigmoid function,
#  this derivative is used to show how close the neural net is to
#  getting the right answer
def sigmoid_derivative(a):
    return a * (1.0 - a)





#  A class is used to bundle data and functions together
#  This class is our actual Neural Network of the program
#  it has 3 parts, initiation, feed forward, and back propagation
class NeuralNetwork:

    #  This is the initiation function, it is what the class does
    #  once it is called
    def __init__(self, x, y):
        #  This makes the input whatever x is, the input is x
        self.input     = x
        #  This is the first weight, it is used to give "importance" to
        #  some functions and less importance to others
        #  Fine tuning the weights an bases is what training the neural net does
        #
        #  np.random.rand() give random values from 0 to 1 in a given shape
        #  in this case the shape is - self.input.shape[1] rows (4) and 4 columns
        #  self.input.shape[1] tells the amount of rows in the input matrix
        self.weights1  = np.random.rand(self.input.shape[1],4)
        #  Sets the second set of weights to random numbers (from 0-1) with a shape of 4 rows
        #  and 3 columns
        self.weights2  = np.random.rand(4,1)
        #  This sets the second initial input to the variable y
        self.y         = y
        #   This outputs an array of zeros in the shape of the y matrix
        self.output    = np.zeros(y.shape)

    #  This is a feed forward function
    #  A feed forward function feeds the data that is given to the program through
    #  the program, this will be done thousands of times
    def feedforward(self):\

        #  Here is the actual neural network being worked through
        #  The layer 1 is created to be equal to the dot of the input and the weights and
        #  Then put on a sigmoid function
        #  This essentially changes the input by the weights
        #  This outputs the first step in the neural net
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        #  This is the next layer in the neural network
        #  It is also the output because this is a 2 layered network
        #  It does the same job as the last line of code, just on the output of that last line
        #  So it takes in the layer 1 data, dots it with the second set of weights, and dots them,
        #  Then put that on a sigmoid function, that is the output of the program
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    #  This is a back propagation function
    #  A back propagation function is used to tune the weights so that eventually
    #  The weights can correctly be used to guess the correct output
    #  This uses calculus (the derivative) to see the amount of change that the code needs to make
    #  Then it changes the weight by how drastic the derivative is
    #  Overtime the value gets closer and closer to 0
    #  This is gradient descent
    def backprop(self):
        #  Here is the second layer of weights going through gradient descent
        #  It starts at the back of the function, hence back propagation, it feeds from the back towards the front
        #  Here is the derivative of layer one doting with what the output should have been to get the
        #  right result, that shows how off it is
        d_weights2 = np.dot(self.layer1.T, (2*(self.y -self.output)*sigmoid_derivative(self.output)))
        #  This is the same as the last line except it is for the first set of weights
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output)*sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        #  Here the first set of weights is adjusted by the amount determined in line 79
        self.weights1 += d_weights1
        #  Here the second set of weights is adjusted by the amount determined in line 81
        self.weights2 += d_weights2



#  This is where the finished neural network is told to run
#  This line is executed once python starts, so this starts the neural net
if __name__ == "__main__":
    #  This is the input of the neural network, this is the data that is trained on, it can be changed
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    #  This is the desired output of the neural net, it is what the program is training itself towards
    y = np.array([[0],[1],[1],[0]])
    #  This just puts that past x and y into the neural network
    nn = NeuralNetwork(X,y)

    #  This is used to say how many times the neural network should train
    #  That is doing one feed forward, one back propagation, repeat
    for i in range(15000):
        nn.feedforward()
        nn.backprop()

    #  This outputs the final result
    print(nn.output)
