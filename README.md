# KobraChessAI
A rudimentary chess AI that uses deep learning to evaluate a chess board to play chess.

## Abstract
The KobraChessAI will work using a model that has been trained to accurately evaluate the winnability of a chess board for the white player. Cycling through the legal moves and finding the move that maximises this winnability (or minimises if the AI is playing black) will then determine what KobraChessAI thinks is the best move to play. Beyond this, in many ways, this project's goal is quite independent of the game of chess. The goal is to approximate, as accurately yet cheaply as possible, the best possible board evaluation function, and this process is not dissimilar from any other function optimisation problem.

There are two sides to the coin. The first is to be able to take a model design and train it to predict the winnability of a chess board as accurately as possible. For this project, the over-arching principle is to use a deep neural network model, so it will be trained using backpropogation and gradient descent. Tensorflow's Keras will be used for this. The second, less straight-forward aspect to the project is to really find out what kind of n-network structure will ultimately yield the most accurate model.

Finally, let's talk a bit about measuring the 'accuracy' of such a model. Whether a value of winnability can be assigned to a given chess board at all is itself quite a philosophical question. For this project, we shall assume that there exists a theoretically optimal function that maps chess boards to their winnability for white (better than all other functions). Any model developed to be used by KobraChessAI will inevitably be an approximation to this theoretically optimal function, and hence its accuracy is, theoretically speaking, how well it predicts the optimal function. Since this function has not (yet?) been determined, we cannot measure the absolute accuracy of any given model. All we can do is reason about whether a certain model is more accurate than another, and this can be done by pitting the models against each other in a game (multiple games) of chess!

## Training the model
### Phase 1: Stockfish
Initially, in order to fast-track the model towards the theoretically optimal board evaluation function, Stockfish is used to serve as a coach, or guide, for the model. This is done by training and validating the model on a dataset of random chess boards and their stockfish evaluations; batches for this are dynamically generated on demand given the memory and computational constraints of my personal machine.

Here, we are assuming that the Stockfish board evaluation function is in the vicinity of the global maxima of board evaluation functions that we are driving towards with the KobraChessAI model.
### Phase 2: Genetic Learning
Once the model is in the vicinity of the theoretically optimal function, we will aim to go above and beyond Stockfish. This will be done using a genetic algorithm to slowly ascend and produce improving board evaluation models with every generation.

## Model Structure
### Inputs
Currently, the neural network takes an input consisting of 320 nodes, calculated as 64x5. For each of the 64 squares on the chess board, 5 separate values are fed into the network. These values represent a semi one-hot encoding of the chess piece situated on that square.

If you are wondering how 6 possible types of pieces are being fed into the network using a 5-long one-hot vector, this is the result of a bit of trickery. The queen is a piece that can move like a bishop, or like a rook. Therefore, the value of a chess board with the queen at a certain square can be treated as the sum of the values of the same chess board with a bishop in place of the queen, and with the rook in place of the queen. This method allows the queen to be encoded as a two-hot vector representing a bishop and a rook at the same time (hence the 'semi' one-hot nomenclature), saving some training time and prediction time of the model.
### The Insides
Currently, the neural networks being trained use a computationally reasonable number of densely connected hidden layers with around 32 nodes each (sn3 has 3 hidden layers with 32,32,8 nodes respectively).
