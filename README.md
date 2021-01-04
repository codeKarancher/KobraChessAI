# KobraChessAI
A rudimentary chess AI that uses deep learning to evaluate a chess board to play chess.

## Training the model
### Inputs
Currently, the neural network takes in a 64-long array of integers denoting the value of pieces at every square on the board.
The model will soon change in order to incorporate special treatment for the King piece, since assigning the king a centipawn value like every other piece is not representative of the meaning of the King in chess.
### Phase 1: Stockfish
Initially, in order to train the board evaluation function towards the theoretical optimum, Stockfish is used to serve as a coach, or guide, for the model. This is done by generating a dataset of random chess boards and their stockfish evaluations.

Here, we are assuming that the Stockfish board evaluation function is in the vicinity of the theoretical optimum that we are driving towards with the KobraChessAI model.
### Phase 2: Genetic Learning
Once the model is in the vicinity of the theoretical optimal function, we will aim to go above and beyond Stockfish. This will be done using a genetic algorithm to slowly ascend and produce improving board evaluation functions with every generation.
