# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

def getClosestFood(pacPos: tuple, foodArray: list):
    closestDist = 9999999999
    for food in foodArray:
        currDist = manhattanDistance(pacPos, food)
        if  closestDist > currDist:
            closestDist = currDist
    if closestDist == 9999999999:
        return 0
    return closestDist

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
    
    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        foodList = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print(getClosestFood(newPos, foodList))
        evaluation = successorGameState.getScore() - getClosestFood(newPos, foodList) * 0.1 + sum(newScaredTimes) - foodList.__len__() * 3
        print(evaluation)
        return evaluation

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

def value(gameState: GameState, agentIndex: int, depth: int):
    if agentIndex >= gameState.getNumAgents():
        agentIndex = 0
        depth -= 1
    if gameState.isWin() or gameState.isLose() or depth == 0:
        return scoreEvaluationFunction(gameState)
    

    if agentIndex == 0:
        return maxValue(gameState, agentIndex, depth)
    return minValue(gameState, agentIndex, depth)

def maxValue(gameState: GameState, agentIndex: int, depth: int):
    v = -99999999
    legalMoves = gameState.getLegalActions(agentIndex)
    for move in legalMoves:
        successorState = gameState.generateSuccessor(agentIndex, move)
        v = max(v, value(successorState, agentIndex + 1, depth))
    return v
    
def minValue(gameState: GameState, agentIndex: int, depth: int):
    v = 99999999
    legalMoves = gameState.getLegalActions(agentIndex)
    for move in legalMoves:
        successorState = gameState.generateSuccessor(agentIndex, move)
        v = min(v, value(successorState, agentIndex + 1, depth))
    return v

def alphaBetaValue(gameState: GameState, agentIndex: int, depth: int, alpha: int, beta: int):
    if agentIndex >= gameState.getNumAgents():
        agentIndex = 0
        depth -= 1

    if gameState.isWin() or gameState.isLose() or depth == 0:
        return scoreEvaluationFunction(gameState)
    
    if agentIndex == 0:
        return alphaBetaMaxValue(gameState, agentIndex, depth, alpha, beta)
    return alphaBetaMinValue(gameState, agentIndex, depth, alpha, beta)

def alphaBetaMaxValue(gameState: GameState, agentIndex: int, depth: int, alpha: int, beta: int):
    v = -99999999
    legalMoves = gameState.getLegalActions(agentIndex)
    for move in legalMoves:
        successorState = gameState.generateSuccessor(agentIndex, move)
        v = max(v, alphaBetaValue(successorState, agentIndex + 1, depth, alpha, beta))
        if v > beta :
            return v
        alpha = max(alpha, v)
        print("MAX alpha, beta: ", alpha, beta)
    return v
    
def alphaBetaMinValue(gameState: GameState, agentIndex: int, depth: int, alpha: int, beta: int):
    v = 99999999
    legalMoves = gameState.getLegalActions(agentIndex)
    for move in legalMoves:
        successorState = gameState.generateSuccessor(agentIndex, move)
        v = min(v, alphaBetaValue(successorState, agentIndex + 1, depth, alpha, beta))
        if v < alpha:
            return v
        beta = min(beta, v)
        print("MIN alpha, beta: ", alpha, beta)
        print("v: ", v)
    return v

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
    
class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState: GameState):

        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        
        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        legalMoves = gameState.getLegalActions(0)
        highestValue = -999999999999
        highestMove = None
        for move in legalMoves:
            successorState = gameState.generateSuccessor(0, move)
            currValue = value(successorState, 1, self.depth)
            if currValue > highestValue:
                highestValue = currValue
                highestMove = move
        return highestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        legalMoves = gameState.getLegalActions(0)
        highestValue = -999999999999
        highestMove = None
        alpha = float('-inf')
        beta = float('inf')
        for move in legalMoves:
            successorState = gameState.generateSuccessor(0, move)
            currValue = alphaBetaValue(successorState, 1, self.depth, alpha, beta)
            if currValue > highestValue:
                highestValue = currValue
                highestMove = move
            alpha = max(alpha, currValue)
        return highestMove
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def exptimaxValue(self, gameState: GameState, agentIndex: int, depth: int):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth -= 1

        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.expMaxValue(gameState, agentIndex, depth)
        return self.expValue(gameState, agentIndex, depth)

    def expMaxValue(self, gameState: GameState, agentIndex: int, depth: int):
        v = -99999999
        legalMoves = gameState.getLegalActions(agentIndex)
        for move in legalMoves:
            successorState = gameState.generateSuccessor(agentIndex, move)
            v = max(v, self.exptimaxValue(successorState, agentIndex + 1, depth))
        return v

    def expValue(self, gameState: GameState, agentIndex: int, depth: int):
        v = 0
        legalMoves = gameState.getLegalActions(agentIndex)
        for move in legalMoves:
            successorState = gameState.generateSuccessor(agentIndex, move)
            p = (1 / legalMoves.__len__())
            v += p * self.exptimaxValue(successorState, agentIndex + 1, depth)
        return v  
    
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        highestValue = -999999999999
        highestMove = None
        for move in legalMoves:
            successorState = gameState.generateSuccessor(0, move)
            currValue = self.exptimaxValue(successorState, 1, self.depth)
            if currValue > highestValue:
                highestValue = currValue
                highestMove = move
        return highestMove

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """
    pacPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    
    # Score
    evaluation = currentGameState.getScore()
    
    # Food
    if foodList:
        closestFoodDist = getClosestFood(pacPos, foodList)
        evaluation -= closestFoodDist
        evaluation -= len(foodList) * 4
    
    # Ghosts
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDist = manhattanDistance(pacPos, ghostPos)
        
        if ghost.scaredTimer > 0:
            if ghostDist <= ghost.scaredTimer:
                evaluation += 100 / (ghostDist + 1)
        elif ghostDist <= 1:
            evaluation -= 500
    
    # Power pellets
    if capsules:
        closestCapsuleDist = min([manhattanDistance(pacPos, cap) for cap in capsules])
        evaluation += 10 / (closestCapsuleDist + 1)
    
    return evaluation

# Abbreviation
better = betterEvaluationFunction
