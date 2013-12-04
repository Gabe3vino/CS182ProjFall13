# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Refiners: Brian Connolly and Gabriel Trevino

from util import manhattanDistance
from game import Directions
import random, util, search, searchAgents

from game import Agent

class ReflexAgent(Agent):
	"""
	A reflex agent chooses an action at each choice point by examining
	its alternatives via a state evaluation function.
	
	The code below is provided as a guide.  You are welcome to change
	it in any way you see fit, so long as you don't touch our method
	headers.
	"""
	
	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.
		
		getAction chooses among the best options according to the evaluation function.
		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {North, South, West, East, Stop}
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
		
	def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.
		
		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.
		
		The code below extracts some useful information from the state, like the
		remaining food (oldFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.
		
		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		oldFood = currentGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
		"*** YOUR CODE HERE ***"
		return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
	"""
	This default evaluation function just returns the score of the state.
	The score is the same one displayed in the Pacman GUI.
	
	This evaluation function is meant for use with adversarial search agents
	(not reflex agents).
	"""
	return currentGameState.getScore()

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
		self.numAg = 0

	# creates a function in self to calculate if a leaf is a terminal node
	def termTest(self, state, layern):
		return (layern == (self.depth*self.numAg) or state.isWin() or state.isLose())

class MinimaxAgent(MultiAgentSearchAgent):
	# runs a minimax function similar to the one in the book, except that we keep
	# track of the agents and the agent in the layer. Effectively counts linearly
	# instead of thinking about it as a tree
	def minimax(self, state, an, actions, layern):
		actionList = []
		for a in actions:
			util = self.direct(state.generateSuccessor(an,a),an+1,layern+1)
			actionList.append((a,util))
		actionList.sort(key=lambda x: x[1], reverse=True)
		return actionList[0][0]
	
	# if its the pac-man, find max, otherwise do min (it's a ghost!)
	def direct(self, state, an, layern):
		agent = an % self.numAg
		if agent == 0:
			return self.maxval(state, agent, layern)
		else:
			return self.minval(state, agent, layern)
  
	# returns the max value, based on algorithm in text
	def maxval(self, state, an, layern):
		if self.termTest(state, layern):
			return self.evaluationFunction(state)
		v = float('-inf')
		actions = state.getLegalActions(an)
		actions.remove(Directions.STOP)
		for a in actions:
			v = max(v,self.direct(state.generateSuccessor(an,a), an+1, layern+1))          
		return v
  
	# returns the min value, based on the book
	def minval(self, state, an, layern):
		if self.termTest(state, layern):
			return self.evaluationFunction(state)
		v = float('inf')
		for a in state.getLegalActions(an):
			v = min(v,self.direct(state.generateSuccessor(an,a), an+1, layern+1))          
		return v
	
	def getAction(self, gameState):
		"""
		Returns the minimax action from the current gameState using self.depth
		and self.evaluationFunction.

		Here are some method calls that might be useful when implementing minimax.
	
		gameState.getLegalActions(agentIndex):
		Returns a list of legal actions for an agent
		agentIndex=0 means Pacman, ghosts are >= 1
		
		Directions.STOP:
		The stop direction, which is always legal
		
		gameState.generateSuccessor(agentIndex, action):
		Returns the successor game state after an agent takes an action
		
		gameState.getNumAgents():
		Returns the total number of agents in the game
		"""
		self.numAg = gameState.getNumAgents()
		actionList = []
		actions = gameState.getLegalActions(agent)
		actions.remove(Directions.STOP)
		return self.minimax(gameState, 0, actions, 0)

# creates a class that makes packaging alpha-beta easier, just holds an action and value
class ABNode():
	def __init__(self, val, action):
		self.value = val
		self.action = action

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent with alpha-beta pruning (question 3)
	"""
	# slightly different than minimax (return the action, don't need a list to sort through options
	def alphabeta(self, an, layern, state):
		self.numAg = state.getNumAgents()
		return self.abdirect(state, an, layern, Directions.STOP, float('-inf'), float('inf')).action
	
	# if pacman, calculate max, otherwise min
	def abdirect(self, state, an, layern, act, alpha, beta):
		agent = an % self.numAg
		if agent == 0:
			return self.maxval(state, agent, layern, act, alpha, beta)
		else:
			return self.minval(state, agent, layern, act, alpha, beta)
			
	# minimax max function, except only calculates if greater than beta
	def maxval(self, state, an, layern, act, alpha, beta):
		if self.termTest(state, layern):
			return ABNode(self.evaluationFunction(state), act)
		v = ABNode(float('-inf'), Directions.STOP)
		actions = state.getLegalActions(an)
		actions.remove(Directions.STOP)
		for a in actions:
			temp = self.abdirect(state.generateSuccessor(an,a), an+1, layern+1, a, alpha, beta)
			temp.action = a
			if temp.value > v.value:
				v = temp
			if v.value >= beta:
				return v
			alpha = max(alpha,v.value)
		return v
  
	# minimax min function, only calculates if less than alpha
	def minval(self, state, an, layern, act, alpha, beta):
		if self.termTest(state, layern):
			return ABNode(self.evaluationFunction(state), act)
		v = ABNode(float('inf'), Directions.STOP)
		for a in state.getLegalActions(an):
			temp = self.abdirect(state.generateSuccessor(an,a), an+1, layern+1, a, alpha, beta)
			temp.action = a
			if temp.value < v.value:
				v = temp
			if v.value <= alpha:
				return v
			beta = min(v.value, beta)
		return v
  
	def getAction(self, gameState):
		"""
		Returns the minimax action using self.depth and self.evaluationFunction
		"""
		return self.alphabeta(0, 0, gameState)\

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  
  # analogous to minimiax, except with differnt expectations
  def expectimax(self, state, an, actions, layern):
	actionList = []
	for a in actions:
		util = self.expectiDirect(state.generateSuccessor(an,a),an+1,layern+1)
		actionList.append((a,util))
	actionList.sort(key=lambda x: x[1], reverse=True)
	return actionList[0][0]
  
  # tells whether to find max or expected val (assuming randomness)
  def expectiDirect(self,state,an,layern):
	agent = an % self.numAg
	if agent == 0:
		return self.maxval(state,agent,layern)
	else:
		return self.expectedval(state,agent,layern)
    
	# finds average when necessary
  def avg(self, one, two):
    if one == float('inf'):
      return two
    else:
      return (one + two) / 2

  def maxval(self,state,an,layern):
	if self.termTest(state, layern):
		return self.evaluationFunction(state)
	v = float("-inf")
	actions = state.getLegalActions(an)
	actions.remove(Directions.STOP)
	for a in actions:
		v = max(v,self.expectiDirect(state.generateSuccessor(an,a),an+1,layern+1))          
	return v
  
  def expectedval(self,state,an,layern):
    v = float("inf")
    for a in state.getLegalActions(an):
      v = self.avg(v, self.expectiDirect(state.generateSuccessor(an,a),an+1,layern+1))  
    return v
  
  def getAction(self, gameState):
	"""
	Returns the minimax action from the current gameState using self.depth
	and self.evaluationFunction.
	
	Here are some method calls that might be useful when implementing minimax.
	
	gameState.getLegalActions(agentIndex):
	Returns a list of legal actions for an agent
	agentIndex=0 means Pacman, ghosts are >= 1
    
	Directions.STOP:
	The stop direction, which is always legal
    
	gameState.generateSuccessor(agentIndex, action):
	Returns the successor game state after an agent takes an action
    
	gameState.getNumAgents():
	Returns the total number of agents in the game
	"""	
	self.numAg = gameState.getNumAgents()
	depth = 0
	agent = 0
	actions = gameState.getLegalActions(agent)
	actions.remove(Directions.STOP)
	return self.expectimax(gameState, agent, actions, depth)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
  """

  pacPos = currentGameState.getPacmanPosition()
  numCapsules = len(currentGameState.getCapsules())

  # Determine distance to nearest Ghost, don't care if farther than 7
  gDistance = 7
  for pos in currentGameState.getGhostPositions():
    problem = searchAgents.PositionSearchProblem(currentGameState, goal=pos, start=pacPos)
    gDistance = min(gDistance, len(search.breadthFirstSearch(problem)))

  # Determine distance to nearest food
  fDistance = 0
  
  foodGrid = currentGameState.getFood()
  foodList = foodGrid.asList()
  numFood = len(foodList)
  if len(foodList) > 0:
    fProblem = searchAgents.PositionSearchProblem(currentGameState, goal=foodList[0], start=pacPos)
    fDistance = len(search.breadthFirstSearch(problem))

  # Make shorter ghost distance attractive when ghosts are scared
  newGhostStates = currentGameState.getGhostStates()
  newScaredTime = 0
  newScaredTime = reduce(lambda x, y: x.scaredTimer + y.scaredTimer, newGhostStates)
  if newScaredTime > 6:
    gDistance = -gDistance

  px, py = pacPos
  fDensity = 0

  def minus1(l):
    l[:] = [x - 1 for x in l]
    return l

  width = len(foodGrid[:][0])
  height = len(foodGrid[0])

  # Compute density of food surrounding Pacman
  for i in minus1(range(5)):
    intx = px + i
    if intx < 0 or intx > width-1:
      continue

    for j in minus1(range(5)):
      inty = py + j
      if inty < 0 or inty > height-1:
        continue
      if foodGrid[intx][inty]:
        fDensity += 1

  # Return linear combination of factors
  return 3 * gDistance - 13*numCapsules + 1.0/(fDistance+1) + 1*fDensity - 2*numFood

# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()