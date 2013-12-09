# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
import math

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  """
  nodeStack = util.Stack()
  tup = (problem.getStartState(), Directions.STOP, 0, None, [])
  nodeStack.push(tup)
  visited = []

  while True:
    if nodeStack.isEmpty():
      return []

    node = nodeStack.pop()
    state, action, cost, parent, actions = node
    
    if state in visited:
      continue

    visited.append(state) 

    if problem.isGoalState(state):
      return actions
    else:
      for x in problem.getSuccessors(state):
        child, action, cost = x
        x = (x + (state, actions + [action]))
        if not(child in visited):
          nodeStack.push(x)

def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
  nodeStack = util.Queue()
  tup = (problem.getStartState(), Directions.STOP, 0, None, [])
  nodeStack.push(tup)
  visited = []

  while True:
    if nodeStack.isEmpty():
      return []

    node = nodeStack.pop()
    state, action, cost, parent, actions = node
    
    if state in visited:
      continue

    visited.append(state) 

    if problem.isGoalState(state):
      return actions
    else:
      for x in problem.getSuccessors(state):
        child, action, cost = x
        x = (x + (state, actions + [action]))
        if not(child in visited):
          nodeStack.push(x)
      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  nodeStack = util.PriorityQueue()
  tup = (problem.getStartState(), Directions.STOP, 0, None, [])
  nodeStack.push(tup, 0)
  visited = []

  while True:
    if nodeStack.isEmpty():
      return []

    node = nodeStack.pop()
    state, action, pcost, parent, actions = node
    
    if state in visited:
      continue

    visited.append(state) 

    if problem.isGoalState(state):
      return actions
    else:
      for x in problem.getSuccessors(state):
        child, action, cost = x
        cost = pcost + cost
        x = (child, action, cost, state, actions + [action])
        if not(child in visited):
          nodeStack.push(x, cost)

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  nodeStack = util.PriorityQueue()
  tup = (problem.getStartState(), Directions.STOP, 0, None, [])
  nodeStack.push(tup, 0)
  visited = []

  while True:
    if nodeStack.isEmpty():
      return []

    node = nodeStack.pop()
    state, action, pcost, parent, actions = node
    
    if state in visited:
      continue

    visited.append(state)

    if problem.isGoalState(state):
      return actions
    else:
      for x in problem.getSuccessors(state):
        child, action, cost = x
        cost = pcost + cost
        x = (child, action, cost, state, actions  + [action])
        if not(child in visited):
          nodeStack.push(x, cost + heuristic(state, problem))
    
  
# Implement bug algorithms HERE
def bug1(problem, heuristic=nullHeuristic):
  '''
    Search the node that is closest to goal until we hit obstacle
    To run: python pacman.py -l basicRobot0 -p SearchAgent -a fn=bug1,heuristic=manhattanHeuristic
  '''
  actions = []
  state = problem.getStartState()
  left = Directions.LEFT
  right = Directions.RIGHT

  while True:
    if problem.isGoalState(state):
      return actions

    bestDistance = float('Inf')
    bestAction = Directions.STOP
    bestState = state

    children = problem.getSuccessors(state)

    # If wall is blocking us, circumnavigate obstacle
    if len(actions) > 0 and obstacleIsBlocking(actions[-1], children):
      origPosition = state
      closestPoint = (state,[])
      closestDist = heuristic(state, problem)

      # Turn left, always keep wall on right side (we could've done opposite)
      myRight = left[actions[-1]]
      obstacleActions = []
      firstTime = True

      while state != origPosition or firstTime:
        firstTime = False
        children = problem.getSuccessors(state)
        myRight = right[myRight]

        # Turn right if we can, otherwise go straight
        while obstacleIsBlocking(myRight, children):
          myRight = left[myRight]

        obstacleActions.append(myRight)
        state = getChildWithAction(myRight, children)
        currDist = heuristic(state, problem)

        if currDist < closestDist:
          closestPoint = (state, obstacleActions[:])
          closestDist = currDist


      # Once we get back to original point of incidence travel back to closest point
      state, halfwayActions = closestPoint
      actions = actions + obstacleActions + halfwayActions

    children = problem.getSuccessors(state)

    for x in children:
      child, action, cost = x

      distance = towardsGoalHeuristic(state, problem, child)
      if distance < bestDistance:
        bestDistance = distance
        bestAction = action
        bestState = child

    actions.append(bestAction)
    state = bestState
    print actions


def obstacleIsBlocking(prevAction, children):
  possActions = []
  for child in children:
    _, action, _ = child
    possActions.append(action)

  return prevAction not in possActions

def getChildWithAction(action, children):
  for child in children:
    state, childAction, _ = child
    
    if action is childAction:
      return state

# Heuristic which gives highest value to closest direction towards goal, not a true heuristic
def towardsGoalHeuristic(position, problem, child):
  deltaX = child[0] - position[0]
  deltaY = child[1] - position[1]
  goalDeltaX = problem.goal[0] - position[0]
  goalDeltaY = problem.goal[1] - position[1]

  direction = math.atan2(deltaY, deltaX)
  goalDirection = math.atan2(goalDeltaY, goalDeltaX)

  return abs(direction-goalDirection)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch