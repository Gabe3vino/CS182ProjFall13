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
    
def manhattanHeuristic(position, problem, info={}):
  "The Manhattan distance heuristic for a PositionSearchProblem"
  xy1 = position
  xy2 = problem.goal
  return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

  
# Implement bug algorithms HERE
def bug1(problem, heuristic=manhattanHeuristic):
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


def obstacleIsBlocking(desiredAction, children):
  possActions = []
  for child in children:
    _, action, _ = child
    possActions.append(action)

  return desiredAction not in possActions

def getChildWithAction(action, children):
  for child in children:
    state, childAction, _ = child
    
    if action is childAction:
      return state

'''
  Bug2 creates an M Line that connects the starting position to the goal. It attempts to follow this
  Line. If it hits an obstacle on its way, it follows the obstacle until it reaches a closer point on
  the M Line. This algorithm is complete.
'''
def bug2(problem, heuristic=manhattanHeuristic):
  actions = []
  state = problem.getStartState() # Current state
  left = Directions.LEFT
  right = Directions.RIGHT
  onMLine = makeMLine(state, problem.goal)

  # Loop until we reach the goal state
  while True:
    if problem.isGoalState(state):
      return actions

    bestAction = getActionOnMLine(state, onMLine, problem)
    children = problem.getSuccessors(state)

    # If wall is blocking us, wall follow obstacle until we reach a closer point on the M line
    if obstacleIsBlocking(bestAction, children):
      startDistance = manhattanHeuristic(state, problem)

      # Turn left, always keep wall on right side (we could've done opposite)
      myRight = left[bestAction]
      obstacleActions = []
      firstTime = True

      # Follow obstacle until we reach a point on the M Line that is closer than the last M Line point
      while (not onMLine(state) or manhattanHeuristic(state, problem) >= startDistance) or firstTime:
        firstTime = False
        children = problem.getSuccessors(state)
        myRight = right[myRight]

        # Turn right if we can, otherwise go straight
        while obstacleIsBlocking(myRight, children):
          myRight = left[myRight]

        obstacleActions.append(myRight)
        state = getChildWithAction(myRight, children)

      # Once we get back to MLine continue on it
      actions = actions + obstacleActions
      continue

    actions.append(bestAction)
    state = getChildWithAction(bestAction, children)
    print actions

'''
  Heuristic which gives highest value to closest direction towards goal, not a
  true heuristic. Not proven to be inadmissable.
'''
def towardsGoalHeuristic(position, problem, child):
  deltaX = child[0] - position[0]
  deltaY = child[1] - position[1]
  goalDeltaX = problem.goal[0] - position[0]
  goalDeltaY = problem.goal[1] - position[1]

  direction = math.atan2(deltaY, deltaX)
  goalDirection = math.atan2(goalDeltaY, goalDeltaX)

  return abs(direction-goalDirection)


indexToAction = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST,
                Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

'''
# Given a state on the M Line, looks at the surrounding 8 squares to figure out
# which is the next closest position to the goal on the M Line. 
# Returns the single action to get to this desired position (or closer to it, as in the case of the diagonal squares)
'''
def getActionOnMLine(state, onMLine, problem):
  children = []
  children.append((state[0],  state[1]+1))
  children.append((state[0],  state[1]-1))
  children.append((state[0]+1,state[1]))
  children.append((state[0]-1,state[1]))

  children.append((state[0]+1,state[1]+1))  
  children.append((state[0]-1,state[1]-1))
  children.append((state[0]+1,state[1]-1))
  children.append((state[0]-1,state[1]+1))

  bestDist = float('Inf')
  bestIndex = None

  # Looks at surrounding squares and determines which is the next on the M Line
  for i in range(len(children)):
    if onMLine(children[i]):
      dist = manhattanHeuristic(children[i], problem)
      if dist < bestDist:
        bestDist = dist
        bestIndex = i
  
  if bestIndex is None:
    return None
  else:
    return indexToAction[bestIndex]

# Returns the function, which given a position, will determine if said location is on the M Line
def makeMLine(start, goal):
  deltaX = float(goal[0] - start[0])
  deltaY = float(goal[1] - start[1])
  print "DeltaX", deltaX
  print "DeltaY", deltaY

  def onMLine(position):

    desiredY = int(round(deltaY/deltaX * (position[0]-start[0]) + start[1]))
    print position, desiredY

    if position[1] == desiredY:
      return True
    else:
      return False

  return onMLine


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch