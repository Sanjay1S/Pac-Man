# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    boundary = util.Stack() # Stack to trace the graph
    visited = set() #Track the visited nodes 
    boundary.push((problem.getStartState(), []))
    #Loop through the graph and find the goal state
    while not boundary.isEmpty():
        state, route = boundary.pop()
        if problem.isGoalState(state):
            return route
        visited.add(state) 
        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                newPath = route + [action]
                boundary.push((successor, newPath))                     
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    visited = set() #Keep a set of visited nodes
    boundary = util.Queue() # Track the nodes using a queue.
    boundary.push((problem.getStartState(), []))
    # Loop through the graph and find the goal state.
    while not boundary.isEmpty():
        state, route = boundary.pop()
        if problem.isGoalState(state):
            return route
        visited.add(state)
        for successor, action, stepcost in problem.getSuccessors(state):
            if successor not in visited and successor not in [s for s, p in boundary.list]:
                newPath = route + [action]
                boundary.push((successor, newPath))
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    boundary = util.PriorityQueue() # We use the priority queue here
    boundary.push((problem.getStartState(), []), 0)
    visited = set() # Keep track of visited node here.
    # Loop through the graph and find the goal state traversing along the path with least total cost first
    while not boundary.isEmpty():
        state, route = boundary.pop()
        if problem.isGoalState(state):
            return route
        if state not in visited:
            visited.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                newPath = route + [action]
                newCost = problem.getCostOfActions(newPath)
                if successor not in visited:
                    boundary.push((successor, newPath), newCost)
                elif successor in [s for s, p, c in boundary.heap]:
                    if newCost < problem.getCostOfActions([a for a in p]): #Compare step costs
                        boundary.update((successor, newPath), newCost)
    util.raiseNotDefined()


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from collections import defaultdict

    start_state = problem.getStartState()
    boundary = util.PriorityQueue() # Priority queue is used here
    boundary.push((start_state, [], 0), 0)
    explored = defaultdict(lambda: float('inf'))  # Keep a dictionary to track the explored nodes
    ## Loop through the graph and find goal state using heuristics[Costt so far and cost to goal]
    while not boundary.isEmpty():
        current_state, actions, current_cost = boundary.pop()

        if problem.isGoalState(current_state):
            return actions

        if current_cost < explored[current_state]: #Compare the costs
            explored[current_state] = current_cost

            for successor, action, step_cost in problem.getSuccessors(current_state):
                new_actions = actions + [action]
                new_cost = current_cost + step_cost
                if new_cost < explored[successor]:
                    boundary.push((successor, new_actions, new_cost), new_cost + heuristic(successor, problem))

    util.raiseNotDefined()




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

