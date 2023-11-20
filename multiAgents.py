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


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth=2):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = better
        self.depth = int(depth)

    def minimax(self, gameState, depth, agentIndex):
        # Check if the game is over or reached the maximum depth
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        successors = [
            gameState.generateSuccessor(agentIndex, action) for action in actions
        ]

        if agentIndex == 0:  # Pacman's turn
            # Find the maximum value among the successors
            return max(self.minimax(successor, depth, 1) for successor in successors)

        # Ghost's turn
        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth - (
            nextAgentIndex == 0
        )  # Reduce depth when all agents have moved
        # Find the minimum value among the successors
        return min(
            self.minimax(successor, nextDepth, nextAgentIndex)
            for successor in successors
        )

    def getAction(self, gameState):
        actions = gameState.getLegalActions(0)  # Pacman's actions
        successors = [gameState.generateSuccessor(0, action) for action in actions]

        # Find the action that leads to the maximum value
        return max(
            actions,
            key=lambda action: self.minimax(
                successors[actions.index(action)], self.depth, 1
            ),
        )


class ExpectimaxAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth=2):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = better
        self.depth = int(depth)

    def expectimax(self, gameState, depth, agentIndex):
        # Check if the game is over or reached the maximum depth
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        successors = [
            gameState.generateSuccessor(agentIndex, action) for action in actions
        ]

        if agentIndex == 0:  # Pacman's turn
            # Find the maximum value among the successors
            return max(self.expectimax(successor, depth, 1) for successor in successors)

        # Ghost's turn
        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth - (
            nextAgentIndex == 0
        )  # Reduce depth when all agents have moved

        # Calculate the expected value among the successors
        expected_value = sum(
            self.expectimax(successor, nextDepth, nextAgentIndex)
            for successor in successors
        )
        return expected_value / len(successors)

    def getAction(self, gameState):
        actions = gameState.getLegalActions(0)  # Pacman's actions
        successors = [gameState.generateSuccessor(0, action) for action in actions]

        # Find the action that leads to the maximum value
        return max(
            actions,
            key=lambda action: self.expectimax(
                successors[actions.index(action)], self.depth, 1
            ),
        )


def betterEvaluationFunction(gameState: GameState):
    # Extract useful information from the game state
    pacmanPosition = gameState.getPacmanPosition()
    foodGrid = gameState.getFood()
    remainingFood = foodGrid.asList()
    ghostStates = gameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    score = gameState.getScore()
    capsules = gameState.getCapsules()

    # Calculate the evaluation score
    evaluationScore = 0
    evaluationScore += score / 5  # Add the current score
    evaluationScore += 12 / bfsFood(gameState)  # Add the distance to the nearest food
    evaluationScore -= 12 * len(
        remainingFood
    )  # Subtract a penalty based on the number of remaining food pellets

    # Add a bonus for eating scared ghosts
    for i in range(0, len(ghostStates)):
        if scaredTimes[i] > 0:
            ghostPosition = ghostStates[i].getPosition()
            GhostDistance = manhattanDistance(pacmanPosition, ghostPosition)
            if (GhostDistance < scaredTimes[i]) & (GhostDistance != 0):
                evaluationScore += 50 / GhostDistance

    # Add a bonus for reaching capsules
    if capsules:
        minCapsuleDistance = min(
            [manhattanDistance(pacmanPosition, capsule) for capsule in capsules]
        )
        evaluationScore += 2 / minCapsuleDistance

    return evaluationScore


# Abbreviation
better = betterEvaluationFunction


def bfsFood(currentGameState: GameState):
    walls = currentGameState.getWalls()
    height = sum(1 for _ in walls)
    width = sum(1 for _ in walls[0])

    start_pos = currentGameState.getPacmanPosition()
    visited = set()
    queue = util.Queue()
    queue.push([start_pos, 0])

    while not queue.isEmpty():
        head = queue.pop()
        pos, dist = head
        x, y = pos
        if currentGameState.hasFood(x, y):
            return dist
        if pos in visited:
            continue
        visited.add(pos)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if 0 < new_x < height and 0 < new_y < width and not walls[new_x][new_y]:
                queue.push([(new_x, new_y), dist + 1])
    return float("inf")  # If no food found
