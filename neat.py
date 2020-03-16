import math
import random


# for virtualization
import pygame
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont("Curier New", 25)
GAMEDISPLAY = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption(f"Neat Network Virtualizer")
CLOCK = pygame.time.Clock()
BACKGROUND = pygame.Surface(GAMEDISPLAY.get_size())
BACKGROUND = BACKGROUND.convert()
BACKGROUND.fill(WHITE)
GAME_QUIT = False


nextConnectionNumber = 0  # put this in population?!

# todo: all percent chanches in a config file
#       => in mutate
#       => in crossover
#       => where else?!


class Network:
    def __init__(self, numberOfInputs, numberOfOutputs):
        self.fitness = 0
        self.unadjustedFitness = 0
        self.score = 0
        self.gen = 0
        self.dead = False
        self.lifespan = 0  # how long the player lived for fitness

        self.vision = [0 for _ in range(numberOfInputs)]  # input values
        self.decision = [0 for _ in range(
            numberOfOutputs)]  # output of network

        # self.genomeInputs = numberOfInputs
        # self.genomeOutputs = numberOfOutputs

        self.genome = Genome(numberOfInputs, numberOfOutputs)


class Genome:
    def __init__(self, numberOfInputs, numberOfOutputs, empty=False):
        if not empty:  # normal genome
            self.genes = []  # connections between nodes
            self.nodes = []  # nodes
            self.network = []  # nodes in order of consideration

            self.inputs = numberOfInputs
            self.outputs = numberOfOutputs

            self.layers = 2
            self.nextNode = 0
            self.biasNode = 0

            # create inputs
            for _ in range(numberOfInputs):
                self.nodes.append(Node(self.nextNode))
                self.nodes[self.nextNode].setLayer(0)
                self.nextNode = self.nextNode + 1

            # create outputs
            for _ in range(numberOfOutputs):
                self.nodes.append(Node(self.nextNode))
                self.nodes[self.nextNode].setLayer(1)
                self.nextNode = self.nextNode + 1

            # create bias node
            self.nodes.append(Node(self.nextNode))
            self.biasNode = self.nextNode
            self.nodes[self.biasNode].setLayer(0)
            self.nextNode = self.nextNode + 1
        else:  # empty genome for cloning
            self.inputs = numberOfInputs
            self.outputs = numberOfOutputs

    def getNode(self, number):
        if number < len(self.nodes):
            return self.nodes[number]

        return None

    def connectNodes(self):
        for node in self.nodes:
            node.outputConnections = []
        for gene in self.genes:
            gene.fromNode.outputConnections.append(gene)

    def feedForward(self, inputValues):
        # output of input nodes are the input values
        for n in range(len(self.inputs)):
            self.nodes[n].outputValue = inputValues[n]
        self.nodes[self.biasNode].outputValue = 1

        for net in self.network:
            net.engage()

        outs = []
        for n in range(len(self.outputs)):
            outs[n] = self.nodes[self.inputs + n].outputValue

        # reset all nodes
        for node in self.nodes:
            node.inputSum = 0

        return outs

    def generateNetwork(self):
        self.connectNodes()
        self.network = []

        # order network based on layer number for calculation priority
        for layerNumber in range(self.layers):
            for node in self.nodes:
                if node.getLayer == layerNumber:
                    self.network.append(node)

    def addNode(self, history):
        if len(self.genes) == 0:
            self.addConnection(history)
            return

        randomConnection = math.floor(random.random() * len(self.genes))
        # do not disconnect bias
        while self.genes[randomConnection].fromNode == self.nodes[self.biasNode] and not len(self.genes) == 1:
            randomConnection = math.floor(random.random() * len(self.genes))

        self.genes[randomConnection].setEnabled(
            False)  # disable chosen connection gene

        self.nodes.append(Node(self.nextNode))
        connectionInnovationNumber = self.getInnovationNumber(
            history, self.genes[randomConnection].fromNode, self.getNode(self.nextNode))
        # new connection with weight 1
        self.genes.append(connectionGene(self.genes[randomConnection].fromNode, self.getNode(
            self.nextNode), 1, connectionInnovationNumber))
        connectionInnovationNumber = self.getInnovationNumber(
            history, self.getNode(self.nextNode), self.genes[randomConnection].toNode)
        # new connection with old weight
        self.genes.append(connectionGene(self.getNode(
            self.nextNode), self.genes[randomConnection].toNode, self.genes[randomConnection].weight, connectionInnovationNumber))

        self.getNode(self.nextNode).setLayer(
            self.genes[randomConnection].fromNode.getLayer() + 1)

        # connect new node to bias with weigth of 0
        connectionInnovationNumber = self.getInnovationNumber(
            history, self.nodes[self.biasNode], self.getNode(self.nextNode))
        self.genes.append(connectionGene(self.nodes[self.biasNode], self.getNode(
            self.nextNode), 0, connectionInnovationNumber))

        # if the layer of the new node is equal to the output layer of the old node,
        # all layers equal or higher have to be shifted
        if self.getNode(self.nextNode).getLayer() == self.genes[randomConnection].toNode.getLayer():
            for node in self.nodes[:-1]:  # except the new node itself
                if node.getLayer() >= self.getNode(self.nextNode).getLayer():
                    node.setLayer(node.getLayer() + 1)
            self.layers = self.layers + 1

        self.nextNode = self.nextNode + 1

        self.connectNodes()

    def addConnection(self, history):
        if self.fullyConnected():
            return  # cant add any more connections

        # get random nodes available for connection
        randomNode1 = math.floor(random.random() * len(self.nodes))
        randomNode2 = math.floor(random.random() * len(self.nodes))
        while self.badConnectionNodes(randomNode1, randomNode2):
            randomNode1 = math.floor(random.random() * len(self.nodes))
            randomNode2 = math.floor(random.random() * len(self.nodes))

        # first node must have first layer
        if self.nodes[randomNode1].getLayer() > self.nodes[randomNode2].getLayer():
            randomNode1, randomNode2 = randomNode2, randomNode1

        # innovation number is new number, if no identical genome has mutated in the same way
        connectionInnovationNumber = self.getInnovationNumber(
            history, self.nodes[randomNode1], self.nodes[randomNode2])

        # add connection with random weight
        self.genes.append(connectionGene(
            self.nodes[randomNode1], self.nodes[randomNode2], random.random() * 2 - 1, connectionInnovationNumber))

        self.connectNodes()

    def badConnectionNodes(self, r1, r2):
        # bad if nodes are on same layer
        if self.nodes[r1].getLayer() == self.nodes[r2].getLayer():
            return True

        # bad if nodes are already connected
        if self.nodes[r1].isConnectedTo(self.nodes[r2]):
            return True

        return False

    def getInnovationNumber(self, history, fromNode, toNode):
        global nextConnectionNumber
        isNew = True
        connectionInnovationNumber = nextConnectionNumber

        for historyEvent in history:
            if historyEvent.matches(self, fromNode, toNode):
                isNew = False
                connectionInnovationNumber = historyEvent.innovationNumber
                break

        if isNew:  # if its a new mutation generate list representing current state
            innoNumbers = []
            for gene in self.genes:
                innoNumbers.append(gene.innovationNr)

            history.append(connectionHistory(
                fromNode.number, toNode.number, connectionInnovationNumber, innoNumbers))
            nextConnectionNumber = nextConnectionNumber + 1

        return connectionInnovationNumber

    def fullyConnected(self):
        maxConnections = 0
        nodesInLayer = [0 for _ in range(self.layers)]

        for node in self.nodes:
            nodesInLayer[node.getLayer()] = nodesInLayer[node.getLayer()] + 1

        for frontLayerNumber in range(self.layers - 1):
            nodesInFront = 0
            for layerNumber in range(frontLayerNumber + 1, self.layers):
                nodesInFront = nodesInFront + nodesInLayer[layerNumber]
            maxConnections = maxConnections + \
                nodesInLayer[frontLayerNumber] * nodesInFront

        if maxConnections == len(self.genes):
            return True

        return False

    def mutate(self, history):
        if len(self.genes) == 0:
            self.addConnection(history)

        rand1 = random.random()
        if rand1 < 0.8:  # 80% mutate weigths
            for gene in self.genes:
                gene.mutateWeight()

        rand2 = random.random()
        if rand2 < 0.08:  # 8% add connection
            self.addConnection(history)

        rand3 = random.random()
        if rand3 < 0.02:  # 2% add node
            self.addNode(history)

    def crossover(self, parent2):
        child = Genome(self.inputs, self.outputs, True)
        child.genes = []
        child.nodes = []
        child.layers = self.layers
        child.nextNode = self.nextNode
        child.biasNode = self.biasNode

        childGenes = []
        isEnabled = []

        for gene in self.genes:
            setEnabled = True  # connection is going to be enabled
            parent2gene = self.matchingGene(parent2, gene.innovationNr)
            if not parent2gene == -1:  # genes match
                # if both parent have this gene disabled
                if (not gene.isEnabled()) and (not parent2.genes[parent2gene].isEnabled()):
                    # 90% chance of disabling gene for child
                    if random.random() < 0.9:
                        setEnabled = False
                # if either parent has this gene disabled
                elif (not gene.isEnabled()) or (not parent2.genes[parent2gene].isEnabled()):
                    # 75% chance of disabling gene for child
                    if random.random() < 0.75:
                        setEnabled = False

                if random.random() < 0.5:  # get gene from first parten (this)
                    childGenes.append(gene)
                else:  # get gene from parent2
                    childGenes.append(parent2.genes[parent2gene])
            else:  # disjoint or excess
                childGenes.append(gene)
                setEnabled = gene.isEnabled()

            isEnabled.append(setEnabled)

        # all excess genes are from the one with the higher fit (this one)
        # so all nodes can be inherited from this parent
        for node in self.nodes:
            child.nodes.appen(node.clone())

        # clone all connections before applying them no child nodes
        for n, childgene in enumerate(childGenes):
            child.genes.append(childgene.clone(child.getNode(
                childgene.fromNode.number), child.getNode(childgene.toNode.number)))
            child.genes[n].setEnabled(isEnabled[n])

        child.connectNodes()
        return child

    def matchingGene(self, parent2, innovationNumber):
        for n, gene in eunmerate(parent2.genes):
            if gene.innovationNr == innovationNumber:
                return n

        return -1  # no matching gene

    def clone(self):
        clone = Genome(self.inputs, self.outputs, True)

        # copy nodes
        for node in self.nodes:
            clone.nodes.append(node.clone())

        # copy connections
        for gene in self.genes:
            clone.genes.append(gene.clone(clone.getNode(
                gene.fromNode.number), clone.getNode(gene.toNode.number)))

        clone.layers = self.layers  # copy?
        clone.nextNode = self.nextNode
        clone.biasNode = self.biasNode
        clone.connectNodes()

        return clone

    def drawGenome(self, surface, startX, startY, width, height):
        allNodes = []
        nodePositions = []
        nodeNumbers = []

        for layerNumber in range(self.layers):  # each layer individually
            temp = []
            for node in self.nodes:  # iterate through all nodes
                if node.getLayer() == layerNumber:  # if node is in layer
                    temp.append(node)
            allNodes.append(temp)

        for layerNumber in range(self.layers):  # each layer individually
            color = (255, 0, 0)
            x = startX + (layerNumber * width) / \
                (self.layers - 1)  # position of layer

            # position of node in layer
            for n, node in enumerate(allNodes[layerNumber]):
                y = startY + ((n + 1) * height) / \
                    (len(allNodes[layerNumber]) + 1)

                nodePositions.append((round(x), round(y)))
                nodeNumbers.append(node.number)

        # draw connections
        linewidth = 3
        linecolor = (0, 0, 0)
        for gene in self.genes:
            fromPosition = nodePositions[nodeNumbers.index(
                gene.fromNode.number)]
            toPosition = nodePositions[nodeNumbers.index(gene.toNode.number)]

            if gene.isEnabled():
                if gene.weight < 0:
                    linewidth = 5
                    linecolor = (255, 0, 0)  # red line is positive
                else:
                    linewidth = 5
                    linecolor = (0, 0, 255)  # blue line is negative
            else:
                linewidth = 3
                linecolor = (200, 200, 200)  # gray is deactive

            pygame.draw.line(surface, linecolor, fromPosition,
                             toPosition, linewidth)

        # draw nodes on top of connections
        textcolor = (0, 0, 0)
        nodecolor = (0, 255, 0)
        radius = 20
        for n, position in enumerate(nodePositions):
            pygame.draw.circle(surface, nodecolor, position, radius)
            textsurface = myfont.render(f"{nodeNumbers[n]}", False, textcolor)
            surface.blit(textsurface, position)


class Node:
    def __init__(self, number):
        self.number = number
        self.layer = 0
        self.inputSum = 0  # current sum before activation
        self.outputValue = 0  # after activation function is applied
        self.outputConnections = []

        self.function = "sigmoid"

        self.drawPosition = (0, 0)

    def setLayer(self, layer):
        self.layer = layer

    def getLayer(self):
        return self.layer

    # send output to connected nodes
    def engage(self):
        if not self.layer == 0:  # no sigmaid for input and bias
            self.outputValue = self.activationFunction()

        for connection in self.outputConnections:
            if connection.isEnabled():
                connection.toNode.inputSum = connection.toNode.inputSum + \
                    connection.weight * self.outputValue

    # activation function(s)
    def activationFunction(self):
        if self.function == "sigmoid":
            return 1 / (1 + math.pow(math.e, -4.9 * self.inputSum))

        elif self.function == "step":
            if self.inputSum < 0:
                return 0
            else:
                return 1

    def isConnectedTo(self, testNode):
        if self.layer == testNode.getLayer():
            return False  # cant be connected on same layer

        if testNode.getLayer() < self.layer:  # if test layer comes before
            for connection in testNode.outputConnections:  # check if this is in targets
                # if this fails, check for node id (number)
                if connection.toNode == self:
                    return True
        else:
            for connection in self.outputConnections:  # check if test is target of this
                # if this fails, check for node id (number)
                if connection.toNode == testNode:
                    return True

        return False

    def clone(self):
        clone = Node(self.number)
        clone.setLayer(self.layer)
        return clone


class connectionGene:
    def __init__(self, fromNode, toNode, weight, innovation):
        self.fromNode = fromNode
        self.toNode = toNode
        self.weight = weight
        self.enabled = True
        self.innovationNr = innovation

    def mutateWeight(self):
        rand2 = random.random()

        if rand2 < 0.1:  # 10% complete change
            self.weight = random.random() * 2 - 1
        else:
            self.weight = self.weight + random.gauss(0, 1) / 50

        # limit weight
        if self.weight > 1:
            self.weight = 1
        elif self.weight < -1:
            self.weight = -1

    def isEnabled(self):
        return self.enabled

    def setEnabled(self, enabled):
        self.enabled = enabled

    def clone(self, fromNode, toNode):
        clone = connectionGene(
            fromNode, toNode, self.weight, self.innovationNr)
        clone.setEnabled(self.enabled)
        return clone


class connectionHistory:
    def __init__(self, fromNode, toNode, inno, innovationNos):
        self.fromNode = fromNode
        self.toNode = toNode
        self.innovationNumber = inno
        self.innovationNumbers = innovationNos.copy()  # or is it copy?

    # whether genome matches original genome and connection is between the same nodes
    def matches(self, genome, fromNode, toNode):
        # if the number of connections are different then the genoemes aren't the same
        if len(genome.genes) == len(self.innovationNumbers):
            if fromNode.number == self.fromNode and toNode.number == self.toNode:
                # check if all innovation numbers match from the genome
                for gene in genome.genes:
                    if not innovationNumbers.contains(gene.innovationNr):
                        return False

                # if reached this far then the innovationNumbers match the genes innovation numbers and the connection is between the same nodes
                # so it does match
                return True

        return False


if __name__ == "__main__":
    print("Running as main")

    net = Network(7, 3)
    innovationHistory = []
    net.genome.generateNetwork()
    for _ in range(100):
        net.genome.mutate(innovationHistory)

    while not GAME_QUIT:

        GAMEDISPLAY.blit(BACKGROUND, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                GAME_QUIT = True
            else:
                pass
                # print(event)

        net.genome.drawGenome(GAMEDISPLAY, 100, 0,
                              DISPLAY_WIDTH - 200, DISPLAY_HEIGHT)

        pygame.display.update()
        CLOCK.tick(60)

    pygame.quit()
    quit()
