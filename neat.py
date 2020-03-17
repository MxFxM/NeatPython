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
# todo: same with coefficients from species constructor?
# todo: same with staleness maximum in population killStaleSpecies
# todo: add mass extinction reason in population natural selection


class Species:
    def __init__(self, instance=None):
        self.instances = []
        self.bestFitness = 0
        self.champ = None
        self.averageFitness = 0
        self.staleness = 0  # generations without improvement
        self.rep = None

        self.excessCoefficient = 1
        self.weightDifferenceCoefficient = 0.5
        self.compatibilityThreshold = 3

        if not instance is None:
            self.instances.append(instance)
            # since this is the only instance
            self.bestFitness = instance.fitness
            self.rep = instance.genome.clone()
            self.champ = instance.clone()

    def sameSpecies(self, genome):
        compatability = 0
        ecxessAndDisjoint = self.getExcessDisjoint(genome, self.rep)
        averageWeightDifference = self.averageWeightDifference(
            genome, self.rep)

        largeGenomeNormalizer = len(genome.genes) - 20
        if largeGenomeNormalizer < 1:
            largeGenomeNormalizer = 1

        compatability = (self.excessCoefficient * ecxessAndDisjoint / largeGenomeNormalizer) + \
            (self.weightDifferenceCoefficient * averageWeightDifference)

        if self.compatibilityThreshold > compatability:
            return True
        return False

    def addToSpecies(self, instance):
        self.instances.append(instance)

    def getExcessDisjoint(self, genome1, genome2):
        matching = 0
        for gene1 in genome1.genes:
            for gene2 in genome2.genes:
                if gene1.innovationNr == gene2.innovationNr:
                    matching = matching + 1
                    break

        # number of excess and disjoint
        return (len(genome1.genes) + len(genome2.genes)) - 2 * matching

    def averageWeightDifference(self, genome1, genome2):
        if len(genome1.genes) == 0 or len(genome2.genes) == 0:
            return 0

        matching = 0
        totalDifference = 0

        for gene1 in genome1.genes:
            for gene2 in genome2.genes:
                if gene1.innovationNr == gene2.innovationNr:
                    matching = matching + 1
                    totalDifference = totalDifference + \
                        abs(gene1.weight - gene2.weight)
                    break

        if matching == 0:
            return 100  # else divide by 0

        return totalDifference / matching

    def sortSpecies(self):
        temp = []

        length = len(self.instances)
        while not len(self.instances) == 0:
            maxFitness = 0
            maxIndex = 0
            for n, instance in enumerate(self.instances):
                if instance.fitness > maxFitness:
                    maxFitness = instance.fitness
                    maxIndex = n
            temp.append(self.instances[maxIndex])
            self.instances.pop(maxIndex)

        self.instances = temp.copy()  # does copy copy the list?
        if len(self.instances) == 0:
            print("No more instances left?!")
            self.staleness = 200
            return

        # new best player?
        if self.instances[0].fitness > self.bestFitness:
            self.staleness = 0
            self.bestFitness = self.instances[0].fitness
            self.rep = self.instances[0].genome.clone()
            self.champ = self.instances[0].clone()
        else:
            self.staleness = self.staleness + 1

    def setAverage(self):
        sum = 0
        for instance in self.instances:
            sum = sum + instance.fitness

        self.averageFitness = sum / len(self.instances)

    def giveMeBaby(self, history):
        if random.random() < 0.25:  # 25% chance for clone
            child = self.selectInstance().clone()
            child.genome.mutate(history)
            return child
        else:  # 75% chance for crossover
            pass
        # still is else
        parent1 = self.selectInstance()
        parent2 = self.selectInstance()

        # order by fitness
        if parent1.fitness < parent2.fitness:
            child = parent2.crossover(parent1)
            child.genome.mutate(history)
            return child
        else:
            pass
        child = parent1.crossover(parent2)
        child.genome.mutate(history)
        return child

    def selectInstance(self):
        fitnessSum = 0
        for instance in self.instances:
            fitnessSum = fitnessSum + instance.fitness

        rand = random.random() * fitnessSum
        runningSum = 0

        for instance in self.instances:
            runningSum = runningSum + instance.fitness
            if runningSum > rand:
                return instance

        return self.instances[0]

    def cull(self):
        if len(self.instances) > 2:
            self.instances = self.instances[:math.floor(
                len(self.instances) / 2)]

    def fitnessSharing(self):
        for instance in self.instances:
            instance.fitness = instance.fitness / len(self.instances)


class Population:
    def __init__(self, instancesPerGeneration, numberOfInputs, numberOfOutputs):
        self.innovationHistory = []
        self.population = []

        self.bestInstance = None
        self.bestScore = 0
        self.generation = 0

        self.generationPlayers = []
        self.species = []

        for n in range(instancesPerGeneration):
            self.population.append(NeatInstance(
                numberOfInputs, numberOfOutputs))
            self.population[-1].genome.generateNetwork()
            self.population[-1].genome.mutate(self.innovationHistory)
        self.instancesPerGeneration = instancesPerGeneration

    def updateActive(self):
        for instance in self.population:
            if instance.isActive():
                instance.getInputs()
                instance.calculate()
                instance.applyOutputs()
                instance.show()  # optional

    def done(self):
        for instance in self.population:
            if instance.isActive():
                return False
        return True

    def setBestInstance(self):
        tempBest = self.species[0].instances[0]
        tempBest.generation = self.generation

        if tempBest.score > this.bestScore:
            self.generationPlayers.append(tempBest.clone())
            print(f"Old best: {self.bestScore}, new best: {tempBest.score}")
            self.bestScore = tempBest.score
            self.bestInstance = tempBest.clone()

    def naturalSelection(self):
        self.speciate()
        self.calculateFitness()
        self.sortSpecies()

        if False:  # mass extinction
            self.massExtinction()

        self.cullSpecies()
        self.setBestInstance()
        self.killStaleSpecies()
        self.killBadSpecies()

        print(
            f"Generation: {self.generation}, Number of mutations: {len(self.innovationHistory)}, Species: {len(self.species)} <<<<<<<<<<<<<<<<<<<<<<<")

        averageSum = self.getAverageFitnessSum()
        children = []
        for species in self.species:
            children.append(species.champ.clone())

            # -1 because champ is already added
            numberOfChildren = math.floor(
                species.averageFitness / averageSum * len(self.population)) - 1
            for _ in range(numberOfChildren):
                children.append(species.giveMeBaby(self.innovationHistory))

        while len(children) < len(self.population):
            children.append(self.species[0].giveMeBaby())

        self.population = []
        self.population = children.copy()  # does this copy the list?

        self.generation = self.generation + 1

        for instance in self.population:
            instance.genome.generateNetwork()

    def speciate(self):
        for speices in self.species:
            species.instances = []

        for instance in self.population:
            speciesFound = False
            for species in self.species:
                if species.sameSpecies(instance.genome):
                    species.addToSpecies(instance.genome)
                    speciesFound = True
                    break
            if not speciesFound:
                self.species.append(Species(instance))

    def calculateFitness(self):
        for instance in self.population:
            instance.calculateFitness()

    def sortSpecies(self):
        for species in self.species:
            species.sortSpecies()

        temp = []
        length = len(self.species)
        while not len(self.species) == 0:
            maxFitness = 0
            maxIndex = 0
            for n, species in enumerate(self.species):
                if species.bestFitness > maxFitness:
                    maxFitness = species.bestFitness
                    maxIndex = n
            temp.append(self.species[maxIndex])
            self.species.pop(maxIndex)

        self.species = temp.copy()  # does copy copy the list?

    def killStaleSpecies(self):
        # does this work?
        # but i get the idea
        for n, species in enumerate(self.species.reverse()):
            if species.staleness >= 15:
                self.species.pop(n)

    def killBadSpecies(self):
        averageSum = self.getAverageFitnessSum()

        for n, species in enumerate(self.species.reverse()):
            if species.averageFitness / averageSum * len(self.population) < 1:
                self.species.pop(n)  # bye

    def getAverageFitnessSum(self):
        averageSum = 0
        for species in self.species:
            averageSum = averageSum + species.averageFitness()
        return averageSum

    def cullSpecies(self):
        for species in self.species:
            species.cull()
            species.fitnessSharing()
            species.setAverage()

    def massExtinction(self):
        for n, species in enumerate(self.species.reverse()):
            if n >= 5:
                self.species.pop(n)


class NeatInstance:
    def __init__(self, numberOfInputs, numberOfOutputs):
        self.fitness = 0
        self.score = 0
        self.generation = 0
        self.active = True

        self.vision = [0 for _ in range(numberOfInputs)]  # input values
        self.decision = [0 for _ in range(
            numberOfOutputs)]  # output of network

        self.genomeInputs = numberOfInputs
        self.genomeOutputs = numberOfOutputs

        self.genome = Genome(numberOfInputs, numberOfOutputs)

    def show(self):
        # draw game on sceen?
        pass

    def incereaseScore(self):
        self.score = self.score + 1

    def isActive(self):
        return self.active

    def getInputs(self):
        # get inputs from game?
        # write inputs in self.vision
        pass

    def calculate(self):
        self.decision = self.genome.feedForward(self.vision)

    def applyOutputs(self):
        # set outputs and step in game?
        # use self.decision for that

        # this might help
        # maxValue = 0
        # maxIndex = 0
        # for n, output in enumerate(self.decision):
        #     if output > maxValue:
        #         maxValue = output
        #         maxIndex = n
        pass

    def clone(self):
        clone = NeatInstance(self.genomeInputs, self.genomeOutputs)
        clone.genome = self.genome.clone()
        clone.fitness = self.fitness
        clone.genome.generateNetwork()
        clone.generation = self.generation
        return clone

    def calculateFitness(self):
        self.fitness = self.score * self.score

    def crossover(self, parent2):
        child = NeatInstance(self.genomeInputs, self.genomeOutputs)
        child.genome = self.genome.crossover(parent2.genome)
        child.genome.generateNetwork()
        return child


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
        for n in range(self.inputs):
            self.nodes[n].outputValue = inputValues[n]
        self.nodes[self.biasNode].outputValue = 1

        for net in self.network:
            net.engage()

        outs = [0 for _ in range(self.outputs)]
        for n in range(self.outputs):
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

    pop = Population(10, 7, 3)

    while not GAME_QUIT:

        GAMEDISPLAY.blit(BACKGROUND, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                GAME_QUIT = True
            else:
                pass
                # print(event)

        if not pop.done():
            pop.updateActive()  # play
        else:
            pop.naturalSelection()  # genetic algorithm

        pygame.display.update()
        CLOCK.tick(60)

    pygame.quit()
    quit()
