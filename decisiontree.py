# Chung Ho (Tony) Lee, netid: chl433

#import all packages needed
import csv
import random
import math
import sys
import copy

#get the Input Arguments
inputFileName=sys.argv[1]
trainingSetSizeInput=sys.argv[2]
numberOfTrials=int(sys.argv[3])
verbose=int(sys.argv[4])

#create the class for decisionTree
class decisionTree:
    def __init__(self, attribute):
        self.attribute=attribute
        self.trueChild="leaf"
        self.falseChild="leaf"

#initialize all important lists
titleList=[] #contains the list of attributes
dataList=[] #contains the list of dictionaries of data
classification=[] #contains the classification for data
#reads the data from the input file
with open(inputFileName,'rb') as csvfile:
    fileReader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    firstBool=True
    for row in fileReader: #Go through each row
        if firstBool: #First row is the title row
            for i in range(0,len(row)-1): #iterate through the current row
                titleList.append(row[i])
            firstBool=False
        else: #Rest of the rows are data to be trained or tested
            dataPoint={}
            for i in range(0,len(row)-1): #iterate through the current row
                boolean=False
                if row[i]=="true": #turns the string into a boolean
                    boolean=True
                dataPoint[titleList[i]]=boolean
            boolean2=False
            if row[len(row)-1]=="true": #turns the string into a boolean
                boolean2=True
            classification.append(boolean2) #appends data to classification list
            dataList.append(dataPoint) #appends data to the dataList list

trainingSetSize=int(trainingSetSizeInput) #sets the training size according to input

#Checks if all elements in the list is equal
def checkEqual(lst): 
    element=lst[0]
    for item in lst:
        if item!=element:
            return False
    return True

#Checks to find the mode for the list
def findMode(lst):
    trueCount=0
    falseCount=0
    for item in lst:
        if item==True:
            trueCount+=1
        else:
            falseCount+=1
    return trueCount>=falseCount

#Main algorithm for ID3
def DTL(trainingExamples, trainingClassification,attributes, default):
    if len(trainingExamples)==0: #return default if trainingExamples is empty
        return default
    elif checkEqual(trainingClassification): #return data if the entire list is the same
        return trainingClassification[0]
    elif len(attributes)==0: #return mode if no attributes left
        return findMode(trainingClassification)
    else:
        best=chooseAtt(trainingExamples, trainingClassification, attributes) #chooses an attribute
        tree=decisionTree(best)
        #Initialize data
        trueExamples=[]
        trueClassification=[]
        falseExamples=[]
        falseClassification=[]
        for i in range(0,len(trainingExamples)): #Goes through all the training examples
            if (trainingExamples[i])[best]==True: #If training example is true appends to true
                trueExamples.append(trainingExamples[i])
                trueClassification.append(trainingClassification[i])
            else: #if training example is false in the attribute chosen then append to false
                falseExamples.append(trainingExamples[i])
                falseClassification.append(trainingClassification[i])
        tempAttribute=copy.deepcopy(attributes) #deepcopy the attribute list
        tempAttribute.remove(best) #remove the attribute already chosen
        subtree1=DTL(trueExamples, trueClassification, tempAttribute, findMode(trainingClassification)) #call DTL again
        tree.trueChild=subtree1 #subtree becomes trueChild
        subtree2=DTL(falseExamples, falseClassification, tempAttribute, findMode(trainingClassification)) #call DTL again
        tree.falseChild=subtree2 #subtree becomes falseChild
        return tree #return resulting tree

#Chooses an attribute for the next branch
def chooseAtt(trainingExamples, trainingClassification, attributes):
    entropyDict={}
    for att in attributes: #Checks the entropy of all the attributes
        lstTrue=[]
        lstFalse=[]
        for i in range(0,len(trainingExamples)): #Goes through the trainingExamples to see if attribute is true or false
            if (trainingExamples[i])[att]==True:
                lstTrue.append(i)
            else:
                lstFalse.append(i)
        #initialize variables
        entropy=0
        countPos=0
        countNeg=0
        for index in lstTrue: #Counts the number that are positive or negative in lstTrue for the attribute
            if trainingClassification[index]==True:
                countPos+=1
            else:
                countNeg+=1
        #initialize frac to 1 so log will result in 0 if not replaced
        frac1=1
        frac2=1
        #If there is data in lstTrue, adjust fractions
        if len(lstTrue)>0 and countPos>0 and countNeg>0:
            frac1=countPos*1.0/len(lstTrue)
            frac2=countNeg*1.0/len(lstTrue)
        #Calculate the entropy
        wValue=1.0*len(lstTrue)/len(trainingExamples)
        entropy=entropy+wValue*(-1.0*math.log(frac1)/math.log(2.0)-1.0*math.log(frac2)/math.log(2.0))

        #initialize variables
        countPos=0
        countNeg=0
        for index in lstFalse: #Counts the number that are positive or negative in lstFalse for the attribute
            if classification[index]==True:
                countPos+=1
            else:
                countNeg+=1
        #initialize frac to 1 so log will result in 0 if not replaced
        frac1=1
        frac2=1
        if len(lstFalse)>0 and countPos>0 and countNeg>0:
            frac1=countPos*1.0/len(lstFalse)
            frac2=countNeg*1.0/len(lstFalse)
        #calculate entropy
        wValue=1.0*len(lstFalse)/len(trainingExamples)
        entropy=entropy+wValue*(-1.0*math.log(frac1)/math.log(2.0)-1.0*math.log(frac2)/math.log(2.0))

        #add entropy for current attribute to dictionary
        entropyDict[att]=entropy
    # The follow returns the attribute with the lowest entropy from the dictionary
    return min(entropyDict,key=entropyDict.get)

#prints the tree in a readable format
def printTree(tree):
    print "Attribute: "+tree.attribute
    if isinstance(tree.trueChild,decisionTree): #if trueChild is a subtree
        print "trueChild: "+tree.trueChild.attribute
    else:
        print "trueChild: "+str(tree.trueChild)
    if isinstance(tree.falseChild,decisionTree): #if falseChild is a subtree
        print "falseChild: "+tree.falseChild.attribute
    else:
        print "falseChild: "+str(tree.falseChild)
    if isinstance(tree.trueChild,decisionTree): #if trueChild is a subtree
        print "---------------"
        printTree(tree.trueChild)
    if isinstance(tree.falseChild,decisionTree): #if falseChild is a subtree
        print "---------------"
        printTree(tree.falseChild)

#classifies all data based on the decision tree generated
def classifyAll(testingExamples,decTree):
    resultLst=[]
    for dict in testingExamples:
        resultLst.append(classifyExample(dict,decTree)) #Classify one data
    return resultLst

#classifies one data based on the decision tree generated
def classifyExample(dict, classifyTree):
    if classifyTree==True: #return True
        return True
    elif classifyTree==False: #return False
        return False
    if dict[classifyTree.attribute]==True: #move down on the tree to trueChild
        return classifyExample(dict,classifyTree.trueChild)
    elif dict[classifyTree.attribute]==False: #move down on the tree to falseChild
        return classifyExample(dict,classifyTree.falseChild)

#Classifies all according to the prior probability
def priorClassifier(testingExamples,priorTrue,priorFalse):
    if priorTrue>=priorFalse:
        return [True]*len(testingExamples)
    else:
        return [False]*len(testingExamples)

treeAccuracy=[]
priorAccuracy=[]

#The main function to be executed
def mainFunction():
    #randomize indices for training and testing data
    trainingIndex=random.sample(range(0,len(dataList)),trainingSetSize)
    testingIndex=set(range(0,len(dataList)))-set(trainingIndex)

    #sets trainingExamples based on random index
    trainingExamples=[dataList[i] for i in trainingIndex]
    if verbose==1: #display additional data for verbose
        print "Examples in the training set: "
        print trainingExamples
        print ""
    trainingClassification=[classification[i] for i in trainingIndex]
    #sets testingExamples based on random index
    testingExamples=[dataList[i] for i in testingIndex]
    if verbose==1: #display additional data for verbose
        print "Examples in the testing set: "
        print testingExamples
        print ""
    testingClassification=[classification[i] for i in testingIndex]

    #number in trainingExamples that's true
    numTrueInTraining=0
    for i in range(0,len(trainingClassification)):
        if trainingClassification[i]==True:
            numTrueInTraining+=1
    #calculate prior probability
    priorTrue=numTrueInTraining*1.0/len(trainingClassification)
    priorFalse=1-priorTrue

    #Make the decision tree by calling DTL
    decTree=DTL(trainingExamples, trainingClassification, titleList, True)
    print "TREE STRUCTURE"
    printTree(decTree)

    #Calculate the proportion of correct classifications made in steps 5 and 6
    priorList=priorClassifier(testingExamples,priorTrue,priorFalse)
    if verbose==1: #display additional data for verbose
        print ""
        print "Classification returned by decision tree: "
        print priorList
        print ""
    ID3List=classifyAll(testingExamples,decTree)
    if verbose==1: #display additional data for verbose
        print "Classification returned by prior probability: "
        print ID3List
        print ""

    #Calculate the percentage correct using ID3 and prior probability
    priorCorrect=0
    ID3Correct=0
    for i in range(0,len(testingClassification)):
        if priorList[i]==testingClassification[i]:
            priorCorrect+=1
        if ID3List[i]==testingClassification[i]:
            ID3Correct+=1
    #append probability to list to take average later
    treeAccuracy.append(ID3Correct*100.0/len(testingClassification))
    priorAccuracy.append(priorCorrect*100.0/len(testingClassification))
    print "Percent of test cases correctly classified by a decision tree built with ID3: "+ str(ID3Correct*100.0/len(testingClassification))+"%"
    print "Percent of test cases correctly classified by using prior probabilities from training set: "+ str(priorCorrect*100.0/len(testingClassification))+"%"
#repeat main function a number of times based on numberOfTrials
for i in range(0,numberOfTrials):
    print ""
    print "Trial "+str(i+1)
    print "--------------------------------------"
    mainFunction()
#Prints the final data
print ""
print "example file used = "+inputFileName
print "number of trials = "+str(numberOfTrials)
print "training set size for each trial = "+trainingSetSizeInput
testSize=len(dataList)-int(trainingSetSizeInput)
print "testing set size for each trial = "+str(testSize)
#calculate the average accuracy for ID3 and prior probability
percentage1=sum(treeAccuracy)*1.0/len(treeAccuracy)
percentage2=sum(priorAccuracy)*1.0/len(priorAccuracy)
print "mean performance of decision tree over all trials = " +str(percentage1)+"% correct classification"
print "mean performance of using prior probability derived from the training set = "+str(percentage2)+"% correct classification"
