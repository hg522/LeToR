
# coding: utf-8


from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt



maxAcc = 0.0
maxIter = 0
C_Lambda = 0.02
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 95
PHI = []
IsSynthetic = False


'''Here Querylevelnorm_t file is read and a 1D list is generated containing all
the target values
'''
def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return t

'''
Here the Querylevelnorm_x file is read which contains all the inputs and these
inputs are then stored in a 2D list. The features in the columns [5,6,7,8,9] are 
zero and hence don't contribute anything. Therefore these are removed and
the transpose of the matrix is taken. Dimension : 41 x 69623
'''
def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   
    
    if IsSynthetic == False :
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)
    dataMatrix = np.transpose(dataMatrix)     
    #print ("Data Matrix Generated..")
    return dataMatrix

'''
Here 80% of the rawTargets read are taken in the training target data set
'''
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

'''
Here 80% of the rawData read is taken as the training data set
'''
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

'''
Here validation data set is created as 10% of the initial rawData data set
'''
def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

'''
Here validation data set is created as 10% of the initial rawData data set
'''
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

'''
Since we are using gaussian radial basis function, bigSigma or the variances need 
to be calculated. It is assumed that the variance between features is 0. A diagnol 
matrix is generated to simply matrix multiplication
'''
def GenerateBigSigma(Data, TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))   #generate 41x41 matrix 
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))      #69k x 80%   
    varVect     = []
    for i in range(0,len(DataT[0])):    
        vct = []
        for j in range(0,int(TrainingLen)):     
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))             
    
    for j in range(len(Data)):  
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

'''
Here we are calculating the value of the basis function using the formula for
Gaussian radial basis function
'''
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

''' here we are calcultaing the gaussian radial basis function matrix
of dimension : 55699 x 10. we have the mu matrix containing center points for 10 clusters.
we generate Phi matrix by iterating over the whole data by number of clusters  and 
calculating the radial basis function for each row taking the Mu of the first cluster.
This 1 iteration becomes the column of the Phi matrix
For each input M number of radial basis function are generated like this
'''
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

''' using w = (lambda  * I + phi_T*phi)INV * t
weight matrix dimension depends on the cluster size.
we apply the Moore-Penrose pseudo-inverse of the PHI matrix to calulate the weights
'''
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W


'''
Here we compute the value of the output using linear regression equation
y = w*(PHI(Transpose))
'''
def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

'''
Root mean square error is calculated as the basis of accuracy between the output
generated and the targets provided initially
'''
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    counter = 0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


def getRange(start,end,step):
    ls = []
    i = start
    while i < end:
        k = round(i, 2)
        ls.append(k)
        i+=step
    return ls
# ## Fetch and Prepare Dataset


RawTarget = GetTargetVector('Querylevelnorm_t.csv')
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)
print("Raw Target Shape:",len(RawTarget))
print("Raw Data Shape:",len(RawData),len(RawData[0]))

# ## Prepare Training Data


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print("Training Target Shape:",TrainingTarget.shape)
print("Training Data Shape:",TrainingData.shape)


# ## Prepare Validation Data


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print("Validation Target Shape:",ValDataAct.shape)
print("Validation Data Shape:",ValData.shape)


# ## Prepare Test Data


TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print("Test Target Shape:",ValDataAct.shape)
print("Test Data Shape:",ValData.shape)


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]


ErmsCSTrainArr = []
ErmsCSValArr = []
ErmsCSTestArr = []
ErmsSGDTrainArr = []
ErmsSGDValArr = []
ErmsSGDTestArr = []
AccuracyArr = []
BigSigma     = GenerateBigSigma(RawData, TrainingPercent,IsSynthetic)

#for C_Lambda in lambdaRange:
'''
The data is divided here into M number of clusters and the matrix containing 
the center or mean of each cluster is calculated. The dimension if the Mu matrix
comes out to be M x (number of features)
'''

kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_

TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


print("Mu Matrix Shape:",Mu.shape)
print("Big Sigma Shape:",BigSigma.shape)
print("Training PHI Matrix Shape:",TRAINING_PHI.shape)
print("Weight Matrix Shape:",W.shape)
print("Validation PHI Matrix Shape:",VAL_PHI.shape)
print("Test PHI Matrix Shape:",TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


print ('\nUBITname      = hgarg')
print ('Person Number = 50292195')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')

print ("M = ",M)
print ("Lamba = ",C_Lambda)
#print ("M = 10 \nLambda = 0.9")

print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
ErmsCSTrainArr.append(float(TrainingAccuracy.split(',')[1]))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
ErmsCSValArr.append(float(ValidationAccuracy.split(',')[1]))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
ErmsCSTestArr.append(float(TestAccuracy.split(',')[1]))

'''
plt.plot(lambdaRange, ErmsCSTrainArr) 
plt.xlabel('Lamda') 
plt.ylabel('ERMS Training set') 
plt.title('Plot of Lamda vs. ERMS for Training set where M = 10') 
plt.show() 

plt.plot(lambdaRange, ErmsCSValArr) 
plt.xlabel('Lamda') 
plt.ylabel('ERMS Validation set') 
plt.title('Plot of Lamda vs. ERMS for Validation set where M = 10') 
plt.show() 

    
plt.plot(m, ErmsCSTestArr) 
plt.xlabel('Cluster Size (M)') 
plt.ylabel('ERMS Test set') 
plt.title('Plot of M vs. ERMS for Test set where lambda = 0.01') 
plt.show()
'''
    
    # ## Gradient Descent solution for Linear Regression
    
    
print ('----------------------------------------------------')
print ('--------------Please Wait for 10 mins!----------------')
print ('----------------------------------------------------')
#m = getRange(10,110,10)

M = 60
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_

TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)
W_Now        = np.dot(220, W)
La           = 2
learningRate = 0.01

#lmbda = getRange(1,12,2)
#lr = getRange(0.01,0.2,0.02)

#for learningRate in lr:
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []


for i in range(0,400):
    '''
    The change in W or weight is calculated for each iteration
    by using Delta_w = neta * Delta_E and Delta_E is the derivative 
    of the error function which containes the regularizer term 
    lambda
    '''
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


print ('----------Gradient Descent Solution--------------------')
print ("M = ",str(M))
print("Lambda = ",str(La))
print("neta = ",str(learningRate))
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
ErmsSGDTestArr.append(np.around(min(L_Erms_Test),5))
'''
plt.plot(lr, ErmsSGDTestArr) 
plt.xlabel('Learning rate') 
plt.ylabel('Test set ERMS') 
plt.title('SGD Plot of Learning rate vs. ERMS for Test set where M = 10 and lambda = 2') 
plt.show()
'''