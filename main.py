

#for batch_X, batch_Y in trainset_loader:
#    print(batch_X)
# How to deal with the small dataset. Cross validation or leave one out?
# Standarization and Normalization: How to normalize for the test set?
# Algorithms
# Imputation
# Will update the dataset
# Neural network designs
# NN + validation + multitask design
# Cross validation vs. Leace one out
# eSOFA and SOFA score
# Neural network is not working at all!!!
  
import pandas as pd
import csv
from statsmodels.imputation.mice import MICE, MICEData
from sklearn.impute import SimpleImputer
import numpy as np

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn  import preprocessing

class ShockDataSet(Dataset):
    def __init__(self, X, Y, normalize=False):
        self.X =  X
        self.Y = Y


        if normalize == True:
            # why not learning with normalizaiton
            X = preprocessing.normalize(X, norm='l2', axis=1)


    def __getitem__(self, index):

        x_to_return = torch.tensor(self.X[index].tolist())


        #if self.transform:
        #    x_to_return = self.transform(x_to_return)

        return ( x_to_return, torch.tensor(self.Y[index].tolist())  )

    def __len__(self):
        return self.Y.shape[0]


def model_performance( y_test, pred, pred_prob = None, title=None ):
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    acc = (tp + tn) / (tn + fp + fn + tp)
    sensitivity_recall = tp / (tp+fn)
    specificity = tn / (tn+fp)
    f1 = 2 * tp / ( 2*tp + fp + fn )


    if title is not None:
        print("### " + title )

    print("Acc: {0:.2f}, Recall/Sensitivity:{1:.2f}, Specificity:{2:.2f}  ".format(acc, sensitivity_recall, specificity))
    print("F1: {0:.2f}, count: {1} / {2} ".format(f1, np.sum(pred==True), len(pred) ))

    if pred_prob is not None:
        auc = roc_auc_score(y_test, pred_prob)
        print("AUC: {0:.2f} ".format(auc))
    else:

        pred_prob = np.random.randn(*pred.shape)
        auc = roc_auc_score(y_test, pred_prob)
        print("AUC (with random prob): {0:.2f} ".format(auc))

    print("\n")

#x_inx_continous = [     6, 7, 8, 9, 10, 11,   13, 14, 15, 16, 17, 18,    20, 21, 22, 23, 24, 25,     27, 28, 29, 30, 31, 32,    34, 35, 36, 37, 38, 39,     41, 42, 43, 44, 45, 46,        49,   51,    53,     55,         58, 59, 60, 61, 62, 63,     65,                                         75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88 ,89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
#x_inx_categorical = [5                  ,  12,                        19,                        26,                         33,                        40,                         47, 48,   50,   52,     54,    56,  57,                         64,      66,  67, 68, 69, 70, 71, 72, 73, 74]
x_col_continuous = ['sbp_mean', 'sbp_median', 'sbp_max', 'sbp_min', 'sbp_variance', 'sbp_count', 'rr_mean', 'rr_median', 'rr_max', 'rr_min', 'rr_variance', 'rr_count', 'temp_mean', 'temp_median', 'temp_max', 'temp_min', 'temp_variance', 'temp_count', 'map_mean', 'map_median', 'map_max', 'map_min', 'map_variance', 'map_count', 'o2sat_mean', 'o2sat_median', 'o2sat_max', 'o2sat_min', 'o2sat_variance', 'o2sat_count', 'hr_mean', 'hr_median', 'hr_max', 'hr_min', 'hr_variance', 'hr_count', 'wbc', 'hgb', 'platelet', 'creatinine', 'lactate_mean', 'lactate_median', 'lactate_max', 'lactate_min', 'lactate_variance', 'lactate_count', 'total_bili', 'count_surgical_cpt', 'count_cardiac_surgery_cpt', 'count_gi_surgery_cpt', 'count_urinary_surgery_cpt', 'count_gu_surgery_cpt', 'count_neuro_surgery_cpt', 'count_resp_surgery_cpt', 'total_abx_count', 'mrsa_target_abx_count', 'pseudomonal_cover_abx', 'pseudomonal_abx_count', 'anaerobe_abx_count', 'amino_abx_count', 'penem_abx_count', 'first_cepha_count', 'third_cepha_count', 'forth_cepha_count', 'macro_count', 'aztreonam_count', 'penicillin_count', 'floxacin_count', 'trimethoprim_count', 'tetra_count', 'azole_count', 'fungin_count', 'ampho_count', 'fungal_count']
x_col_categorical = ['map_exist', 'rr_exist', 'temp_exist', 'map_exist', 'o2sat_exist', 'hr_exist', 'prbc_exist', 'wbc_exist', 'hgb_exist', 'platelet_exist', 'creatinine_exist', 'lactate_exist', 'lactate_exist.1', 'total_bili_exist', 'intubated', 'pressor_use', 'vasopressin_use', 'norepinephrine_use', 'phenylephrine_use', 'dopamine_use', 'dobutamine_use', 'epinephrine_use', 'milrinone_use']

df = pd.read_csv('shock_cases.csv')

df_continous = df.loc[:, x_col_continuous ]
#df_continous = df.iloc[:, x_inx_continous ]
#df_categorical = df.iloc[:, x_inx_categorical]
df_categorical = df.loc[:, x_col_categorical]


headers = list(df_continous.columns.values) + list(df_categorical.columns.values)
headers_continuous = list(df_continous.columns.values)
headers_categorical = list(df_categorical.columns.values)


# Shock
y1 = df.iloc[:, 1].as_matrix()

# Septic
y2 = df.iloc[:, 2].as_matrix()

y2 = np.array([False if v is np.nan else v for v in y2]) #to be removed


# Cardiogenic
y3 = df.iloc[:, 3].as_matrix()
y3 = np.array([False if v is np.nan else v for v in y3]) #to be removed

# Hypovolemic
y4 = df.iloc[:, 4].as_matrix()
y4 = np.array([False if v is np.nan else v for v in y4]) #to be removed



X_continous = df_continous.as_matrix()
X_categorical = df_categorical.as_matrix()

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imp_mean.fit(X_continous)
imp_frequent.fit(X_categorical)

X_continous = imp_mean.transform(X_continous)
X_categorical = imp_frequent.transform(X_categorical)


X = np.concatenate((X_continous, X_categorical), axis=1)

logistic_regression_model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=50000)





# Scaling/Normalization

np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.3)

logistic_regression_model.fit(X_train, y_train)

pred = logistic_regression_model.predict(X_test)
pred_prob = logistic_regression_model.predict_proba(X_test)[:,1]


print("###### Shock vs. Non-Shock: {} Cases.".format( np.sum(y_test) ))
model_performance( y_test, pred, pred_prob= pred_prob, title="Shock vs Non-Shock: Logistic Regression" )
pred = (y_test == 20928208222) # Always False (== not shock)

model_performance( y_test, pred, title="Shock vs Non-Shock: Dumb")

# Lasso Test
lasso = Lasso(alpha=0.02, max_iter=10000)
lasso.fit(X_train, y_train)
selected_features = np.array(headers)[ lasso.coef_ > 0 ]

print( "--Lasso: {0} variables selected out of {1}\n".format( len(selected_features), X_train.shape[1] ))
print( selected_features )
print("\n")

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

gbm = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.08, objective="binary:logistic").fit(X_train, y_train)

pred = gbm.predict(X_test)
pred_prob = gbm.predict_proba(X_test)[:,1]
model_performance( y_test, pred, pred_prob,title="Shock vs Non-Shock: XGBoost")
#print(y_test)
#print(pred)

# Infection

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.3)

logistic_regression_model.fit(X_train, y_train)

pred = logistic_regression_model.predict(X_test)
pred_prob = logistic_regression_model.predict_proba(X_test)[:,1]

print("###### Infection vs. Non-Infection: {} Cases.".format( np.sum(y_test) ))
model_performance( y_test, pred, pred_prob= pred_prob, title="Infection: Logistic Regression" )

#abx_count = df.loc[:, 'total_abx_count' ]
#abx_criteria = (abx_count > 2).as_matrix()
#model_performance( y_test, abx_criteria, title="Infection: N_Abx >=2" )

pred = (y_test == 20928208222) # Always False (== not shock)

model_performance( y_test, pred, title="Infection: Dumb")

# Lasso Test
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_train, y_train)
selected_features = np.array(headers)[ lasso.coef_ > 0 ]

print( "--Lasso: {0} variables selected out of {1}\n".format( len(selected_features), X_train.shape[1] ))
print( selected_features )
print("\n")

gbm = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.08).fit(X_train, y_train)
pred = gbm.predict(X_test)
pred_prob = gbm.predict_proba(X_test)[:,1]
model_performance( y_test, pred, pred_prob,title="XGB - Infection vs Non-Infection: XGBoost")
#xgb.plot_importance(booster=gbm)
#plt.show()

# Cardio

X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.3)

logistic_regression_model.fit(X_train, y_train)

pred = logistic_regression_model.predict(X_test)
pred_prob = logistic_regression_model.predict_proba(X_test)[:,1]


model_performance( y_test, pred, pred_prob= pred_prob, title="Cardiogenic: Logistic Regression" )
pred = (y_test == 20928208222) # Always False (== not shock)

model_performance( y_test, pred, title="Cardiogenic: Dumb")

# Lasso Test
lasso = Lasso(alpha=0.02, max_iter=10000)
lasso.fit(X_train, y_train)
selected_features = np.array(headers)[ lasso.coef_ > 0 ]
print( "--Lasso: {0} variables selected out of {1}\n".format( len(selected_features), X_train.shape[1] ))
print( selected_features )
print("\n")


gbm = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.08).fit(X_train, y_train)
pred = gbm.predict(X_test)
pred_prob = gbm.predict_proba(X_test)[:,1]
model_performance( y_test, pred, pred_prob,title="XGB - Cardiac vs Non-Cardio")


# Hypovolemia

X_train, X_test, y_train, y_test = train_test_split(X, y4, test_size=0.3)

logistic_regression_model.fit(X_train, y_train)

pred = logistic_regression_model.predict(X_test)
pred_prob = logistic_regression_model.predict_proba(X_test)[:,1]


model_performance( y_test, pred, pred_prob= pred_prob, title="Hypovolemic: Logistic Regression" )
pred = (y_test == 20928208222)  # Always False (== not shock)

model_performance( y_test, pred, title="Hypovolemic: Dumb")


# Lasso Test
lasso = Lasso(alpha=0.002, max_iter=10000)
lasso.fit(X_train, y_train)
selected_features = np.array(headers)[ lasso.coef_ > 0 ]
print( "--Lasso: {0} variables selected out of {1}\n".format( len(selected_features), X_train.shape[1] ))
print( selected_features )
print("\n")


gbm = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.5).fit(X_train, y_train)
pred = gbm.predict(X_test)
pred_prob = gbm.predict_proba(X_test)[:,1]
model_performance( y_test, pred, pred_prob,title="XGB - Hypovolemia")



#######################################################################################################################################################
### Neural Network: Multitask Learning
#######################################################################################################################################################




class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNeuralNetwork, self).__init__()

        # Initial Layer
        self.linear = nn.Linear(input_dim, 256)

        # Hidden Layer 1
        self.linear2 = nn.Linear(256, 256)

        # Hidden Layer 2
        self.linear3 = nn.Linear(256, 256)

        # Hidden Layer 3
        self.linear4 = nn.Linear(256, 256)

        # Multitask ReadOut
        self.linear5 = nn.Linear(256, 4)

        # Sigmoid and ReLU
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # DropOut

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)

    # Here's where we connect all the layers together
    def forward(self, x):

        out = self.linear(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.linear4(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.linear5(out)

        out = self.sigmoid(out)

        return out


Y = np.concatenate( ( y1.reshape(-1,1), y2.reshape(-1,1), y3.reshape(-1,1), y4.reshape(-1,1) ), axis=1 )
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


trainset = ShockDataSet(X_train,  Y_train, normalize=True  )
testset = ShockDataSet(X_test, Y_test, normalize=True )

trainset_loader = DataLoader(trainset,batch_size=16,shuffle=True )
test_loader =  DataLoader(testset,batch_size=16,shuffle=True )

input_dim = X_train.shape[1]

model =  FeedforwardNeuralNetwork( input_dim )

learning_rate = 10e-5
optimizer = torch.optim.Adam( model.parameters(), lr= learning_rate, weight_decay=0.01 )
criterion = torch.nn.BCELoss()
#regularizer =

n_epochs = 1000
iter = 0
for epoch in range(n_epochs):
    #print("-- Epoch: {}".format(epoch))
    for batch_X, batch_Y in trainset_loader:

        # omg lol. The performance is better with random input.
        #batch_X = torch.Tensor(np.random.rand( *(batch_X.numpy().shape) ))
        outputs = model(batch_X)
        optimizer.zero_grad()

        out1 = outputs[:,0]
        out2 = outputs[:, 1]
        out3 = outputs[:, 2]
        out4 = outputs[:, 3]

        label1 = batch_Y[:,0]
        label2 = batch_Y[:,1]
        label3 = batch_Y[:,2]
        label4 = batch_Y[:,3]

        loss1 = criterion(out1, label1.float())
        loss2 = criterion(out2, label2.float())
        loss3 = criterion(out3, label3.float())
        loss4 = criterion(out4, label4.float())

        weight1 = 1.0
        weight2 = 1.0
        weight3 = 1.0
        weight4 = 1.0

        loss = weight1*loss1 + weight2*loss2 + weight3*loss3 + weight4*loss4

        loss.backward()
        optimizer.step()
        iter += 1

        if iter % 50 == 0:
            # Calculate Accuracy

            # Training Accuracy/Loss
            correct = 0
            total = 0
            # Iterate through training dataset

            correct = 0
            total = 0

            for training_X, training_Y in trainset_loader:

                output_training = model(training_X)

                out_training_1 = output_training[:, 0]
                label_training_1 = training_Y[:, 0]
                pred_training_1 = out_training_1 > 0.5
                total = total + pred_training_1.size()[0]
                correct = correct + float(torch.sum(pred_training_1 == label_training_1))

            accuracy = float(correct / total)

            # Print Loss
            print('\n# Performance on Training Set - Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

            # Test Accuracy/Loss
            correct = 0
            total = 0

            for test_X, test_Y in test_loader:
                output_test = model(test_X)


                out_test_1 = output_test[:, 0]
                label_test_1 = test_Y[:, 0]
                pred_test_1 = out_test_1 > 0.5
                total = total + pred_test_1.size()[0]
                correct = correct + float(torch.sum(pred_test_1 == label_test_1))

            accuracy = float(correct / total)


            #X_test, Y_test




            # Print Loss
            print('# Performance on Test Set - Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(),
                                                                                                 accuracy))

            '''
            test_set = torch.Tensor(X_test.astype(float))
            logit = model(test_set).detach().numpy()
            output_test = (logit > 0.5)

            model_performance(Y_test.astype(float)[:,0], output_test[:,0], pred_prob=logit[:,0], title="NN - Shock Vs. Non-Shock")
            '''


#output = model( X_test )
#print(output)




'''
X = df.as_matrix()

print(X)

print('------------------------------')
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X)


print( imp_mean.transform(X) )

'''


'''
imp = MICEData(df)


print(imp.conditional_formula)
mice = MICE( imp.conditional_formula, imp.model_class,data=imp )
results = mice.fit(10, 10)

print(results)
print(results.summary())
'''




'''
X = df.as_matrix()

print(X)

print('------------------------------')
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X)


print( imp_mean.transform(X) )

'''


'''
imp = MICEData(df)


print(imp.conditional_formula)
mice = MICE( imp.conditional_formula, imp.model_class,data=imp )
results = mice.fit(10, 10)

print(results)
print(results.summary())
'''



#output = model( X_test )
#print(output)




'''
X = df.as_matrix()

print(X)

print('------------------------------')
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X)


print( imp_mean.transform(X) )

'''


'''
imp = MICEData(df)


print(imp.conditional_formula)
mice = MICE( imp.conditional_formula, imp.model_class,data=imp )
results = mice.fit(10, 10)

print(results)
print(results.summary())
'''




'''
X = df.as_matrix()

print(X)

print('------------------------------')
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X)


print( imp_mean.transform(X) )

'''


'''
imp = MICEData(df)


print(imp.conditional_formula)
mice = MICE( imp.conditional_formula, imp.model_class,data=imp )
results = mice.fit(10, 10)

print(results)
print(results.summary())
'''