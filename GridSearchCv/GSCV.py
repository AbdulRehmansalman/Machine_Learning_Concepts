from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()
# todo Second Paramter is Your Parameter Grid
cv = GridSearchCV(SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel': ['linear','rbf' ],
},cv=5,return_train_score=False)

# todo It uses Cross Validation as we are Making the For Loop Code Block Convenient and Writing in OneLine ofCode
cv.fit(iris.data,iris.target)
print(cv.cv_results_) #Print CrossValidatio nResults

# Cv Results are NotEasy to View So Take Put it in Pnadas Datsframe
df = pd.DataFrame(cv.cv_results_)
print(df)

print(cv.best_params_)
print(cv.best_score_)


# computational Cost is a Parameter For Computational Cost If C = 1 to 50 then  as it Will try Every Combination and Permutation For Every Vlaue
# in Each of These Paramters

# Randomised Search Cv Will not Try Every combination and Permutaion of the Paramters But it Will Try Randon Combination of the Parameter Vlaues
# And You Choose Waht These iterations Could be
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(SVC(gamma='auto'),{
     'C':[1,10,20],
     'kernel': ['linear','rbf' ],
},cv=5,return_train_score=False,n_iter=2)

# todo In BackWard WE try 6 combination here we Try 2 Combination
rs.fit(iris.data,iris.target)
df =pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
print(df)

# Now in real Life To Check Which the Model is Gud
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# todo You have to Define Your Parameter Grid as Jsom Obkjject or Python Dicionries: That Svm With These Paramters and Other With This
model_params={
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params':{
            "C":[1,10,20],
            "kernel":['linear','rbf' ],
        }
    },
    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[1,5,10],
        }
    },
    'logistic_regression':{
        'model':LogisticRegression(solver='liblinear' ,multi_class='auto'),
        'params':{
            'C':[1,5,10],  #Paramter in Logistic Regression Classifier
        }
    }
}
# Do the For Loop it Going throg the Dictionarie sVlaues and it Uses GridSearchCv fIRST paramater as Classifier  amd Next param as Paramter in Each Classifier

scores=[]
for model_name, mp in model_params.items:
    clf = GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(iris.data,iris.target)
    scores.append({
        'model':model_name,
        'best_score':clf.best_score_,
        'best Params':clf.best_params_,
    })

# Convert those into Pnadas DatafRame
df =pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)