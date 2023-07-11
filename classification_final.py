# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  preprocessing
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.formula.api import ols
from sklearn import datasets
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import copy
import pickle
from sklearn.preprocessing import LabelEncoder
import os



# %%
COLS = ['Hotel_Name', 'day', 'month', 'year', 'Review_Date', 'Hotel_Address', 'days_since_review', 'lat', 'lng', 'Positive_Review', 'Negative_Review']

# %%
def save_model(model, filename):
    # Save the model to disk
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    # Load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

# %%
class preprocess:
    def __init__(self , X , Y , data):
        self.X = X
        self.Y = Y
        self.data = data
    
    def remove_row(self,df, lis):
        lis = sorted(set(lis))
        df = df.drop(lis)
        return df
    
    def outliars(self,df, column):
        low, high = df[column].quantile([.015,.985])
        mask_r = df[column].between(low,high)
        df = df[mask_r]
        return df
    
    def fillNull(self):
        self.X.fillna(" ", inplace=True)
        self.X.fillna(0, inplace=True)
        self.Y.fillna(" ", inplace=True)
        return self.X , self.Y

    def dropNull(self):
        self.X.dropna(axis=0, inplace=True)
        self.Y.dropna(axis=0, inplace=True)
        return self.X , self.Y

    def convertDate(self):
        self.X[["day", "month", "year"]] = self.X["Review_Date"].str.split("/", expand = True)
        return self.X
    
    def convert_to_int(self):
        try:
          self.X['Review_Date'] = self.X.Review_Date.astype(np.int64)//10**9
        except:
          pass

    def split_1(self):
        self.X['days_since_review']=self.X['days_since_review'].str.split(' ',expand=True)[0]
        return self.X

    def uniqe1(self):
        self.X['Reviewer_Nationality'].nunique()
        return self.X

    def label_encoder(self):
        #label_encoder = preprocessing.LabelEncoder()
        #self.X['Tags']= label_encoder.fit_transform(self.X['Tags'])
        # self.X['Tags'].unique()
        label_encoder2 = preprocessing.LabelEncoder()
        self.X['Reviewer_Nationality']= label_encoder2.fit_transform(self.X['Reviewer_Nationality'])
        # self.X['Reviewer_Nationality'].unique()
        return self.X

    def address(self):
        self.X["Main_Address"] = self.X["Hotel_Address"].str.split(" ").str[-2:].str.join(' ')
        self.X['Main_Address'] = label_encode_y(self.X["Main_Address"])
        return self.X
    
    def remove_outlairs(self):
        self.X = pd.concat([self.X, self.Y], axis=1)
        # , 'Reviewer_Score'
        # 'Review_Total_Positive_Word_Counts', 'Review_Total_Negative_Word_Counts', 'Tags', , 'Review_Date', 'Reviewer_Nationality'
        # 'Additional_Number_of_Scoring',
        # 'Total_Number_of_Reviews_Reviewer_Has_Given'
        for feature in ['Average_Score', 'Total_Number_of_Reviews']:
            self.X = self.outliars(self.X, feature)
        self.Y = self.X['Reviewer_Score']
        self.X = self.X.drop(columns='Reviewer_Score')
        return self.X , self.Y
    
    def correlatrion(self):
        corr = self.X.select_dtypes('number').corr()
        sns.heatmap(corr)
        return self.X
    
    def to_numirec(self):
        self.X['lat']=pd.to_numeric(self.X['lat'], errors='coerce')
        self.X['lng']=pd.to_numeric(self.X['lng'], errors='coerce')
        return self.X
    
    
    def scater_box(self):
        fig = px.scatter_mapbox(
            self.data,
            lat='lat',
            lon='lng',
            center={"lat": 51.498241, "lon": -0.004514},  # Map will be centered on Brazil
            width=1200,
            height=600,
            color='Total_Number_of_Reviews',
            hover_data=["Reviewer_Score"],  # Display price when hovering mouse over house
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.show()
        
    def drop_columns(self, cols = COLS):
      for col in cols:
        if col in self.X.columns:
            self.X.drop(col, axis=1, inplace=True)
    

    def label_encoder_y(self):
        le = LabelEncoder()
        le.fit(self.Y)
        y_encoded = le.transform(self.Y)
        self.Y = y_encoded
        

    def process(self, draw=False, outliers=False, cols = COLS):
        self.convertDate()
        self.split_1()
        self.uniqe1()
        self.label_encoder()
        #self.fillNull()
        self.dropNull()
        # if outliers:
        #     self.remove_outlairs()
        self.label_encoder_y()

        if draw:
          self.correlatrion()
        self.to_numirec()
        if draw:
          self.scater_box()
        self.convert_to_int()
        self.address()
        self.drop_columns(cols)
        
    def pipeline(self, draw=False, outliers=False, cols = COLS):
      #self.fillNull()
      #self.dropNull()
      self.process(draw = draw, outliers=outliers, cols = cols)
        


data = pd.read_csv('hotel-classification-dataset.csv')



TAGS = ['trip', 'night']
TRIPS = ['trip', ]
LABELENCODERPATH = "labeler"

class featureEningeering:
    def __init__(self, x_train, x_test, tags = TAGS,
                    trips = TRIPS, 
                    label_path = LABELENCODERPATH):
        self.xtrain = x_train
        self.xtest = x_test
        self.tags = tags
        self.trips = trips

        self.train_label = False
        self.label_path = label_path
        # check path 
        if not os.path.exists(label_path):
            self.train_label = True
            self.label_encoder = LabelEncoder()
        else:
            self.label_encoder = load_model(label_path)

    def __tags_parsing(self, df):
        df['Tags'] = df['Tags'].apply(lambda x: x.strip("[']").replace("', '", ","))
        for tag in self.tags:
            df[tag] = df['Tags'].str.extract(r"(\d+|\w+)\s+{}".format(tag))
        df.drop("Tags", axis=1, inplace=True)
        return df

    def __label_encoding_fitting(self, tags):
        for col in tags:
            self.xtrain[col] = self.label_encoder.fit_transform(self.xtrain[col])
        if self.train_label:
            save_model(self.label_encoder, self.label_path)

    def __label_encoding_transform(self, tags):
        for col in tags:
            self.xtest[col] = self.label_encoder.transform(self.xtest[col])

    def engineering_train(self):
        for df in [self.xtrain]:
            df = self.__tags_parsing(df)
            df['night'] = pd.to_numeric(df['night'], errors='coerce').fillna(0).astype(int)
        self.__label_encoding_fitting(self.trips)

    def engineering_test(self):
        for df in [self.xtest]:
            df = self.__tags_parsing(df)
            df['night'] = pd.to_numeric(df['night'], errors='coerce').fillna(0).astype(int)
        self.__label_encoding_transform(self.trips)


# train all data fel beit
# delete label path "labeler"
"""
feat = featureEningeering(x_train, x_test)
feat.engineering_train()
feat.engineering_test()


# college
feat = featureEningeering(X, X)
feat.engineering_test()
"""









def main_classy():

    # %%
    import pandas as pd

    def label_encode_y(y):
        le = LabelEncoder()

        le.fit(y)

        y_encoded = le.transform(y)

        return y_encoded

    data = pd.read_csv('hotel-classification-dataset.csv')
    nlp_data = copy.deepcopy(data)
    data = data.drop_duplicates()


    y = data['Reviewer_Score']

    # y_encoded = label_encode_y(y)

    # data['Reviewer_Score'] = y_encoded

    #print(data['Reviewer_Score'])

    # %%
    X = data.drop(columns='Reviewer_Score')
    Y = data['Reviewer_Score']


    x_train1, x_test1, y_train1, y_test1 = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

    train = preprocess(x_train1 , y_train1 , data)
    train.pipeline()
    x_train1, y_train1 = train.X, train.Y

    test = preprocess(x_test1 , y_test1 , data)
    test.fillNull()
    test.process()
    x_test1, y_test1 = test.X, test.Y



    from sklearn.preprocessing import LabelEncoder


    # %%
    """   from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train1["Total_Number_of_Reviews"] = scaler.fit_transform(x_train1["Total_Number_of_Reviews"].to_numpy().reshape(-1,1))
    x_train_scaled = x_train1
    # ! only transform
    x_test1["Total_Number_of_Reviews"] = scaler.fit_transform(x_test1["Total_Number_of_Reviews"].to_numpy().reshape(-1,1))
    x_test_scaled = x_test1

    #x_train_scaled.to_numpy()[1]"""

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train1)

    x_test_scaled = scaler.transform(x_test1)

    x_train_scaled = x_train1
    x_test_scaled = x_test1

   

    # %%
    import time


    from sklearn.ensemble import RandomForestClassifier

    model_randforst = RandomForestClassifier(n_estimators=40, max_depth=5)
    start_time = time.time()
    model_randforst = model_randforst.fit(x_train_scaled, y_train1)
    R_training_time = time.time() - start_time
    start_time = time.time()
    y_pred_randforst = model_randforst.predict(x_test_scaled)
    R_testing_time = time.time() - start_time

    mse = mean_squared_error(y_test1, y_pred_randforst)
    print('\nMean squared error Testing Set:', round(mse, 2))

    randomfrst_AccTR = model_randforst.score(x_train_scaled, y_train1)
    print("randomfrst Accuracy train: ", randomfrst_AccTR * 100, "%")

    randomfrst_AccTS = model_randforst.score(x_test_scaled, y_test1)
    print("randomfrst Accuracy test: ", randomfrst_AccTS * 100, "%")


    print("Training time of Random Forest : %.4f seconds" % R_training_time)
    print("Testing time of Random Forest: %.4f seconds" % R_testing_time)

    # save the model    
    filenamelr = 'classification_models/rndclass.sav'

    save_model(model_randforst, filenamelr)
    model_rndclass_loaded= load_model(filenamelr)

    #model_gpreg_loaded.predict(x_test1)

    ###############################
    #LOGISTIC

    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics

    logreg = LogisticRegression(C=1,solver='liblinear', random_state=16)
    start_time = time.time()
    logreg.fit(x_train1, y_train1)
    L_training_time = time.time() - start_time
    start_time = time.time()
    y_pred = logreg.predict(x_test1)
    L_testing_time = time.time() - start_time
    cnf_matrix = metrics.confusion_matrix(y_test1, y_pred)
    cnf_matrix
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    #print ("Accuracy : ", accuracy_score(y_test1, y_pred))

    mse = mean_squared_error(y_test1, y_pred)
    print('\nMean squared error Testing Set:', round(mse, 2))

    logistic_accTR = logreg.score(x_train_scaled, y_train1)
    print("logistic Accuracy train: ", logistic_accTR * 100, "%")

    logistic_accTS = logreg.score(x_test_scaled, y_test1)
    print("logistic Accuracy test: ", logistic_accTS * 100, "%")


    # Print the results
    print("Training time of Logistic Regression : %.4f seconds" % L_training_time)
    print("Testing time of Logistic Regression: %.4f seconds" % L_testing_time)


    # save the model    
    filenamelr = 'classification_models/logclass.sav'

    save_model(logreg, filenamelr)
    model_logclass_loaded= load_model(filenamelr)

    #model_gpreg_loaded.predict(x_test1)
    ###################################

    #GRADIENT

    from sklearn.ensemble import GradientBoostingClassifier
    import matplotlib.pyplot as plt

    #linear reg model
    model_gpreg = GradientBoostingClassifier(n_estimators = 50, max_depth = 5)
    start_time = time.time()
    model_gpreg = model_gpreg.fit(x_train_scaled, y_train1)
    G_training_time = time.time() - start_time
    start_time = time.time()
    y_pred_gpreg = model_gpreg.predict(x_test_scaled)
    G_testing_time = time.time() - start_time

    mse = mean_squared_error(y_test1, y_pred_gpreg)
    print('\nMean squared error Testing Set:', round(mse, 2))

    accuracy = model_gpreg.score( x_train_scaled , y_train1 )
    print( "gpr Accuracy train scaled: ", accuracy * 100, "%")

    linear_Acc=model_gpreg.score( x_test_scaled , y_test1)
    print( "gpr Accuracy test scaled: ", linear_Acc * 100, "%")

    print("Training time of GradientBoostingClassifier : %.4f seconds" % G_training_time)
    print("Testing time of GradientBoostingClassifier: %.4f seconds" % G_testing_time)


    # save the model    
    filenamelr = 'classification_models/gbclass.sav'

    save_model(logreg, filenamelr)
    model_gprclass_loaded= load_model(filenamelr)

    #model_gpreg_loaded.predict(x_test1)

    ########################################
    #DECISION TREE

    from sklearn import tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    #above max depth 10 occur overfitting..zezo
    clf = tree.DecisionTreeClassifier(splitter="best",max_depth=10 )
    start_time_tree_train = time.time()
    clf.fit( x_train_scaled , y_train1)
    TD_training_time_tree = time.time() - start_time_tree_train
    start_time_tree_test = time.time()
    y_pred = clf.predict(x_test_scaled)
    TD_testing_time_tree = time.time() -start_time_tree_test
    accuracy = accuracy_score(y_test1, y_pred)
    print("\nAccuracy of tree :",accuracy*100)
    train_error = 1 - clf.score(x_train_scaled, y_train1)
    test_error = 1 - clf.score(x_test_scaled, y_test1)
    print("Training Error:", train_error)
    print("Test Error:", test_error)
    print("Training time of Decision Tree : %.4f seconds" % TD_training_time_tree)
    print("Testing time of Decision Tree: %.4f seconds" % TD_testing_time_tree)


    # save the model    
    filenamelr = 'classification_models/treeclass.sav'

    save_model(clf, filenamelr)
    model_treeclass_loaded= load_model(filenamelr)

    #####################################
    #SVM

    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    svm = SVC(kernel='rbf', C=0.1)

    start_time_svm_train = time.time()
    svm.fit( x_train_scaled[:40000] , y_train1[:40000])
    training_time_svm = time.time() - start_time_svm_train
    y_pred_train=svm.predict(x_train_scaled[:40000])
    start_time_svm_test = time.time()
    y_pred_test = svm.predict(x_test_scaled[:10000])
    testing_time_svm = time.time() -start_time_svm_test

    accuracy_test = accuracy_score( y_test1[:10000], y_pred_test[:10000])
    print("\nAccuracy SVM model:",accuracy_test*100 )
    train_accuracy = accuracy_score(y_train1[:40000], y_pred_train[:40000])
    train_error = 1 - train_accuracy
    test_error = 1 - accuracy_test
    print("Training Error:", train_error)
    print("Testing Error:", test_error)
    print("Training time of SVM : %.4f seconds" % training_time_svm)
    print("Testing time of SVM: %.4f seconds" % testing_time_svm)

    # save the model    
    filenamelr = 'classification_models/svmclass.sav'

    save_model(clf, filenamelr)
    model_svmclass_loaded= load_model(filenamelr)


    q=['Logistic','Random','GradientBoosting','DecisionTree','SVM']
    v = [logistic_accTS * 100,randomfrst_AccTS * 100,linear_Acc * 100,accuracy*100,accuracy_test*100]
    plt.bar(q,v,color=['skyblue'],width = 0.3,linewidth = 0.5)
    plt.xlabel('Models',fontweight ='bold', fontsize = 10)
    plt.ylabel("Accuracy",fontweight ='bold', fontsize = 10)
    plt.title('Classification Accuracy',fontweight ='bold', fontsize = 15)
    plt.show()


    q=['Logistic','Random','GradientBoosting','DecisionTree','SVM']
    v = [L_training_time,R_training_time,G_training_time,TD_training_time_tree,training_time_svm]
    plt.bar(q,v,color=['seagreen'],width = 0.3,linewidth = 0.5)
    plt.xlabel('Models',fontweight ='bold', fontsize = 10)
    plt.ylabel("Training Time (sec)",fontweight ='bold', fontsize = 10)
    plt.title('Total Training Time',fontweight ='bold', fontsize = 15)
    plt.show()


    q=['Logistic','Random','GradientBoosting','DecisionTree','SVM']
    v = [L_testing_time,R_testing_time,G_testing_time,TD_testing_time_tree,testing_time_svm]
    plt.bar(q,v,color=['purple'],width = 0.3,linewidth = 0.5)
    plt.xlabel('Models',fontweight ='bold', fontsize = 10)
    plt.ylabel("Testing Time (sec)",fontweight ='bold', fontsize = 10)
    plt.title('Total Testing Time',fontweight ='bold', fontsize = 15)
    plt.show()


# %%
from sklearn.model_selection import train_test_split

class NLPDataPreprocessor:
    def __init__(self , a , b):
        self.a = a
        self.b = b
        

    def label_encoder_y(self):
        le = LabelEncoder()
        le.fit(self.b)
        y_encoded = le.transform(self.b)
        self.b = y_encoded   
        
    def preprocess_nlp(self, nlp_data):
        nlp_data['Positive_Review'] = nlp_data['Positive_Review'].str.lower()
        nlp_data['Negative_Review'] = nlp_data['Negative_Review'].str.lower()
        nlp_data ['Reviewer_Score']= np.reshape(self.b, (-1,)) 

        nlp_data.fillna("", inplace=True)
        nlp_data.dropna(axis=0, inplace=True)
        nlp_data.drop_duplicates(inplace=True)
        self.label_encoder_y()
        return nlp_data

    def process_nlp(self, nlp_data):
        preprocessed_data = self.preprocess_nlp(nlp_data)
        self.a = preprocessed_data[['Positive_Review','Negative_Review']]
        self.b = preprocessed_data['Reviewer_Score']
        self.label_encoder_y()




# %%
"""
def label_encoder_y(self):
        le = LabelEncoder()
        le.fit(self.Y)
        y_encoded = le.transform(self.Y)
        self.Y = y_encoded"""

# %%
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def label_encode_y(b):
    le = LabelEncoder()

    le.fit(b)
    y_encoded = le.transform(b)
    return y_encoded

nlp_data = pd.read_csv('hotel-classification-dataset.csv')
nlp_data = copy.deepcopy(data)

b = nlp_data['Reviewer_Score']

# %%



# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# tvec= TfidfVectorizer()
# reg= LinearRegression()

def lol(x):
    return x['Positive_Review']
def lol_(x):
    return x['Negative_Review']

class nlp_model:
    def __init__(self, pred_model = GradientBoostingClassifier(n_estimators=40,learning_rate=0.2), transformer = None):
        self.predictor = pred_model()

        self.transformer = transformer
        if transformer is None:
        #   self.tvec= TfidfVectorizer()
          self.transformer = FeatureUnion([('Positive_Review_tfidf', Pipeline([('extract_field',
                                    FunctionTransformer(lol, validate=False)), ('tfidf', TfidfVectorizer())])),
                                    ('Negative_Review_tfidf', Pipeline([('extract_field', 
                                    FunctionTransformer(lol_, validate=False)), ('tfidf', TfidfVectorizer())]))])

        self.model = Pipeline([('Vectorizer',self.transformer), ('Predictor',self.predictor)])

    def train(self, x_train, y_train, data):
        self.transformer.fit(data)
        self.model.fit(x_train,y_train)
    
    def predict(self, x_test):
        predictions = self.model.predict(x_test)
        return predictions

    def score(self, x, y):
        return self.model.score(x,y)

# %%
def main_nlp_classy():

    train_data, test_data = train_test_split(nlp_data, test_size=0.2, shuffle=True, random_state=42)

    train_size = 40000
    test_size = 10000

    train_data = train_data.sample(n=train_size, random_state=42)
    test_data = test_data.sample(n=test_size, random_state=42)

    train_preprocessor = NLPDataPreprocessor(train_data[['Positive_Review','Negative_Review']], train_data['Reviewer_Score'])
    test_preprocessor = NLPDataPreprocessor(test_data[['Positive_Review','Negative_Review']], test_data['Reviewer_Score'])

    train_preprocessor.process_nlp(train_data)
    test_preprocessor.process_nlp(test_data)

    a_train, b_train = train_preprocessor.a, train_preprocessor.b
    a_test, b_test = test_preprocessor.a, test_preprocessor.b

    mdl1_ = nlp_model(GradientBoostingClassifier)
    mdl1_.train(a_train, b_train, nlp_data)

    print("accuracy", mdl1_.score(a_train,b_train)* 100, "%")
    print("accuracy test", mdl1_.score(a_test, b_test)* 100, "%")


    print(mean_squared_error(mdl1_.predict(a_test), b_test) )

    preds = mdl1_.predict(a_test)

    import joblib

    # Save the pipeline to a file
    joblib.dump(mdl1_, 'nlp_dod.joblib')

    # Load the pipeline from the file
    loaded_pipeline = joblib.load('nlp_dod.joblib')

    # Use the loaded pipeline to make predictions
    pred2 = loaded_pipeline.predict(a_test)


    # save the model    
    filenamelr = 'classification_models/idf_nlp_class.sav'

    save_model(mdl1_, filenamelr)
    model_idfclass_loaded= load_model(filenamelr)
    #model_gpreg_loadedddd.predict(a_test)

    


