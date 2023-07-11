import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
import warnings
import copy
import pickle

warnings.filterwarnings("ignore")


# %%
COLS = ['Hotel_Address', 'Hotel_Name', 'Review_Date', 'days_since_review', 'lat', 'lng', 'Positive_Review', 'Negative_Review']

# %%
def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
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
        self.X.fillna("", inplace=True)
        self.Y.fillna("", inplace=True)
        return self.X , self.Y

    def dropNull(self):
        self.X.dropna(axis=0, inplace=True)
        self.Y.dropna(axis=0, inplace=True)
        return self.X , self.Y

    def convertDate(self):
        self.X['Review_Date'] = pd.to_datetime(self.X['Review_Date'])
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
        label_encoder = preprocessing.LabelEncoder()
        self.X['Tags']= label_encoder.fit_transform(self.X['Tags'])
        label_encoder2 = preprocessing.LabelEncoder()
        self.X['Reviewer_Nationality']= label_encoder2.fit_transform(self.X['Reviewer_Nationality'])
        return self.X

    def remove_outlairs(self):
        self.X = pd.concat([self.X, self.Y], axis=1)
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
    
    def process(self, draw=True, cols = COLS):
        self.convertDate()
        self.split_1()
        self.uniqe1()
        self.label_encoder()
        #self.remove_outlairs()
        if draw:
          self.correlatrion()
        self.to_numirec()
        if draw:
          self.scater_box()
        self.convert_to_int()
        self.drop_columns(cols)
    def pipeline(self, draw=True, cols = COLS):
        #self.fillNull()
        #self.dropNull()
        self.process(draw = draw, cols = cols)
        self.drop_columns(['Tags'])
        

def main_reg():
    # %%
    data = pd.read_csv('hotel-regression-dataset.csv')
    nlp_data = copy.deepcopy(data)

    # %%
    X = data.drop(columns='Reviewer_Score')
    Y = data['Reviewer_Score']


    # %%
    x_train1, x_test1, y_train1, y_test1 = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

    # %%
    train = preprocess(x_train1 , y_train1 , data)
    train.pipeline()
    x_train1, y_train1 = train.X, train.Y

    test = preprocess(x_test1 , y_test1 , data)
    test.fillNull()
    test.process()
    x_test1, y_test1 = test.X, test.Y

    # %%
    """  from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    ## /*-6522
    x_train1["Total_Number_of_Reviews"] = scaler.fit_transform(x_train1["Total_Number_of_Reviews"].to_numpy().reshape(-1,1))
    x_train_scaled = x_train1.drop("Tags", axis=1)

    x_test1["Total_Number_of_Reviews"] = scaler.fit_transform(x_test1["Total_Number_of_Reviews"].to_numpy().reshape(-1,1))
    x_test_scaled = x_test1.drop("Tags", axis=1)"""

    # %%
    #x_train_scaled.to_numpy()[1]
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train1)

    x_test_scaled = scaler.transform(x_test1)

    x_train_scaled = x_train1
    x_test_scaled = x_test1


    # %%

    ## /*-6522
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)



    # %%

    from sklearn.ensemble import RandomForestRegressor

    model_randforst = RandomForestRegressor(n_estimators = 40, max_depth = 5,)
    model_randforst = model_randforst.fit(x_train_scaled, y_train1)
    y_pred_randforst = model_randforst.predict(x_test_scaled)
    y_pred_randforst= y_pred_randforst.clip(0,10)

    mse = mean_squared_error(y_test1, y_pred_randforst)
    print('\nMean squared error Testing Set:', round(mse, 2))

    randomfrst_Acc = model_randforst.score( x_train_scaled , y_train1 )
    print( "randomfrst Accuracy train: ", randomfrst_Acc * 100, "%")

    randomfrst_Acc=model_randforst.score( x_test_scaled , y_test1)
    print( "randomfrst Accuracy test: ", randomfrst_Acc * 100, "%")

    plt.scatter(y_test1, y_test1, label="data",color='red')
    plt.scatter(y_test1, y_pred_randforst, color='green', linewidth=2, label="actual prediction")
    plt.plot(y_pred_randforst, y_pred_randforst, label="prediction line")

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()
    
    # save the model    
    filenamelr = 'reg_models/RandomForestRegressor.sav'

    save_model(model_randforst, filenamelr)
    model_RandomForestRegressor_loaded= load_model(filenamelr)


    # %%
    model_linreg = LinearRegression()
    model_linreg = model_linreg.fit(x_train_pca, y_train1)
    y_pred_linreg = model_linreg.predict(x_test_pca)
    y_pred_linreg = y_pred_linreg.clip(0,10)


    mse = mean_squared_error(y_test1, y_pred_linreg)
    print('\nMean squared error Testing Set:', round(mse, 2))

    #mae = mean_absolute_error(y_test1, y_pred_linreg)
    #print('Mean absolute error Testing Set:', round(mae, 2))
    #x_train_pca,x_test_pca
    accuracy = model_linreg.score( x_train_pca , y_train1 )
    print( "linear Accuracy train_pca: ", accuracy * 100, "%")

    linear_Acc=model_linreg.score( x_test_pca , y_test1)
    print( "linear Accuracy test_pca: ", linear_Acc * 100, "%")

    plt.scatter(y_test1, y_test1, label="data",color='red')
    plt.scatter(y_test1, y_pred_linreg, color='green', linewidth=2, label="actual prediction")
    plt.plot(y_pred_linreg, y_pred_linreg, label="prediction line")

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

    # save the model    
    filenamelr = 'reg_models/LinearRegression_pca.sav'

    save_model(model_linreg, filenamelr)
    model_LinearRegression_pca_loaded= load_model(filenamelr)


    # %%

    from sklearn.ensemble import GradientBoostingRegressor
    import matplotlib.pyplot as plt

    #linear reg model
    model_gpreg = GradientBoostingRegressor()
    model_gpreg = model_gpreg.fit(x_train_scaled, y_train1)
    y_pred_gpreg = model_gpreg.predict(x_test_scaled)
    y_pred_gpreg = y_pred_gpreg.clip(0,10)


    mse = mean_squared_error(y_test1, y_pred_gpreg)
    print('\nMean squared error Testing Set:', round(mse, 2))


    accuracy = model_gpreg.score( x_train_scaled , y_train1 )
    print( "gpr Accuracy train scaled: ", accuracy * 100, "%")

    linear_Acc=model_gpreg.score( x_test_scaled , y_test1)
    print( "gpr Accuracy test scaled: ", linear_Acc * 100, "%")

    plt.scatter(y_test1, y_test1, label="data",color='red')
    plt.scatter(y_test1, y_pred_gpreg, color='green', linewidth=2, label="actual prediction")
    plt.plot(y_pred_gpreg, y_pred_gpreg, label="prediction line")

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

 # save the model    
    filenamelr = 'reg_models/GradientBoostingRegressor.sav'

    save_model(model_gpreg, filenamelr)
    model_GradientBoostingRegressor_loaded= load_model(filenamelr)


    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt

    #linear reg model
    model_linreg_sc = LinearRegression()
    model_linreg_sc = model_linreg_sc.fit(x_train_scaled, y_train1)
    y_pred_linreg   = model_linreg_sc.predict(x_test_scaled)
    y_pred_linreg   = y_pred_linreg.clip(0,10)


    mse = mean_squared_error(y_test1, y_pred_linreg)
    print('\nMean squared error Testing Set:', round(mse, 2))

    accuracy = model_linreg_sc.score( x_train_scaled , y_train1 )
    print( "linear Accuracy train scaled: ", accuracy * 100, "%")

    linear_Acc=model_linreg_sc.score( x_test_scaled , y_test1)
    print( "linear Accuracy test scaled: ", linear_Acc * 100, "%")

    plt.scatter(y_test1, y_test1, label="data",color='red')
    plt.scatter(y_test1, y_pred_linreg, color='green', linewidth=2, label="actual prediction")
    plt.plot(y_pred_linreg, y_pred_linreg, label="prediction line")

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

 # save the model    
    filenamelr = 'reg_models/LinearRegression_scaled.sav'

    save_model(model_linreg_sc, filenamelr)
    model_LinearRegression_scaled_loaded= load_model(filenamelr)
    


    model_dtr = DecisionTreeRegressor(random_state=42,
        max_depth = 10,
        min_samples_split = 2,
        min_samples_leaf= 1,
        min_weight_fraction_leaf= 0,
        max_features= 6,
        max_leaf_nodes = 16,)

    model_dtr = model_dtr.fit(x_train_scaled, y_train1)
    y_pred_dtr = model_dtr.predict(x_test_scaled)

    mse = mean_squared_error(y_test1, y_pred_dtr)
    print('\nMean squared error Testing Set:', round(mse, 2))

    tree_Acc=model_dtr.score( x_train_scaled , y_train1)
    print( "tree Accuracy train: ", tree_Acc * 100, "%")

    tree_Acc=model_dtr.score( x_test_scaled , y_test1)
    print( "tree Accuracy test: ", tree_Acc * 100, "%")



    plt.scatter(y_test1, y_test1, label="data",color='red')
    plt.scatter(y_test1, y_pred_dtr, color='green', linewidth=2, label="actual prediction")
    plt.plot(y_pred_dtr, y_pred_dtr, label="prediction line")

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

     # save the model    
    filenamelr = 'reg_models/DecisionTreeRegressor.sav'

    save_model(model_dtr, filenamelr)
    model_DecisionTreeRegressor_loaded= load_model(filenamelr)

    # %%
    from sklearn.linear_model import Lasso


    #lasso regression
    #print("\nLasso Regression")
    lasso = Lasso(alpha=0.1)
    lasso.fit(x_train_scaled, y_train1)
    y_predlasso = lasso.predict(x_test_scaled)
    y_predlasso = y_predlasso.clip(0,10)


    mselasso = mean_squared_error(y_test1, y_predlasso)
    print('Mean squared error Testing Set (lasso regression):', round(mselasso, 2))


    lasso_Acc=lasso.score( x_train_scaled , y_train1)
    print( "lasso Accuracy train: ", lasso_Acc * 100, "%")

    lasso_Acc=lasso.score( x_test_scaled , y_test1)
    print( "lasso Accuracy test: ", lasso_Acc * 100, "%")

    plt.scatter(y_test1, y_test1, label="data",color='red')
    plt.scatter(y_test1, y_predlasso, color='green', linewidth=2, label="actual prediction")
    plt.plot(y_predlasso, y_predlasso, label="prediction line")

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

      # save the model    
    filenamelr = 'reg_models/lasso.sav'

    save_model(lasso, filenamelr)
    model_lassoRegressor_loaded= load_model(filenamelr)

    # %%
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import numpy as np

    # Ridge regression
    ridgereg = Ridge(alpha=0.1)
    ridgereg.fit(x_train_scaled, y_train1)
    y_pred_ridg = ridgereg.predict(x_test_scaled)
    y_pred_ridg = y_pred_ridg.clip(0,10)


    mse_ridge = mean_squared_error(y_test1, y_pred_ridg)
    print('Mean squared error Testing Set (ridge regression):', round(mse_ridge, 2))

    ridge_train_acc = ridgereg.score(x_train_scaled, y_train1)
    print("Ridge train accuracy:", ridge_train_acc * 100, "%")

    ridge_test_acc = ridgereg.score(x_test_scaled, y_test1)
    print("Ridge test accuracy:", ridge_test_acc * 100, "%")


    plt.scatter(y_test1, y_test1, label="data",color='red')
    plt.scatter(y_test1, y_pred_ridg, color='green', linewidth=2, label="actual prediction")
    plt.plot(y_pred_ridg, y_pred_ridg, label="prediction line")

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

    # save the model    
    filenamelr = 'reg_models/ridgereg.sav'

    save_model(ridgereg, filenamelr)
    model_ridgeregRegressor_loaded= load_model(filenamelr)

    # %%
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    poly = PolynomialFeatures(degree=2)
    x_poly_train = poly.fit_transform(x_train1)

    model = LinearRegression()
    model.fit(x_poly_train, y_train1)

    x_poly_test = poly.transform(x_test1)
    y_pred_poly = model.predict(x_poly_test)
    y_pred_poly = y_pred_poly.clip(0,10)

    mse_poly = mean_squared_error(y_test1, y_pred_poly)
    print('Mean squared error Testing Set poly:', round(mse_poly, 2))

    poly_train_acc = model.score(x_poly_train, y_train1)
    print("Poly train accuracy:", poly_train_acc * 100, "%")

    poly_test_acc = model.score(x_poly_test, y_test1)
    print("Poly test accuracy:", poly_test_acc * 100, "%")

  # save the model    
    filenamelr = 'reg_models/Polynomial.sav'

    save_model(model, filenamelr)
    model_PolynomialRegressor_loaded= load_model(filenamelr)


# %%

class NLPDataPreprocessor:
    def __init__(self , a , b):
        self.a = a
        self.b = b
        
    def preprocess_nlp(self, nlp_data):
        nlp_data['Positive_Review'] = nlp_data['Positive_Review'].str.lower()
        nlp_data['Negative_Review'] = nlp_data['Negative_Review'].str.lower()

        nlp_data.fillna("", inplace=True)
        nlp_data.dropna(axis=0, inplace=True)
        nlp_data.drop_duplicates(inplace=True)
        return nlp_data

    def process_nlp(self, nlp_data):
        preprocessed_data = self.preprocess_nlp(nlp_data)
        self.a = preprocessed_data[['Positive_Review','Negative_Review']]
        self.b = preprocessed_data['Reviewer_Score']



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline



def lol(x):
    return x['Positive_Review']
def lol_(x):
    return x['Negative_Review']

class nlp_model:
    def __init__(self, pred_model = LinearRegression, transformer = None):
        self.predictor = pred_model()

        self.transformer = transformer
        if transformer is None:
          self.transformer = FeatureUnion([('Positive_Review_tfidf', Pipeline([('extract_field',
                                    FunctionTransformer(lol, validate=False)), ('tfidf', TfidfVectorizer())])),
                                    ('Negative_Review_tfidf', Pipeline([('extract_field', 
                                    FunctionTransformer(lol_, validate=False)), ('tfidf', TfidfVectorizer())]))])

        self.model = Pipeline([('Vectorizer',self.transformer), ('Predictor',self.predictor)])

    def train(self, x_train, y_train, data):
        self.transformer.fit(data)
        self.model.fit(x_train,y_train)
    
    def predict(self, x_test):
        predictions = self.model.predict(x_test).clip(0,10)
        return predictions

    def score(self, x, y):
        return self.model.score(x,y)



def main_nlp():
    # %%
    
    data = pd.read_csv('hotel-regression-dataset.csv')
    nlp_data = copy.deepcopy(data)

    train_data, test_data = train_test_split(nlp_data, test_size=0.2, shuffle=True, random_state=42)

    train_preprocessor = NLPDataPreprocessor(train_data[['Positive_Review','Negative_Review']], train_data['Reviewer_Score'])
    test_preprocessor = NLPDataPreprocessor(test_data[['Positive_Review','Negative_Review']], test_data['Reviewer_Score'])

    train_preprocessor.process_nlp(train_data)
    test_preprocessor.process_nlp(test_data)

    a_train, b_train = train_preprocessor.a, train_preprocessor.b
    a_test, b_test = test_preprocessor.a, test_preprocessor.b

    """   data = pd.read_csv('hotel-regression-dataset.csv')
    nlp_data = copy.deepcopy(data)

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
    a_test, b_test = test_preprocessor.a, test_preprocessor.b"""

# %%

    mdl1_ = nlp_model(GradientBoostingRegressor)
    mdl1_.train(a_train, b_train, nlp_data)
    print("accuracy  TfidfVectorizer nlp", mdl1_.score(a_train,b_train)* 100, "%")
    print("accuracy test", mdl1_.score(a_test, b_test)* 100, "%")
    print(mean_squared_error(mdl1_.predict(a_test), b_test) )

    preds = mdl1_.predict(a_test)
    plt.scatter(b_test, b_test, label="data",color='red')
    plt.scatter(b_test, preds, color='green', linewidth=2, label="actual prediction")
    plt.plot(preds, preds, label="prediction line")

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

    # save the model    
    filenamelr = 'reg_models/id_nlp.sav'

    save_model(mdl1_, filenamelr)
    model_idnlp_loaded= load_model(filenamelr)



    # %%

    mdl2_ = nlp_model(GradientBoostingRegressor)
    mdl2_.train(a_train, b_train, nlp_data)

    print("accuracy GradientBoostingRegressor nlp", mdl2_.score(a_train,b_train)* 100, "%")
    print("accuracy test", mdl2_.score(a_test, b_test)* 100, "%")
    print(mean_squared_error(mdl2_.predict(a_test), b_test) )

    preds = mdl2_.predict(a_test)
    plt.scatter(b_test, b_test, label="data",color='red')
    plt.scatter(b_test, preds, color='green', linewidth=2, label="actual prediction")
    plt.plot(preds, preds, label="prediction line")

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

 # save the model    
    filenamelr = 'reg_models/gb_nlp.sav'

    save_model(mdl2_, filenamelr)
    model_gbnlp_loaded= load_model(filenamelr)


"""
# sha8al
mdl3 = Pipeline([('Vectorizer',transformer), ('Predictor', RandomForestRegressor(n_estimators=50, max_leaf_nodes=10))])
mdl3.fit(a_train,b_train)
pred3 = mdl3.predict(a_test)
predd = pred3.clip(0,10)

print("accuracy train", mdl3.score(a_train,b_train)* 100, "%")
print("accuracy test", mdl3.score(a_test,b_test)* 100, "%")
print(mean_squared_error(pred3, b_test) )


plt.scatter(b_test, b_test, label="data",color='red')
plt.scatter(b_test, pred3, color='green', linewidth=2, label="actual prediction")
plt.plot(pred3, pred3, label="prediction line")

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()
"""

