############################################
# Feature Engineering: Diabetes dataset
############################################

############################################
# Imports, functions and settings
############################################
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date

from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

## functions
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name,q1 = q1,q3=q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers
def grab_col_names(dataframe, cat_th=10, car_th=20):
        """

        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optinal
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.
            Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

        """

        # cat_cols, cat_but_car
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                       dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car
def replace_with_thresholds(dataframe, variable,q1 = 0.25,q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable,q1=q1,q3=q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


## information functions

def check_df(dataframe, head=5,plot = False,corr = False):
    print("##   Shape    ##")
    print(dataframe.shape)
    print("##   Types    ##")
    print(dataframe.dtypes)
    print("##   Head   ##")
    print(dataframe.head(head))
    print("##   Tail   ##")
    print(dataframe.tail(head))
    print("##   Missing entries   ##")
    print(dataframe.isnull().sum())
    print("##   Quantiles   ##")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##   general information   ##")
    print(dataframe.describe().T)
    if plot:
        showAllDists(dataframe)
    if corr:
        corr = dataframe.corr()

        # plot the heatmap
        sns.heatmap(corr,
                    xticklabels=corr.columns,
                    yticklabels=corr.columns)
        plt.show(block=True)

def showAllDists(dataframe):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    for col in num_cols:
        sns.distplot(df[col])
        plt.title("Distrubution of " + col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, cat_col):
    print(dataframe.groupby(cat_col).agg({target: "mean"}), end="\n\n\n")

### one hot encoder

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

### correlation functions

def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 10 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

def plot_Corr_Dot(dff):
    sns.set(style='white', font_scale=1)
    g = sns.PairGrid(dff, aspect=1.4, diag_sharey=False)
    g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
    g.map_diag(sns.distplot, kde_kws={'color': 'black'})
    g.map_upper(corrdot)
    plt.show(block = True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
import warnings
warnings.filterwarnings("ignore")
############################################
# EDA
############################################

df = pd.read_csv("diabetes.csv")

check_df(df)

# As a first look at the data, we notice no obvious outliers except for the insulin, Although the
# data doesn't contain missing values, we notice that some values are zero while they can't be zero
# the insulin or the Glucose for example. this could mean that the missing values are replaced with zero.

# Show distributions of the data
showAllDists(df)

# Plot the correlations of the data
plot_Corr_Dot(df)

# Separate the numeric and the categorical variables
cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    target_summary_with_cat(df,"Outcome",col)
    cat_summary(df,col)

for col in num_cols:
    target_summary_with_num(df,"Outcome",col)

# We can see a clear difference in the values of the insulin and glucose between the 1 and 0 labeled
# rows, which is expected.

############## Correlation ##############
for col in num_cols:
    print(col, check_outlier(df,col,q1=0.05,q3=0.95))

# As mentioned before, only the insulin has outliers, we will replace them with the upper quantile (95%)

replace_with_thresholds(df,"Insulin",q1=0.05,q3=0.95)

############## Missing Values ##############

# As mentioned before, the data doesn't show any missing values, but it has some not reasonable values for
# some variables, we will replace these variables using KNN method.


# Replace zeros with nans
To_Replace_List = ["Glucose","BloodPressure","SkinThickness","Insulin","Insulin"]

for col in To_Replace_List:
    df[col].replace(0,np.nan,inplace=True)

## fill it by predictions (K Nearset Neighbours - KNN)

## standardize the variables
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns) ## to apply the predictions
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns) ## to reverse to the normal values


############## Feature Engineering ##############

def check_BMI(row):
    if row["BMI"] < 18.5:
        return "underweight "
    elif row["BMI"] >= 18.5 and row["BMI"] <= 24.9:
        return "healthyweight"
    elif  row["BMI"] >= 25 and row["BMI"] <= 29.9:
        return "overweight"
    else:
        return "obesse"

df["BMI_STATUS"] = df.apply(check_BMI,axis=1)

def check_Blood_Pressure(row):
    if row["BloodPressure"] < 80 :
        return "Normal "
    elif row["BloodPressure"] >= 80 and row["BloodPressure"] <= 89 :
        return "Stage I Hypertension (mild)"
    elif row["BloodPressure"] >= 90  and row["BloodPressure"] <= 120  :
        return "Stage 2 Hypertension (moderate)"
    elif  row["BloodPressure"] >= 120:
        return "Hypertensive Crisis (get emergency care)"

df["BloodPressure_STATUS"] = df.apply(check_Blood_Pressure,axis=1)

def check_Age(row):
    if row["Age"] < 35:
        return "Young"
    elif row["Age"] >= 35 and row["Age"] < 55:
        return "Mid_Age"
    else:
        return "Old"

df["Age_Group"] = df.apply(check_Age,axis=1)
sns.countplot(data = df,x = "Age_Group")
plt.show(block = True)

def check_Glucose(row):
    if row["Glucose"] < 140:
        return "Normal"
    elif row["Glucose"] >= 140 and row["Glucose"] <= 199:
        return "Prediabetes"

df["Glucose_Status"] = df.apply(check_Glucose,axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols.remove("Outcome")
############## Encode ##############

df = one_hot_encoder(df,cat_cols)
df.head()

############## Build the model ##############

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)




def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y)

# We noticed that Random Forests and GBM got the best results, now we try to make some hyperparameter optimization


knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5,7,9,3,11,13,15,19, None],
             "max_features": [5,7,9,3,11,13,15,19, "auto"],
             "min_samples_split": [5,7,9,3,11,13,15,19],
             "n_estimators": [100,200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": range(1, 20),
                  "n_estimators": [100,200, 300],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1,0.02,0.03,0.07],
                   "n_estimators": [100,200, 300],
                   "colsample_bytree": [0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)

# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)

## Our final results
# Voting Classifier...
# Accuracy: 0.7994791666666666
# F1Score: 0.7017543859649122
# ROC_AUC: 0.8630142314531625