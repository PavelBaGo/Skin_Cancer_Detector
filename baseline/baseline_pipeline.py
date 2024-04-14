from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import cross_validate, learning_curve, train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression


from baseline.preprocess import sampler, labelize

preproc1 = FunctionTransformer(lambda df: labelize(df))
preproc2 = FunctionTransformer(lambda df: sampler(df))
preproc3 = FunctionTransformer(lambda df: df.drop(columns='label')/255)

preproc = make_pipeline(preproc1,preproc2,preproc3)
