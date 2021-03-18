```python
import pandas as pd
```


```python
import numpy as np
from sklearn import tree
from sklearn import preprocessing
```


```python
titanic_train = pd.read_csv("Titanic_R.csv")
titanic_train.columns
```




    Index(['pclass', 'survived', 'Residence', 'name', 'age', 'sibsp', 'parch',
           'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest',
           'Gender'],
          dtype='object')




```python
new_age_var=np.where(titanic_train["age"].isnull(),28,titanic_train["age"])
```


```python
titanic_train["age"]=new_age_var
```


```python
le=preprocessing.LabelEncoder()
```


```python
encoded_age=le.fit_transform(titanic_train["age"])
```


```python
tree_model=tree.DecisionTreeClassifier()
```


```python
tree_model.fit(X=pd.DataFrame(encoded_age),y=titanic_train["survived"])
```




    DecisionTreeClassifier()




```python
with open("DT1.dot",'w') as f:
    f=tree.export_graphviz(tree_model,feature_names=["Gender"],out_file=f)
```


```python
#open the file ,copy and the open the browser and type webgraphviz and then paste it over there and generate the graph
```


```python
with open("DT2.dot",'w') as f:
    f=tree.export_graphviz(tree_model,feature_names=["age"],out_file=f)
```


```python
#with open("DT3.dot",'w') as f:
 #   f=tree.export_graphviz(tree_model,feature_names=["Gender","pclass"],out_file=f)
```


```python
predictors=pd.DataFrame([encoded_age,titanic_train["pclass"]]).T
```


```python
tree_model.fit(X=predictors,y=titanic_train["survived"])
```




    DecisionTreeClassifier()




```python
with open("DT4.dot",'w') as f:
    f=tree.export_graphviz(tree_model,feature_names=["Gender","pclass"],out_file=f)
```


```python
tree_model.score(X=predictors,y=titanic_train['survived'])
```




    0.747135217723453




```python

```


```python

```
