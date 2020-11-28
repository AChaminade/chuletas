# Aprendizaje SUpervisado con Python
## Set de entrenamiento y test

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
```

## Regresión
### Regresión lineal

```python

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train,y_train)

reg.predict(X_test)

# comprobación de la validez del modelo

from sklearn.metrics import mean_absolute_error

mean_absolute_error(reg.predict(X_test),y_test) # MEA

np.mean(np.abs(reg.predict(X_test)-y_test)/y_test) #MAPE

```
