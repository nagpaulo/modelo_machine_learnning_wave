## Modelo de Aprendizagem de Dados
    Modelo criado através de práticas e cases nos usos de boas práticas  
    em ciência de dados explorando dados diversos.
    Aplicar técnicas de Machine Learning, Deep Learning
    e Generative AI para endereçar problemas de negócio, melhorando
    produtividade e relacionamento  com clientes e usuários.

## CODE: Train e Test - Split-out validation dataset
```    
    array = df.values
    X = array[:,3:8]
    Y = array[:,1]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    print('Resultado do Training: ' + str(len(X_train)), 'Test: '+str(len(X_validation)))
```

## Test options and evaluation metric
```
seed = 7
scoring = 'accuracy'
```

## CODE: Spot Check Algorithms
```
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
   kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
   cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring, error_score='raise')
   results.append(cv_results)
   names.append(name)
   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   print(msg)
```


## CODE: Make predictions on validation dataset
```
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
```

## Result the train
```
0.21875
[[0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 0 1 1 0 6 1 1 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
 [0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 2 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1]]
                          precision    recall  f1-score   support

             ALCINÓPOLIS       0.00      0.00      0.00         1
               ANASTÁCIO       0.00      0.00      0.00         1
            ANAURILÂNDIA       0.00      0.00      0.00         1
    APARECIDA DO TABOADO       0.00      0.00      0.00         0
              AQUIDAUANA       0.00      0.00      0.00         0
                  BONITO       0.00      0.00      0.00         0
            CAMPO GRANDE       0.33      0.55      0.41        11
                 CORUMBÁ       0.00      0.00      0.00         0
                DOURADOS       0.00      0.00      0.00         3
                IVINHEMA       0.00      0.00      0.00         1
                MARACAJU       0.00      0.00      0.00         1
                 MIRANDA       0.00      0.00      0.00         2
                 NAVIRAÍ       0.00      0.00      0.00         2
                 NIOAQUE       0.00      0.00      0.00         1
               PARANAÍBA       0.00      0.00      0.00         3
           RIO BRILHANTE       0.00      0.00      0.00         1
RIO VERDE DE MATO GROSSO       0.00      0.00      0.00         1
                SELVÍRIA       0.00      0.00      0.00         1
             TRÊS LAGOAS       0.50      0.50      0.50         2

                accuracy                           0.22        32
               macro avg       0.04      0.06      0.05        32
            weighted avg       0.15      0.22      0.17        32

/home/bebeto/Documentos/Desenvolvimento/CienciaDeDados/modelo_machine_learning_vitae/.venv/lib64/python3.12/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/bebeto/Documentos/Desenvolvimento/CienciaDeDados/modelo_machine_learning_vitae/.venv/lib64/python3.12/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/bebeto/Documentos/Desenvolvimento/CienciaDeDados/modelo_machine_learning_vitae/.venv/lib64/python3.12/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/bebeto/Documentos/Desenvolvimento/CienciaDeDados/modelo_machine_learning_vitae/.venv/lib64/python3.12/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/bebeto/Documentos/Desenvolvimento/CienciaDeDados/modelo_machine_learning_vitae/.venv/lib64/python3.12/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/bebeto/Documentos/Desenvolvimento/CienciaDeDados/modelo_machine_learning_vitae/.venv/lib64/python3.12/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
```
