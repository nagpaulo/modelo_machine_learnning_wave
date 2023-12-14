## Modelo de Aprendizagem de Dados
    Modelo criado através de práticas e cases nos usos de boas práticas  
    em ciência de dados explorando dados diversos.
    Aplicar técnicas de Machine Learning, Deep Learning
    e Generative AI para endereçar problemas de negócio, melhorando
    produtividade e relacionamento  com clientes e usuários.

### CODE de Train e Test
```    
    array = df.values
    X = array[:,3:8]
    Y = array[:,1]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    print('Resultado do Training: ' + str(len(X_train)), 'Test: '+str(len(X_validation)))
```
