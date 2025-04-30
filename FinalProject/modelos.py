# 1. Importação das bibliotecas necessárias
# Bibliotecas gerais
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


from keras import optimizers

# ARVORE DE DECISAO --------------------------------------------------------------------------------------------
def modelo_arvore_decisao(X_train, X_test, y_train, y_test):
    
    """
    Treina e ajusta uma árvore de decisão com poda usando Minimal Cost-Complexity,
    de forma otimizada para grandes volumes de dados.
    """

    # Treina árvore inicial com profundidade máxima limitada (evita árvore muito pesada)
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    ccp_alphas = ccp_alphas[ccp_alphas >= 1e-10]  #remover valores muito pequenos
    
    param_grid = {'ccp_alpha': ccp_alphas}
    grid_search = GridSearchCV( #GridSearchCV faz o ajuste automatico dos hiperparametros
        DecisionTreeClassifier(random_state=0), # class_weight='balanced' se quiser balancear as classes
        param_grid=param_grid,
        cv=5,
        n_jobs=-1, #permite que o processo utilize todos os núcleos disponíveis
        verbose=0 #desativa as mensagens de log durante a execução.
    )
    # random_state=0 (pode ser qualquer número inteiro, que será fixo) fixo, 
    # significa que o resultado será sempre o mesmo, mesmo com diferentes execuções
    # se não colocar nenhum valor, o resultado será diferente a cada execução

    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_ #melhor estimador encontrado pelo GridSearchCV

    # Avaliações
    y_pred_train = best_clf.predict(X_train)
    y_pred_test = best_clf.predict(X_test)

    Ein = 1 - accuracy_score(y_train, y_pred_train)
    Eout = 1 - accuracy_score(y_test, y_pred_test)

    print("Árvore de Decisão Otimizada")
    print(f"Melhor alpha: {grid_search.best_params_['ccp_alpha']}")
    print(f"Ein: {Ein:.4f}")
    print(f"Eout: {Eout:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_test))
    best_tree_depth = best_clf.tree_.max_depth
    print(f"A profundidade da árvore para o melhor alpha é: {best_tree_depth}")
    
    # Plot da árvore (pode limitar o número de folhas se quiser também)
    plt.figure(figsize=(20,8))
    
    plot_tree(
        best_clf,
        feature_names=X_train.columns, 
        class_names=[str(i) for i in np.unique(y_train)],
        filled=True,
        proportion = True,
        fontsize=6,
        rounded = True)
    plt.show()

    return best_clf

#-- CALCULO DA PROFUNDIDADE DA ÁRVORE  --------------------------------------------------------------------------------------------

def best_depth_for_tree(X_train, X_test, y_train, y_test):
    max_depth = [i for i in range(1, 20)]
    tree_metrics_dict_val = {i: {'train_accuracy': None, 'test_accuracy': None, 'precision': None, 'recall': None, 'f1': None} for i in max_depth}

    """
    Treina e ajusta uma árvore de decisão com diferentes profundidades,
    avaliando as métricas de treino e teste.
    """

    

    print('max_depth tested:', end='  ')
    for i in max_depth:
        # Instancia e treina o modelo
        tree_classifier = DecisionTreeClassifier(max_depth=i, random_state=0)
        tree_classifier.fit(X_train, y_train)

        # Predição nos dados de treino e teste
        yhat_train = tree_classifier.predict(X_train)
        yhat_test = tree_classifier.predict(X_test)

        # Calcula as métricas
        train_accuracy = accuracy_score(y_train, yhat_train)
        test_accuracy = accuracy_score(y_test, yhat_test)
        precision = precision_score(y_test, yhat_test, average='weighted')
        recall = recall_score(y_test, yhat_test, average='weighted')
        f1 = f1_score(y_test, yhat_test, average='weighted')

        # Armazena as métricas no dicionário
        tree_metrics_dict_val[i] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        print(f'{i}', end='  ')

    # Encontra a melhor profundidade com base na acurácia de teste
    best_max_depth_tree = max(tree_metrics_dict_val, key=lambda k: tree_metrics_dict_val[k]['test_accuracy'])

    # Cria um dicionário para as melhores métricas
    dt_metrics_best_val = tree_metrics_dict_val[best_max_depth_tree]

    # Plot das métricas
    train_accuracy = [tree_metrics_dict_val[n]['train_accuracy'] for n in max_depth]
    test_accuracy = [tree_metrics_dict_val[n]['test_accuracy'] for n in max_depth]
    precision = [tree_metrics_dict_val[n]['precision'] for n in max_depth]
    recall = [tree_metrics_dict_val[n]['recall'] for n in max_depth]
    f1 = [tree_metrics_dict_val[n]['f1'] for n in max_depth]

    plt.figure(figsize=(10, 6))
    plt.plot(max_depth, train_accuracy, '-o', label='Train Accuracy')
    plt.plot(max_depth, test_accuracy, '-o', label='Test Accuracy')
    # plt.plot(max_depth, precision, '-o', label='Precision')
    # plt.plot(max_depth, recall, '-o', label='Recall')
    # plt.plot(max_depth, f1, '-o', label='F1 Score')

    # Configurações do gráfico
    plt.xlabel('Max_depth')
    plt.ylabel('Score')
    plt.title('Comparison of Decision Tree Metrics (Train vs Test)')
    plt.xticks(max_depth)
    plt.legend(loc='best')
    plt.grid(True)

    plt.show()

    print(f'Best max_depth_tree: {best_max_depth_tree}')
    print(f'Metrics for best max_depth: {dt_metrics_best_val}')

    return best_max_depth_tree, dt_metrics_best_val

# SVM --------------------------------------------------------------------------------------------
def modelo_svm(X_train, X_test, y_train, y_test):
    """
    Treina e ajusta um modelo SVM com ajuste de C e gamma via GridSearchCV.
    """

    param_grid = {
        'C': [10,100,1000],
        'gamma': [0.1,0.01,0.001], 
    }

    grid = GridSearchCV(SVC(kernel='rbf', random_state=0), 
                        param_grid=param_grid, 
                        cv=10, 
                        n_jobs=-1, 
                        verbose=0)


    grid.fit(X_train, y_train)

    best_svm = grid.best_estimator_

    y_pred_train = best_svm.predict(X_train)
    y_pred_test = best_svm.predict(X_test)

    Ein = 1 - accuracy_score(y_train, y_pred_train)
    Eout = 1 - accuracy_score(y_test, y_pred_test)

    print("SVM - Resultados")
    print(f"Melhores parâmetros: {grid.best_params_}")
    print(f"Erro de treino (Ein): {Ein:.4f}")
    print(f"Erro de teste (Eout): {Eout:.4f}")
    print(f"Número total de vetores de suporte: {sum(best_svm.n_support_)}")
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_test))

    plotar_matriz_confusao(y_test, y_pred_test, classes=np.unique(y_test))
    return best_svm


def plotar_matriz_confusao(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()

# REDE NEURAL --------------------------------------------------------------------------------------------
def modelo_rede_neural(X_train, X_test, y_train, y_test, input_dim, output_dim, n_neurons):
    
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    
    accuracies_train = []
    accuracies_val = []
    history_list = []
    ein_list = []
    eval_list = []

    i = 0
    for train_index, val_index in skf.split(X_train, y_train):
        i = i + 1
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # Criando a arquitetura da rede neural    
        model = Sequential()
        
                
        # Model 2
        model.add(Dense(units=n_neurons, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

        # Compilar o modelo
        
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])


        BATCH_SIZE = 256
        # Treina o modelo
        history = model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=100, batch_size=BATCH_SIZE, verbose=0)
        history_list.append(history) 
        


        # Avalia o modelo
        E_in, accuracy_train = model.evaluate(X_train_fold, y_train_fold, batch_size=BATCH_SIZE, verbose=0)
        E_val, accuracy_val = model.evaluate(X_val_fold, y_val_fold, batch_size=BATCH_SIZE, verbose=0)
        
        accuracies_train.append(accuracy_train)
        accuracies_val.append(accuracy_val)
        ein_list.append(E_in)
        eval_list.append(E_val)    
        
        # # Exibe o historico de treinamento para um fold especifico
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title(f'Metrica de erro - Fold {i}')
        # plt.ylabel('Erro')
        # plt.xlabel('Epoca')
        # plt.legend(['Treinamento'])
        # plt.show()

        # print(f'--> Acuracia (treino): {accuracy_train:.4f}')
        # print(f'--> Acuracia (validacao): {accuracy_val:.4f}')
        # print(f"--> E_val - E_in = {E_val - E_in:.4f}")
        # print(f"--> acc_in - acc_val = {accuracy_train - accuracy_val:.4f}\n")    

    # Calcula a acuracia media
    avg_accuracy_train = np.mean(accuracies_train)
    avg_accuracy_val = np.mean(accuracies_val)
    avg_ein = np.mean(ein_list)
    avg_eval = np.mean(eval_list)

    # Historico com valores medios dos folds
    history_loss_avg = []
    history_val_loss_avg = []
    aux_list1 = []
    aux_list2 = []

    for i in range(len(history.history['loss'])):
        for j in range(len(history_list)):
            aux_list1.append(history_list[j].history['loss'][i])
            aux_list2.append(history_list[j].history['val_loss'][i])
        history_loss_avg.append(np.mean(aux_list1))
        history_val_loss_avg.append(np.mean(aux_list2))                            
                    
    plt.plot(history_loss_avg)
    plt.plot(history_val_loss_avg)
    plt.title('Metrica de erro - Media dos Folds')
    plt.ylabel('Erro')
    plt.xlabel('Epoca')
    plt.legend(['Treinamento','Validacao'])
    plt.show()

    print(f'--> Acuracia (treino): {avg_accuracy_train:.4f}')
    print(f'--> Acuracia (validacao): {avg_accuracy_val:.4f}')
    print(f"--> E_in = {avg_ein:.4f}")
    print(f"--> E_val = {avg_eval:.4f}")
    print(f"--> E_val - E_in = {avg_eval - avg_ein:.4f}")
    print(f"--> acc_in - acc_val = {avg_accuracy_train - avg_accuracy_val:.4f}\n")    

    # Obtendo a acuracia no conjunto de teste
    E_out, acc_test = model.evaluate(X_test, y_test, verbose=0)

    print(f"--> E_out = {E_out:.4f}")
    print(f'--> Acuracia (teste): {acc_test:.4f}')


    # Converte as previsões e os rótulos reais para classes binárias
    y_pred_classes = (model.predict(X_test) > 0.5).astype("int32").flatten()
    y_test_classes = y_test.flatten()

    print("\nRelatório de Classificação:")
    print(classification_report(y_test_classes, y_pred_classes))

    # Matriz de Confusão
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()
    return model