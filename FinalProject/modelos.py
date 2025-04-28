# 1. Importação das bibliotecas necessárias
# Bibliotecas gerais
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
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

    #ccp_alphas são os valores de poda
    if len(ccp_alphas) > 20:
        ccp_alphas = np.linspace(min(ccp_alphas), max(ccp_alphas), 20)
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
    max_depth = [i for i in range(1, 15)]
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
        'C': [10],
        'gamma': [0.1] 
    }

    grid = GridSearchCV(SVC(kernel='rbf', random_state=0), 
                        param_grid=param_grid, 
                        cv=3, 
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
def modelo_rede_neural(X_train, X_test, y_train, y_test, input_dim, output_dim):
    """
    Treina um modelo de Rede Neural simples.
    """
    # Ajustar os rótulos para começar em 0
    # y_train = y_train - 1
    # y_test = y_test - 1
    # Modelo sequencial básico
    model = Sequential()

    # Adicionar camadas
    model.add(Dense(32, input_dim=X_train.shape[1],  activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax')) # softmax para multi-classe #sigmoid se binário

    sgd = optimizers.SGD(learning_rate=0.01)

    # Compilar
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Treinamento
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=256,
        # callbacks=[early_stopping],
        verbose=0
    )

    # Avaliações
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    Ein = 1 - train_acc
    Eout = 1 - test_acc

    print("Rede Neural")
    print(f"Ein: {Ein:.4f}")
    print(f"Eout: {Eout:.4f}")

    # Plotando curva de treino
    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='Loss treino')
    plt.plot(history.history['val_loss'], label='Loss validação')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss x Épocas')
    plt.show()

    return model