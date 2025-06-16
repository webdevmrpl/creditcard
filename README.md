# Credit Card Fraud Detection

## Dataset

Projekt wykorzystuje dataset **Credit Card Fraud Detection** dostępny na Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset zawiera transakcje kartami kredytowymi dokonane przez europejskich posiadaczy kart w ciągu dwóch dni we wrześniu 2013 roku. Ze względów poufności, większość cech została przekształcona za pomocą analizy głównych składowych (PCA).

### Charakterystyka danych:
- **284,807 transakcji** z informacjami o kartach kredytowych
- **30 cech** (V1-V28 to cechy po transformacji PCA, Time, Amount, Class)
- **Klasy niezbalansowane**: 99.83% prawidłowych transakcji (492 oszustwa z 284,807 transakcji)
- **Cechy numeryczne**: wszystkie wartości wejściowe są numeryczne
- **Brak wartości brakujących**


## Opis Projektu

Projekt **Credit Card Fraud Detection** to kompleksowy system wykrywania oszustw w transakcjach kartami kredytowymi wykorzystujący techniki uczenia maszynowego. System implementuje różne algorytmy klasyfikacji oraz sieci neuronowe do identyfikacji potencjalnie podejrzanych transakcji.

## Główne Funkcjonalności

- **Preprocessing danych** za pomocą sklearn pipelines
- **Balansowanie datasetu** przez undersampling
- **Trenowanie wielu modeli** (Logistic Regression, Random Forest, Gradient Boosting, SVM, Neural Networks)
- **Ewaluacja modeli** z wykorzystaniem ROC curves i metryk wydajności
- **Modułowa architektura** z reuzywalnym kodem

## Struktura Projektu

```
project/
├── README.md                           # Dokumentacja projektu
├── dataloader.py                       # Klasa do ładowania i preprocessingu danych
├── credit_card_fraud_detection.ipynb   # Główny notebook z analizą
├── Detecting_Credit_Card_Fraud.ipynb  # Oryginalny notebook
├── creditcard.csv                      # Dataset (nie wersjonowany)
└── shallow_nn_b.keras                 # Wytrenowany model sieci neuronowej
```


## Dataset

Projekt wykorzystuje dataset **Credit Card Fraud Detection** zawierający:

- **284,807 transakcji** z informacjami o kartach kredytowych
- **30 cech** (V1-V28 to cechy po transformacji PCA, Time, Amount, Class)
- **Klasy niezbalansowane**: 99.83% prawidłowych transakcji, 0.17% oszustw


### Pipeline Preprocessingu

1. **MinMaxScaler** dla kolumny 'Time' (normalizacja 0-1)
2. **RobustScaler** dla kolumny 'Amount' (odporna na outliers)
3. **Passthrough** dla pozostałych cech (V1-V28, Class)

## Modele Implementowane

- **Logistic Regression** - logistic regression
- **Random Forest** - ensemble method
- **Gradient Boosting** - boosting algorithm
- **Linear SVC** - support vector classifier
- **Shallow Neural Network** (2 hidden units + BatchNorm)

## Metryki Ewaluacji

### Podstawowe Metryki
- **Accuracy** - dokładność ogólna
- **Precision** - precyzja dla klasy oszustw
- **Recall** - czułość dla klasy oszustw
- **F1-Score** - średnia harmoniczna precision i recall
- **AUROC** - Area Under ROC Curve

## Przykłady Wyników

### Zbalansowany Dataset
```
Model                Validation AUROC    Test AUROC
Logistic Regression     0.945             0.923
Random Forest           0.912             0.905
Gradient Boosting       0.934             0.918
Linear SVM              0.941             0.925
Neural Network          0.952             0.930
```

### Niezbalansowany Dataset
```
Model                Validation AUROC    Test AUROC
Logistic Regression     0.974             0.970
Random Forest           0.968             0.965
Gradient Boosting       0.971             0.968
Linear SVM              0.975             0.972
```