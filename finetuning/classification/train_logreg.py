import polars as pl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# from sklearn.metrics import balanced_accuracy_score
from sklearn.dummy import DummyClassifier


def load_dataframes():
    splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet', 'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
    train_df = pl.read_parquet('hf://datasets/stanfordnlp/imdb/' + splits['train'])
    test_df = pl.read_parquet('hf://datasets/stanfordnlp/imdb/' + splits['test'])

    # Split train_df into train and val (80/20)
    train_size = int(0.8 * len(train_df))
    df_train = train_df[:train_size]
    df_val = train_df[train_size:]
    df_test = test_df

    return df_train, df_val, df_test


def eval_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # make pred
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # accuracy and balanced accuracy
    accuracy_train = accuracy_score(y_train, y_pred_train)
    # balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)

    accuracy_val = accuracy_score(y_val, y_pred_val)
    # balanced_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    # balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)

    print(f"Training Accuracy: {accuracy_train * 100:.2f}%")
    print(f"Validation Accuracy: {accuracy_val * 100:.2f}%")
    print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

if __name__ == "__main__":
    df_train, df_val, df_test = load_dataframes()

    
    # text to bag-of-words model
    vectorizer = CountVectorizer()
    

    X_train = vectorizer.fit_transform(df_train["text"].to_list())
    X_val = vectorizer.transform(df_val["text"].to_list())
    X_test = vectorizer.transform(df_test["text"].to_list())
    y_train, y_val, y_test = df_train["label"].to_list(), df_val["label"].to_list(), df_test["label"].to_list()

    # model training and eval
    # dummy classifier with the strategy to predict the most frequent class
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)

    print("Dummy classifier:")
    eval_model(dummy_clf, X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n\nLogistic regression classifier:")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    eval_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
