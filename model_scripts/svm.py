import numpy as np
import pandas as pd
import spacy

from sklearn.svm import LinearSVC

run_validation = True

# The large language model for word embeddings
nlp = spacy.load('en_core_web_lg')

# Load the data
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

# Text column will be used as our "text" for vectorization
text = train_df['text'].values.tolist()
text = [i.lower() for i in text]
test_text = test_df['text'].values.tolist()
test_text = [i.lower() for i in test_text]

# Temporarily disable other pipes and then create the vectors
# for each of the words in the dataset
with nlp.disable_pipes():
    vectors = np.array([nlp(t).vector for t in text])
    test_vectors = np.array([nlp(t).vector for t in test_text])

# Splitting up the data
x_train = vectors
y_train = train_df['target']

if run_validation:
    # Create the validation set
    data_size = int(x_train.shape[0])
    training_size = int(x_train.shape[0] * 0.9)
    s = np.arange(x_train.shape[0])

    x_train = x_train[s]
    y_train = y_train[s]

    partial_x_train = x_train[1:training_size]
    partial_y_train = y_train[1:training_size]
    x_val = x_train[training_size:data_size]
    y_val = y_train[training_size:data_size]

    # Create the model
    model = LinearSVC(random_state=1, dual=False)

    # Fit the model
    model.fit(partial_x_train, partial_y_train)

    # Model accuracy
    print(f'Model test accuracy: {model.score(x_val, y_val)*100:.3f}%')

else:
    # Predict on the test dataset
    model = LinearSVC(random_state=1, dual=False)
    result = model.predict(test_vectors)

    # Write the result to a csv
    test_df['target'] = result
    test_df = test_df.drop(columns=['keyword', 'location', 'text'])
    test_df.to_csv("submission_svc.csv")
