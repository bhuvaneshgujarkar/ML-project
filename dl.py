# 4
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load CSV dataset
df = pd.read_csv(r"C:\Users\bhuva\OneDrive\Desktop\mnist_dataset.csv")

# Separate features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# Normalize and reshape X
X = X.reshape(-1, 28, 28, 1) / 255.0

# One-hot encode labels
y = to_categorical(y, num_classes=10)

# Split into train/test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


# 3
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split

# Load your CSV
df = pd.read_csv(r"C:\Users\bhuva\OneDrive\Desktop\imdb_top_1000.csv")

# Get texts and labels
texts = df['review'].astype(str).values  # make sure all are strings
labels = df['sentiment'].values          # should be 0 or 1

# Tokenize the text
max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_len = 200
padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=32, input_length=max_len),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


# 2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load your local MNIST CSV
df = pd.read_csv(r"C:\Users\bhuva\OneDrive\Desktop\mnist_dataset.csv")

# Separate features and labels
X = df.iloc[:, 1:].values  # All pixel values
y = df.iloc[:, 0].values   # Labels (digits)

# One-hot encode the labels
y = to_categorical(y, num_classes=10)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # 784 pixels
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes (digits 0–9)
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# 1
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dummy data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = Sequential([
    Dense(32, activation='relu', input_shape=(20,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))



# 5
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load your local MNIST CSV
df = pd.read_csv(r"C:\Users\bhuva\OneDrive\Desktop\mnist_dataset.csv")

# Separate features and labels
X = df.iloc[:, 1:].values.astype('float32')  # pixel values
y = df.iloc[:, 0].values                     # digit labels

# Normalize pixel values
X = X / 255.0

# Reshape to (28, 28, 1) for CNN input
X = X.reshape(-1, 28, 28, 1)

# One-hot encode labels
y = to_categorical(y, num_classes=10)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))