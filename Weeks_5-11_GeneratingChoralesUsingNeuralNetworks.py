#!/usr/bin/env python
# coding: utf-8

# <H1>Generating Chorales With Neural Networks</H1>

# <h3>Explaining the code</h3>
# <br>
# <p>The following code produces music based on Bach Chorales - this code can used for various purposes:</p>
# <ol>
#     <li>It can be used to inspire further musical compositions - providing inspiration to musicians,</li>
#     <li>The code can be used to educate others, teaching people about the aspects of music creation,</li>
#     <li>Another purpose of this code may be for educational purposes, by creating code that gives the individual control over music creation can allow for an analysis of music generation.</li>
#     <li>Finally, the code may also be used for leisure - allowing those with a passion for coding and music creation to create music without having to play a musical instrument.</li>
# </ol>
# <br>
# <p>The following code imports the necessary libraries: tensorflow, pathlib, pandas, keras, and numpy. Retrieving the files needed, the code then trains, validates and tests the sets. From this the neural network is created - in this code, an LSTM (Long Short-Term Memory) neural network has been used. This model is then trained on the training sets and evaluates the performance of the of the validation set. This then generates various Chorales that are dependant on various changes made in the code. Once the model has been taught ad the Chorales have been created, the code then plays the Chorales.</p>
# <br>
# <p>With an understanding of thise code, adaptations may be made that allow for the code to function differently.</p>
# <br>
# <p>Throughout this code I will use Hugging Chat if necessary to assist in the descriptions and understanding of the code.</p>

# <hr>

# <h3>Getting the Data</h3>

# In[1]:


import tensorflow as tf

tf.keras.utils.get_file(
    "jsb_chorales.tgz",
    "https://github.com/ageron/data/raw/main/jsb_chorales.tgz",
    cache_dir=".",
    extract=True)


# <p>The above code imports tensorflow and retrieves the file "jsb_chorales.tgz"</p>

# In[2]:


from pathlib import Path

jsb_chorales_dir = Path("datasets/jsb_chorales")
train_files = sorted(jsb_chorales_dir.glob("train/chorale_*.csv"))
valid_files = sorted(jsb_chorales_dir.glob("valid/chorale_*.csv"))
test_files = sorted(jsb_chorales_dir.glob("test/chorale_*.csv"))


# The above code seperates the data into "train", "valid" and "test" files

# In[5]:


import pandas as pd

def load_chorales(filepaths):
    return [pd.read_csv(filepath).values.tolist() for filepath in filepaths]

train_chorales = load_chorales(train_files)
valid_chorales = load_chorales(valid_files)
test_chorales = load_chorales(test_files)


# The above code reads the files that were previously seperated and stores the data in the "train", "valid", and "test" chorales

# <hr>

# <h3>Preparing the Data</h3>

# In[22]:


notes = set()
for chorales in (train_chorales, valid_chorales, test_chorales):
    for chorale in chorales:
        for chord in chorale:
            notes |= set(chord)

n_notes = len(notes)
min_note = min(notes - {0}) #0 denotes no notes being played
max_note = max(notes)

assert min_note == 36
assert max_note == 81


# <p>The above code defines the minimum and maximum notes in the set. Asserting the minimum note to 36 and the maximum note to 81.</p>

# <h3>Code for Synthesiser</h3>
# <br>
# <p>The following code is used for a synthesiser to play MIDI and is used to listen to the results.</p>
# <br>
# <p>The code also creates an output to test if it works.</p>

# In[23]:


from IPython.display import Audio
import numpy as np

def notes_to_frequencies(notes):
    # Frequency doubles when you go up one octave; there are 12 semi-tones
    # per octave; Note A on octave 4 is 440 Hz, and it is note number 69.
    return 2 ** ((np.array(notes) - 69) / 12) * 440

def frequencies_to_samples(frequencies, tempo, sample_rate):
    note_duration = 60 / tempo # the tempo is measured in beats per minutes
    # To reduce click sound at every beat, we round the frequencies to try to
    # get the samples close to zero at the end of each note.
    frequencies = (note_duration * frequencies).round() / note_duration
    n_samples = int(note_duration * sample_rate)
    time = np.linspace(0, note_duration, n_samples)
    sine_waves = np.sin(2 * np.pi * frequencies.reshape(-1, 1) * time)
    # Removing all notes with frequencies ≤ 9 Hz (includes note 0 = silence)
    sine_waves *= (frequencies > 9.).reshape(-1, 1)
    return sine_waves.reshape(-1)

def chords_to_samples(chords, tempo, sample_rate):
    freqs = notes_to_frequencies(chords)
    freqs = np.r_[freqs, freqs[-1:]] # make last note a bit longer
    merged = np.mean([frequencies_to_samples(melody, tempo, sample_rate)
                     for melody in freqs.T], axis=0)
    n_fade_out_samples = sample_rate * 60 // tempo # fade out last note
    fade_out = np.linspace(1., 0., n_fade_out_samples)**2
    merged[-n_fade_out_samples:] *= fade_out
    return merged

def play_chords(chords, tempo=160, amplitude=0.1, sample_rate=44100, filepath=None):
    samples = amplitude * chords_to_samples(chords, tempo, sample_rate)
    if filepath:
        from scipy.io import wavfile
        samples = (2**15 * samples).astype(np.int16)
        wavfile.write(filepath, sample_rate, samples)
        return display(Audio(filepath))
    else:
        return display(Audio(samples, rate=sample_rate))

## testing the synthesiser
for index in range(3):
    play_chords(train_chorales[index])


# In[24]:


import tensorflow as tf

def create_target(batch):
    X = batch[:, :-1]
    Y = batch[:, 1:] # predict next note in each arpegio, at each step
    return X, Y

def preprocess(window):
    window = tf.where(window == 0, window, window - min_note + 1) # shift values
    return tf.reshape(window, [-1]) # convert to arpegio

def bach_dataset(chorales, batch_size=32, shuffle_buffer_size=None,
                 window_size=32, window_shift=16, cache=True):
    def batch_window(window):
        return window.batch(window_size + 1)

    def to_windows(chorale):
        dataset = tf.data.Dataset.from_tensor_slices(chorale)
        dataset = dataset.window(window_size + 1, window_shift, drop_remainder=True)
        return dataset.flat_map(batch_window)

    chorales = tf.ragged.constant(chorales, ragged_rank=1)
    dataset = tf.data.Dataset.from_tensor_slices(chorales)
    dataset = dataset.flat_map(to_windows).map(preprocess)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(create_target)
    return dataset.prefetch(1)


# <p>The above code prepares the data for the neural network model and processes the data.</p>

# In[25]:


train_set = bach_dataset(train_chorales, shuffle_buffer_size=1000)
valid_set = bach_dataset(valid_chorales)
test_set = bach_dataset(test_chorales)


# <p>The above code uses the "bach_dataset" function which was previously defined and creates three datasets and defines the shuffle_buffer_size to 1000.</p>
# <br>
# <p>In my first run of the code, I forgot to input the first data block in the "Preparing the Data" section. The above  was not working as min_note was not defined, after doing research and attempting to use Hugging Chat - I added "min_note = 0". However, after review, I found that to resolve this issue I had to isnert the block of data.</p>

# <h3>Building the Model</h3>

# In[26]:


n_embedding_dims = 5

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_notes, output_dim=n_embedding_dims,
                           input_shape=[None]),
    tf.keras.layers.Conv1D(32, kernel_size=2, padding="causal", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(48, kernel_size=2, padding="causal", activation="relu", dilation_rate=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(64, kernel_size=2, padding="causal", activation="relu", dilation_rate=4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(96, kernel_size=2, padding="causal", activation="relu", dilation_rate=8),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dense(n_notes, activation="softmax")
])


# <p>The above code defines a neural network model using TensorFlow Keras. It adds each layer to the model and applies filters to the data and uses LSTM to process the data.</p>

# In[27]:


model.summary()


# <p>The above codes presents the parameters of the machine learning model - giving the total, trainable and non-trainable paramaters.</p>

# <h3>Training the Model</h3>

# In[28]:


optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
model.fit(train_set, epochs=20, validation_data=valid_set)


# <p>The above code trains the neural network model on the training data for 20 epochs and evaluates its performance on the validation data.</p>
# <br>
# <p>Within the output it recommended to alter the code - where it says "tf.keras.optimizers.Nadam", it recommends to use "tf.keras.optimizers.legacy.Nadam". After trying this, it did not increase the rate the code ran the Epochs. Therefore, I stuck with the existing code.</p>

# <h3>Saving and Evaluating the Model</h3>

# In[29]:


model.save("my_bach_model", save_format="tf")
model.evaluate(test_set)


# <p>The above code saves the model to a file named "my_bach_model" and outputs an evaluation on the models performance - [0.6596127152442932, 0.8173877000808716]. The first part of this output represents the loss of the model on the test set, the lower the value, the better the model is at predicting classes. The second part represents the accuracy of the model on the test set, indicating the proportion of correctly predicted examples - whereby a higher value represents higher accuracy. </p>

# <h3>Generating Chorales</h3>

# In[30]:


def generate_chorale_v2(model, seed_chords, length, temperature=1):
    arpegio = preprocess(tf.constant(seed_chords, dtype=tf.int64))
    arpegio = tf.reshape(arpegio, [1, -1])
    for chord in range(length):
        for note in range(4):
            next_note_probas = model.predict(arpegio)[0, -1:]
            rescaled_logits = tf.math.log(next_note_probas) / temperature
            next_note = tf.random.categorical(rescaled_logits, num_samples=1)
            arpegio = tf.concat([arpegio, next_note], axis=1)
    arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)
    return tf.reshape(arpegio, shape=[-1, 4])


# <p>The code above defines a function called "generate_chorale_v2" - with the model, seed_chords,length and temperature (a factor which controls the randomness, which is set to 1). The code shapes the seed_chorcs to a matix of [1, -1] and ultimately returns the chorale as a tensor with a shape [-1, 4].</p>

# In[31]:


seed_chords = test_chorales[2][:8]
play_chords(seed_chords, amplitude=0.2)


# <p>The above code plays a chord using the function play_chord with the amplitude set to 0.2. The seed_chords variable contains the first 8 chords of the 2nd test chorale.</p>

# In[32]:


new_chorale_v2_cold = generate_chorale_v2(model, seed_chords, 56, temperature=0.8)
play_chords(new_chorale_v2_cold, filepath="bach_cold.wav")


# <p>This code above generates a new chorale with a length of 56 and a temperature of 0.8.</p>

# In[33]:


new_chorale_v2_medium = generate_chorale_v2(model, seed_chords, 56, temperature=1.0)
play_chords(new_chorale_v2_medium, filepath="bach_medium.wav")


# <p>The above code operates similarly to "bach_cold.wav" - however the temperature (randomness) has been increased to 1.0 - this results in the suggested name of "bach_medium.wav"</p>

# In[34]:


new_chorale_v2_hot = generate_chorale_v2(model, seed_chords, 56, temperature=1.5)
play_chords(new_chorale_v2_hot, filepath="bach_hot.wav")


# <p>Once again, the code above is similar to the previous two. However, the randomness has been increase to 1.5.</p>

# In[ ]:




