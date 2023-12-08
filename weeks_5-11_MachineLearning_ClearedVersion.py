#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

assert sys.version_info >= (3, 7)


# <b><p>Student ID: 2661047</p></b>
# <b><a href="https://github.com/2661047/2661047.github.io">Github URL</a></b><p> or if you aren't able to select it, it is below.</p>
# <p>https://github.com/2661047/2661047.github.io</p>

# In[ ]:


from packaging import version #import the package "version"
import sklearn # import scikit-learn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")


# In[ ]:


import matplotlib.pyplot as plt

plt.rc('font', size=14) #general font size
plt.rc('axes', labelsize=14, titlesize=14) #font size for the titles of x and y axes
plt.rc('legend', fontsize=14) # font size for legends
plt.rc('xtick', labelsize=10) # the font size of labels for intervals marked on the x axis
plt.rc('ytick', labelsize=10) # the font size of labels for intervals marked on the y axis


# In[ ]:


from pathlib import Path

IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# <hr>

# ![A view of the University of Glasgow](glasgow_view.jpg)
# [A link to the University of Glasgow](https://www.gla.ac.uk)

# ---

# <p>In the above code I have imported sys, sklearn, matplotlib.pyplot and pathlib. Sys is used to check that the version of Python is greater than 3.7. Sklearn provides tools for machine-learning and modelling. Matplotlib.pyplot allows the user to make changes to plots, in this case it is used to make changes to the output of code. For example, changing the font size. Pathlib is used to better the interaction and retrieval of files and paths. In this code pathlib is defining a function and then saving a matplotlib to a file - whilst also controlling the output with the use of matplotlib. The code above doesn't use any machine learning, however, Sklearn was imported to the code and can be used for machine learning in relation to clustering, classifying and investigating relationships. Using this code, I was able to import an image of the University of Glasgow, as well as providing a link their website - as well as how it is formatted. </p>

# <p>This code may also be used by others in various proffessions, one example may be with museum curators. This code can be beneficial in digitising a collection and better educating those with an interest, analysing large data sets within the museums collection and detecting trends and patterns, collaborating with other museums to further the development and understanding of the past, and also marketing the museum exhibit to the public and promoting further visits from the public. </p>

# ### Reflection
# <ol>
#     <li>I originally had difficulty in inputing an image - In my attempt I was trying to put the code in a code cell - rather than a markdown cell. This was resolved by looking at what the code was doing and seeing what I was trying to make it do. </li>
#     <li>When advising others on how to learn this topic, I would advise using w3schools. It is a useful tool in understanding what the code is wanting to do and what each part of the code represents. 
#     <a href="https://www.w3schools.com">A link to w3schools.</a></li>
#     <li>When increasing engagement, changing fonts - be it with colour or sixe, as well as the use of images, combine to make the display more attractive. This encourages user engagement and may thus encourage the attractiveness of learning code. </li>
#     <li>Another tip for new users to help learning code may be to carefully review code if errors occur - when copying this code over to this notebook I missed a colon and a comma which meant the code wouldn't run. Fortunately, Jupyter Notebook showed me where the possible errors were, however, if coding on another text editor this error may not be shown and it may be difficult to find where the issue is. </li>
# </ol>

# <hr>

# <h3>Understanding how framing the problem affects data selection</h3>
# <b><a href="https://link.springer.com/article/10.1007/s10506-021-09306-3">Rethinking the field of automatic prediction of court decisions</a></b>
# <br>
# <b><a href="https://www.bbc.co.uk/news/technology-67022005">AI facial recognition: Campaigners and MPs call for ban</a></b>
# <p><u>The importance of framing your problem precisely</u></p>
# <p>It is important to be concerned with how you fram a problem. As depending on how, and too what extent, a problem has been framed can alter the perception and understanding when viewing the code and its output. Whether its a piece of written text to explain the problem, or the use of '#' alongside the code. It should provide a clear explanation of the problem and specify what data is required to use. By using and inputting data sources that are not relevant this can alter the output significantly.
# <br>
#         An example may be in the legislative process, looking at the "Rethinking the field of automatic prediction of court decisions" paper, it discusses the large pool of information that can be gathered internationally to help better decision making in the legislative process and see if machine learning can get results close to that of human decision making. To check the accuracy of various models, they were using F1-scores, the Lage-Freitas et al. (2019) on Brazilian appeal cases and Bertalan and Ruiz (2020) on São Paulo Justice Court cases achieved an F1-score of 79% and up to 98%. This shows the models to be successful, however, if too much data was used because the creator poorly defined the problem, the model may provide a more varied result that shows less accuracy in answering the problem. </p>
# <p><u>Explaining how the machine learning model will be used down the road?</u></p>
# <p>By clearly framing the problem it can create a better understanding of how to use the data and information in the code for further steps down the road. For example, in the case of AI facial recognition, a large data set that hasn't been clearly defined may impact future steps. By clearly framing the problem and explaining what the use of the machine learning model does allows for its use to be relevent for its specific purporse. With AI facial recognition it will have large amounts of biometric data that, with the use of machine learning, will be able to recognise features at a distance and compile small differences to a match that recognises a person. By explaining what the machine learning model will do and its purpose can allow people to understand what its doing, and thus generate an opinion on its relevance and applicability. With AI facial recognition, its implementation on the streets was prevented as people preffered their privacy. Therefore, ethically it may be important to explain the use of a model and being transparent with its purpose. There may also be legislative and regulatory requirements to comply with, as such with a breech of privacy with AI facial recognition. </p>

# <h3>How to select your algorithm?</h3>
# <br>
# <p>Regression and classification are two types of supervised learning algorithms. Regression is used to predict outputs that are continuous, whilst classification is used for finite possible outcomes.</p>
# <ol>
#     <li>Would regression or classification fit better for predicting median housing prices?
#         <br>
#         Regression would be used for predicting median housing prices as this is continuous - it would provide an output that reflects the predicted pricing of house prices.
#     </li>
#     <li>Would regression or classification fit better for handwritten digit recognition?
#         <br>
#         As there are limited options, 0-9, a classification model would be preferable in recognising handwritten digits. An F1-score may be beneficial when doing this as it may measure the accuracy of the model
#     </li>
# </ol>

# <h3>Before data collection</h3>
# <br>
# <ol>
#     <li>what kind of information about housing in a district you think would help predict the median housing price in the district? 
#         <br>
#         When wanting to create a machine learning model that predicts the median housing price within a district, data may need to be collected. Data on the current median housing price, as well as the median housing price within the chosen district at different points in the past, may both be useful in wanting to predict future trends. Measuring the percentage increase over time and applying that to the future to recognise trends. It may also be useful to collect data on nearby districts in order to understand any annomolies or differences that must be accounted for. It may further be beneficial to discover external forces trends that may affect housing prices. When all this is combined, it may be possible to create an accurate prediction of future median housing prices in a district.
#     </li>
#     <li>How might these decisions depend on geographical and/or cultural differences and how the information you collect would already bias the data?
#         <br>
#         Results may vary depending on geographical and/or cultural differences. Different regions and cultures are growing at different rates, rates of population growth - either naturally or by people entering a region may differ depending on geographical and cultural factors. A district within a good geographical location may attract more people to move to a region or encourage more people to create families in that district - be it for jobs, schools, shops, whether its rural or urban. There are various factors and each may impact the differences in the rate of growth of median housing prices. Similarly with cultural differences, this may affect the rate by which housing prices increase. A more liberal culture may have legislation that caps the rate of growth per annum, whilst others may allow demand to control the housing prices over time.
#         <br>
#         A personal bias may be unknowingly applied, as depending on where someone grows up and learns may impact what data sets may be counted as relevant. As previously stated, different cultures may have different ways of allowing housing prices to increase over time, these biases may be brought in and may result in either a lack of data being applied to a model - or too much. 
#     </li>
# </ol>

# <hr>

# <h3>Downloading relevant data and having a look</h3>

# In[ ]:


from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data(): #defines a function that loads the housing data available as .tgz file on a github URL
    tarball_path = Path("datasets/housing.tgz") # where you will save your compressed data
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True) #create datasets folder if it does not exist
        url = "https://github.com/ageron/data/raw/main/housing.tgz" # url of where you are getting your data from
        urllib.request.urlretrieve(url, tarball_path) # gets the url content and saves it at location specified by tarball_path
        with tarfile.open(tarball_path) as housing_tarball: # opens saved compressed file as housing_tarball
            housing_tarball.extractall(path="datasets") # extracts the compressed content to datasets folder
    return pd.read_csv(Path("datasets/housing/housing.csv")) #uses panadas to read the csv file from the extracted content

housing = load_housing_data() #runsthe function defined above


# In[ ]:


housing.info()


# In[ ]:


housing["ocean_proximity"].value_counts() # tells you what values the column for `ocean_proximity` can take


# In[ ]:


housing.hist(bins=50, figsize=(12, 8))
save_fig("attribute_histogram_plots")  # extra code
plt.show()


# In[ ]:


housing.describe()


# In[ ]:


from sklearn.datasets import fetch_openml
import pandas as pd

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')

mnist_dataframe = pd.DataFrame(data=mnist.data, columns=mnist.feature_names) #Is this relevant or should be hashtagged?


# In[ ]:


print(mnist.DESCR)


# <h3>Review of the data description</h3>
# <br>
# <ol>
#     <li>What is the size of each image?
#         <br>
#         "The images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field."
#     </li>
#     <li>What they did do?
#         <br>
#         They mixed NIST's datasets - "SD-3 was collected among Census Bureau employees", whilst "SD-1 was collected among high-school students" and drew "sensible conclusions from learning experiments" and "test among the complete set of samples." They also scrambled this data and then reorganised this data. They then divided the 60,000 large dataset in two, a training set and a test set. They were therefore able to have two datasets of 30,000 examples to make an assessment of 60,000 training patterns.
#     </li>
#     <li>Why do you think they did this?
#         <br>
#         They likely gathered data from two sources to provide more variability to the data set, and with more differences it can allow to create a better average and estimation. Scambling the data further allows the removal of possible trends in data collection that may negatively impact the model. By dividing it into two and by applying different tests to each, it can once again allow for better results with a large data set contributing to information and different methods to analyse different trends and results among a large data set.
#     </li>
#     <li>Was it justified?
#         <br>
#         Though an extensive process, this process is justifiable wanting to achieve the most accurate results.
#     </li>
# </ol>

# In[ ]:


mnist.keys()


# In[ ]:


images = mnist.data
categories = mnist.target
print("images", images)
print("categories:", categories)


# In[ ]:


import matplotlib.pyplot as plt

## the code below defines a function plot_digit. The initial key work `def` stands for define, followed by function name.
## the function take one argument image_data in a parenthesis. This is followed by a colon. 
## Each line below that will be executed when the function is used. 
## This cell only defines the function. The next cell uses the function.

def plot_digit(image_data): # defines a function so that you need not type all the lines below everytime you view an image
    image = image_data.reshape(28, 28) #reshapes the data into a 28 x 28 image - before it was a string of 784 numbers
    plt.imshow(image, cmap="binary") # show the image in black and white - binary.
    plt.axis("off") # ensures no x and y axes are displayed


# In[ ]:


# visualise a selected digit with the following code

some_digit = mnist.data[0]
plot_digit(some_digit)
plt.show()


# <hr>

# <h3>Setting Aside the Test data</h3>

# <p>Splitting the data. 20% for testing and 80% for training. Using the function train_test_split</p>
# <br>
# <p>Putting a number in random_state, (42), gives the same split each time with the same data set</p>

# In[ ]:


from sklearn.model_selection import train_test_split

tratio = 0.2 #to get 20% for testing and 80% for training

train_set, test_set = train_test_split(housing, test_size=tratio, random_state=42) 
## assigning a number to random_state means that everytime you run this you get the same split, unless you change the data.


# <p>Data sets may not be appropriately weighted and representative - for example the females represent 51.1% of the US population, so the following is the probability of getting a sample with less than 48.5% or greater than 53.5% females if you take a random sample withoput stratifying.</p>

# In[ ]:


# extra code – shows another way to estimate the probability of bad sample

import numpy as np

sample_size = 1000
ratio_female = 0.511

np.random.seed(42)

samples = (np.random.rand(100_000, sample_size) < ratio_female).sum(axis=1)
((samples < 485) | (samples > 535)).mean()


# <p>The following adds a column to the housing data to create bins of data according to interval brackets of median income of districts. This is a first step to creating a stratified sample across different income brackets.</p>

# In[ ]:


import numpy as np
import pandas as pd

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


# <p>The following code uses the above bins to implement startified sampling - that is, it will randomly sample 20% (because we set test ratio tratio to 0.2) from each income bracket defined above.</p>

# In[ ]:


from sklearn.model_selection import train_test_split

tratio = 0.2 #to get 20% for testing and 80% for training

strat_train_set, strat_test_set = train_test_split(housing, test_size=tratio, stratify=housing["income_cat"], random_state=42)


# <p>The code below prints out the proportion of each income category in the stratified test set above.</p>

# In[ ]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set) #Prints out in order of the highest proportion first.


# <h3>Why a stratified sample based on median income is reasonable</h3>
# <br>
# <ol>
#     <li>By looking at the median rather than the mean it may be less sensitive to random data and makes sure that the sample reflects the distribution of income across a population with better accuracy.</li>
#     <li>It may be beneficial in interpreting the data, by seeing how the model performs across different groups it can help in identifying biases and room for improvement.</li> 
#     <li>Furthermore, by stratifying a sample based on median income, it may be possible to see variations in the data and create a more robust model that works across different economic, geographical and cultural regions.</li> 
#     <li>By stratifying, it removes the possibility of biases as it allows for the model to be more representative. It can prevent a sample being dominated by a single income group but rather showing the whole result in different groups.</li>
# </ol>

# In[ ]:


type(mnist.data)


# In[ ]:


X_train = mnist.data[:60000]
y_train = mnist.target[:60000]

X_test = mnist.data[60000:]
y_test = mnist.target[60000:]


# <hr>

# <h3>Selecting and Training a Model</h3>

# <p>Assigning a copy of the stratified training set we created earlier to the variable housing</p>

# In[ ]:


housing = strat_train_set.copy()


# In[ ]:


corr_matrix = housing.corr(numeric_only=True) # argument is so that it only calculates for numeric value features
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[ ]:


from pandas.plotting import scatter_matrix

features = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[features], figsize=(12, 8))
def save_fig(scatter_matrix_plot):  
    save_fig("scatter_matrix_plot") #The hashtag was before this line. Removing it the first time took me through several steps to remove errors and this is the result.

#The line above is extra code you can uncomment (remove the hash at the begining) to save the image.
#But, to use this, make sure you ran the code at the beginning of this notebook defining the save_fig function

    plt.show()


# In[ ]:


housing = strat_train_set.drop("median_house_value", axis=1) ## 1)
housing_labels = strat_train_set["median_house_value"].copy() ## 2)


# In[ ]:


print(housing)


# In[ ]:


print(housing_labels)


# In[ ]:


housing.info()


# <p>In the above code, there are 16,512 entries and this is the same for 0,1,2,3,5,6,7,8,9 - however, 4 provided the result of 16,344 non-null entries. This means that there are 168 missing values for total_bedrooms</p>

# <p>(Option 1) Drop the row with missing value. This causes you to lose data points. In our scenario with the housing data, 168 rows will be removed.
#     <br>
#     (Option 2) Drop the column with missing values. This causes you to lose one of your features.
#     <br>
#     (Option 3) Fill in the missing value with some value such as the median or mean or fixed value that makes sense. This is called imputing.
#     <br>
#     <b>You can also use SimpleImputer from the sklearn.impute library to fill missing values with the median</b>
# </p>

# In[ ]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median") # initialises the imputer

housing_num = housing.select_dtypes(include=[np.number]) ## includes only numeric features in the data

imputer.fit(housing_num) #calculates the median for each numeric feature so that the imputer can use them

housing_num[:] = imputer.transform(housing_num) # the imputer uses the median to fill the missing values and saves the result in variable X

housing_num.info()


# In[ ]:


housing_num.describe()


# In[ ]:


from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)


# In[ ]:


housing_num[:]=std_scaler.fit_transform(housing_num)


# In[ ]:


print(housing_num)


# In[ ]:


from sklearn.preprocessing import StandardScaler #This line is not necessary if you ran this prior to running this cell. 
#We are however including it here for completeness sake.

target_scaler = StandardScaler() #instance of Scaler
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame()) #calculate the mean and standard deviation and use it to transform the target labels.


# In[ ]:


from sklearn.linear_model import LinearRegression #get the library from sklearn.linear model

model = LinearRegression() #get an instance of the untrained model
model.fit(housing_num, scaled_labels)
#model.fit(housing[["median_income"]], scaled_labels) #fit it to your data
#some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

#scaled_predictions = model.predict(some_new_data)
#predictions = target_scaler.inverse_transform(scaled_predictions)


# In[ ]:


some_new_data = housing_num.iloc[:5] #pretend this is new data
#some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)


# In[ ]:


print(predictions, housing_labels.iloc[:5])


# In[ ]:


# extra code – computes the error ratios discussed in the book
error_ratios = housing_predictions[:5].round(-2) / housing_labels.iloc[:5].values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))


# <p>As 'housing_predictions' has not been defined - this code fails to work.</p>

# In[ ]:


from sklearn.model_selection import cross_val_score

rmses = -cross_val_score(model, housing_num, scaled_labels,
                              scoring="neg_root_mean_squared_error", cv=10)


# In[ ]:


pd.Series(rmses).describe()


# <hr>

# <h3>Handwritten Digit Classification</h3>

# <p>Retrieve the necessary data</p>

# In[ ]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist.load_data()


# <p>Reviewing the data</p>

# In[ ]:


print(type(mnist))


# <p>Getting the data from a and b</p>

# In[ ]:


(X_train_full, y_train_full), (X_test, y_test) = mnist 
# (X_train_full, y_train_full) is the 'tuple' related to `a` and (X_test, y_test) is the 'tuple' related to `b`.
# X_train_full is the full training data and y_train_full are the corresponding labels 
# - labels indicate what digit the image is of, for example 5 if it is an image of a handwritten 5.


# <p>Scaling the Pixel values to a specified range</p>

# In[ ]:


X_train_full = X_train_full / 255.
X_test = X_test / 255.


# <p>Splitting the training data into training and validation data</p>

# In[ ]:


X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


# <p>Using a numpy library to increase dimensions to include colour channels</p>

# In[ ]:


import numpy as np # you won't need to run this line if you ran it before in this notebook. But for completeness.

X_train = X_train[..., np.newaxis] #adds a dimension to the image training set - the three dots means keeping everything else the same.
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# <p>Building the neural networks and fitting it to the data</p>

# In[ ]:


tf.keras.backend.clear_session()

tf.random.set_seed(42)
np.random.seed(42)

# Unlike scikit-learn, with tensorflow and keras, the model is built by defining each layer of the neural network.
# Below, everytime tf.keras.layers is called it is building in another layer

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", 
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
#numbers at the bottom tell you how many parameters need learning in this model


# In[ ]:


model.summary() # not necessary for the machine learning task.


# <p>Training and evaluating the model</p>

# In[ ]:


model.evaluate(X_test, y_test)


# <p>Comparing with another model - using a Stochastic Gradient Decent Classifier. Applying the stochastic gradient descent optimiser (cf. the nadam optimiser used with the CNN above) with any number of algorithms but by default it applies it to a Support Vector Machine.</p>

# In[ ]:


# getting the data again from Scikit-Learn, so that we know the image dimens fit for the model!

from sklearn.datasets import fetch_openml
import pandas as pd

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')

# getting the data and the categories for the data
images = mnist.data
categories = mnist.target


# <p>Abbreviating and use the entire data and evaluate using cross validation</p>

# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

sgd_clf = SGDClassifier(random_state=42)

#cross validation on training data for fit accuracy

accuracy = cross_val_score(sgd_clf, images, categories, cv=10)

print(accuracy)


# <hr>

# <h2>Reflection</h2>

# <p><b>What would you need to do for your code if:</b></p>
# <br>
# <ul>
#     <li>Your were to use your own data (for example, discuss survey data data and photos)?
#     <br>
#         <code>import pandas as pd</code>
#             <br>
#         <code>import tarfile</code>
#             <br>
#         <code>import urllib.request</code>
#             <br>
#         <code>return pd.read_csv(Path("datasets/housing/housing.csv"))</code>
#             <br>
#          <p>The code above was used in the code in this notebook, "import urllub.request" is used to import the necessary data from the URL and saves it to a file path. It then uses a Tarfile to extract the contents and then used Pandas to read the file and return it as a Pandas DataFrame.
#             <br>
#             If i was to use my own data, rather than importing from a URL, I would redirect the code to import from a local path. The data would be stored on a plain text file, which the computer reads as a CSV file and returns it as a Pandas DataFrame.
#         </p>
#         <br>
#     </li>
#     <li>You were changing:
#     <br>
#     </li>
#     <li>Your model?
#     <br>
#         <p>The model used within this code of this notebook uses linear regression - 
#             <br>
#             <code>from sklearn.linear_model import LinearRegression </code>
#             <br>
#             to use a different model we would have to look at other possible models and calculate which is the optimal model to use for our purpose. Decision trees and neural networks are examples of other models that can be used, they may not work the best for this purpose but they are other options that may provide different results - this could be beneficial or unhelpful. Using sklearn again, you can import these other models.
#         </p>
#         <br>
#     </li>
#     <li>Your scaling method?
#     <br>
#         <p>The following code was used to to change the scaling of pixel values to a specified range - </p>
#         <br>
#            <code>X_train_full = X_train_full / 255.</code>
#         <br>
#            <code>X_test = X_test / 255.</code>
#         <br>
#         <p>this can be changed by altering the numbers "255".</p>
#         <br>
#         <p>Rescaling also applies to how the data is fed into the model - this was done in this notebook with min-max scaling.
#         </p>
#         <br>
#         <code>from sklearn.preprocessing import MinMaxScaler</code>
#         <br>
#         <code>min_max_scaler = MinMaxScaler(feature_range=(-1, 1))</code>
#         <br>
#         <code>housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)</code>
#         <br>
#         <p> This code scales to a range of "-1, 1" but these numbers could be changed.</p>
#     </li>
#     <li>Your approach to handling missing data?
#     <br>
#         <p>If there is missing data, as seen with data used in this notebook, you can use SimpleImputer from the Sklearn.impute library to fill missing values with the median.</p>
#         <br>
#         <code>from sklearn.impute import SimpleImputer</code>
#         <br>
#         <code>imputer = SimpleImputer(strategy="median")</code>
#         <br>
#         <code>housing_num = housing.select_dtypes(include=[np.number])</code>
#         <br>
#         <code>imputer.fit(housing_num)</code>
#         <br>
#         <code>housing_num[:] = imputer.transform(housing_num)</code>
#         <br>
#         <p>Other ways may be to remove the dataset that has missing data. Another method may be to use the mean rather than the median, but this assumes data follows a normal distribution but it can be easily influenced by random data.</p>
#     </li>
# </ul>
# <br>
# <p><b>What is the significance of cross validation?</b></p>
# <br>
#     <p>It may be important to perform cross validation of machine learning models and compare results. Firstly, it may be beneficial in identifying the quality of a model used. Sometimes a couple models may be appropriate, however, they may produce different outputs. By doing cross validation it may assist in identifying if the best model was used. Secondly, cross validation can show the performance of a model and its ability to be used again for different data. Thirdly, by cross validating, it can present errors in the code that may allow for further improvement of the code. Cross validating was performed in this notebook with the following code:</p>
#     <br>
#     <code>from sklearn.model_selection import cross_val_score</code>
#     <br>
#     <code>rmses = -cross_val_score(model, housing_num, scaled_labels,
#     scoring="neg_root_mean_squared_error", cv=10)</code>
#     <br>
#     <code>pd.Series(rmses).describe()</code>
#     <p>This code can cross validate in different ways, but the selected method here performs 10-fold cross validation.</p>

# <hr>

# <h3>Tensorflow Playground</h3>
# <br>
# <b><a href="https://playground.tensorflow.org/">Tensorflow Playground</a></b>
# <br>
# <p>In my attempt at using Tensorflow Playground, I am using a spiral neural network to attempt to bring the training loss down to, or below, 0.2. I successfully did this by having two features, x1 and x2, six hidden layers, going in the order of 6 neurons, 3 neurons, 3 neurons, 4 neurons, 2 neurons, then 3 neurons. I achieved a training loss of 0.187 after 4,923 Epochs.</p>
# <br>
# <p>In my second attempt with two featurs, one layer, and eight neurons, I achieved a training loss of 0.197 after 1,806 Epochs</p>
# <br>
# <p>In the third attempt with two features, eight layers, and eight neurons, I achieved a training loss of 0.2 after 412 Epochs</p>
# <br>
# <p>From my multiple tests, the fastest in getting to the desired result of 0.2 was my third attempt.</p>
# <br>
# <p><b>Examing the patterns displayed in the neural network nodes, what kinds of patterns the neural network might be learning at different layers and nodes</b></p>
# <br>
# <p>A visual representation may be beneficial in showing you the wheights and what has been learned through the process of a neural network. It may further show what the nodes process through each stage, giving greater detail of the network processing and of the features being processed.</p>

# <hr>

# <h3>Exploring the pre-trained model VGG-19</h3>

# In[ ]:


from keras.applications.vgg19 import VGG19

model = VGG19() ### this will take some time!!


# In[ ]:


print(model.summary())


# <hr>

# In[ ]:




