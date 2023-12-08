#!/usr/bin/env python
# coding: utf-8

# <b><p>Student ID: 2661047</p></b>
# <b><a href="https://github.com/2661047/2661047.github.io">Github URL</a></b><p> or if you aren't able to select it, it is below.</p>
# <p>https://github.com/2661047/2661047.github.io</p>

# <h2>Critically engaging with AI ethics</h2>

# <p>Checking the version of python used in this notebook</p>

# In[ ]:


import sys

assert sys.version_info >= (3, 7)


# <hr>

# <h3>Identifying biases and exploring the Jigsaw Toxic Comment Classification Challenge</h3>

# <b><a href="https://www.kaggle.com/code/alexisbcook/identifying-bias-in-ai/tutorial">Identifying Bias in AI</a></b>

# <ol>
#     <li>How many types of biases are described on the page?
#         <br>
#         This page displays 6 types of biases: Historical Bias, Representation Bias, Measurement Bias, Aggregation Bias, Evaluation Bias, and Deployment Bias.
#         <br>
#         <p>Kaggle provides the following description of each of the biases</p>
#     <ul>
#       <li><b>Historical bias</b> occurs when the state of the world in which the data was generated is flawed.</li>
#       <li><b>Representation bias</b> occurs when building datasets for training a model, if those datasets poorly represent the people that the model will serve.</li>
#       <li><b>Measurement bias</b> occurs when the accuracy of the data varies across groups.</li>
#       <li><b>Aggregation bias</b> occurs when groups are inappropriately combined, resulting in a model that does not perform well for any group or only performs well for the majority group.</li>
#       <li><b>Evaluation bias</b> occurs when evaluating a model, if the benchmark data (used to compare the model to other models that perform similar tasks) does not represent the population that the model will serve.</li>
#       <li><b>Deployment bias</b> occurs when the problem the model is intended to solve is different from the way it is actually used. If the end users don’t use the model in the way it is intended, there is no guarantee that the model will perform well.</li>
#     </ul>
#     <br>
#     <li>Which type of bias did you know about already before this course and which type was new to you?
#         <br>
#         The majority of the biases were new to me, apart from <b>Historical Biase</b> and <b>Aggregregation Bias</b>
#     </li>
#     <br>
#     <li>Can you think of any others?
#         <br>
#         Other biases that may be considered must be: Confirmation bias, Outcome bias, Overconfidence bias, Hindsight bias, and Self-serving bias.
#     </li>
# </ol>

# <p><b>Findings about the biases in data - Kaggle</b></p>
# <br>
# <p>In this code we look at how biases may be brought across from the data into the code and how its run. For example, in one part of the tutorial, we were looking at how historical bias has come in to alter the results in an innacurate way.</p>

# In[ ]:


from IPython.display import Image
Image (filename = "Kaggle_code_1.png", width = 600, height = 300)


# In[ ]:


Image (filename = "Kaggle_code_2.png", width = 400, height = 150)


# In[ ]:


Image (filename = "Kaggle_code_3.png", width = 600, height = 300)


# In[ ]:


Image (filename = "Kaggle_code_4.png", width = 400, height = 150)


# <p>The code above shows how, though not toxic, the model associates wrong things as being toxic. This shows a bias in favour of white over black in the code. This may be an example of historical bias on behalf of the creator.</p>

# <hr>

# <h3>Word Embedding Demo</h3>

# <b><a href="http://projector.tensorflow.org/">Embedding Projector on Tensorflow</a></b>

# <p>I first begin by ensuring Word2Vec 10k is selected as my Tensor. Typing words in like Apple, Silver, Sound, I am able to see the cluster data and where these words are located within in the three dimensional data projection. After typing in a word, I am able to isolate points - this removes clutter and presents with the upmost related words. Words with similar relations tend to appear closer together, this can be seen with words like phone and computer, cat and dog.</p>
# <br>
# <p>Gender bias may be an issue that coders must consider - automatically, some words are associated withc certain biases. For example, in relation to professions, a lawyer is typically a male-dominated profession, while teaching is predominantly a female profession.</p>

# In[ ]:


Image (filename = "Embedding_Projector_Man.png", width = 600, height=300)


# In[ ]:


Image (filename = "Embedding_Projector_Lawyer.png", width = 600, height=300)


# <ul>
#     <li>In the two images above are the words "man" and "lawyer" - here you can see they're fairly close to each other. You can also notice the top 999 similar words located around both key words. These images are taken from the Embedding Projector on Tensorflow.</li>
# </ul>

# In[ ]:


Image (filename = "Embedding_Projector_Woman.png", width = 600, height=300)


# In[ ]:


Image (filename = "Embedding_Projector_Teacher.png", width = 600, height=300)


# <ul>
#     <li>In the two images above are the words "woman" and "teacher" - here you can see they're fairly close to each other. You can also notice the top 999 similar words located around both key words. These images are taken from the Embedding Projector on Tensorflow.</li>
# </ul>

# <p>From these select words, using this program, you can notice there is some gender bias that is notable - however, whether this is avoidable is questionable. I believe am effort has been made by the programmers to remove the issue of gender bias. Even though a teacher is a profession that is usually associated as being a female proffession, the word "father" is just below it, and the word "man" is not far away also.</p>

# <hr>

# <h3>AI Fairness</h3>

# <b><a href="https://www.kaggle.com/code/alexisbcook/ai-fairness">AI Fairness on Kaggle</a></b>

# <p>Assessing the fairness of AI before it's deployed to the real world.</p>
# <ol>
#     <li>How many criteria are described on the page?
#         <br>
#         This page displays four fairness criteria: Demographic parity/Statistical parity, Equal opportunity, Equal Accuracy, and Group unaware/"Fairness through unawareness", 
#         <br>
#         <p>Kaggle provides the following description of each of these</p>
#     <ul>
#       <li><b>Demographic parity</b> says the model is fair if the composition of people who are selected by the model matches the group membership percentages of the applicants.</li>
#       <li><b>Equal opportunity</b> fairness ensures that the proportion of people who should be selected by the model ("positives") that are correctly selected by the model is the same for each group. We refer to this proportion as the true positive rate (TPR) or sensitivity of the model.</li>
#       <li><b>Equal accuracy</b>. That is, the percentage of correct classifications (people who should be denied and are denied, and people who should be approved who are approved) should be the same for each group. If the model is 98% accurate for individuals in one group, it should be 98% accurate for other groups.</li>
#       <li><b>Group unaware</b> fairness removes all group membership information from the dataset. For instance, we can remove gender data to try to make the model fair to different gender groups. Similarly, we can remove information about race or age.</li>
#     </ul>
#     <br>
#     <li>Which criteria did you know about already before this course and which, if any, was new to you?
#         <br>
#         I had a broad understanding of fairness in data collection and understanding data - however, these presented to me on the Kaggle worksheet, and seen above, were new to me.
#     </li>
#     <br>
#     <li>Can you think of any other criteria?
#         <br>
#         Another way that could allow fairness to be assessed may be to check whether the models output would be the same if individuals attributes and factors differed - this would allow for a check on whether the machine learning code is considering their attributes as a factor towards descision making. A check can be done on whether the machine is benefiting or negatively targeting a group. Checking this may allow for corrections to be made prior to its implementation. One of the examples given by Kaggle is "Group unaware", whereby information about a group is actively forgotten or removed from the data to actively avoid the machine learning being influenced by this. However, this has to be done carefully in case there are other ways of indirectly calculating attributes. The example given by Kaggle is that eventhough race may be removed, the postcode may still present biases and information about race in a racially segregated city. Therefore, by pushing and recommending the machine learning model to consider these factors and actively avoid influence, this may be better at coming to a conclusion with mitigated risk.
#     </li>
# </ol>

# In[ ]:


Image (filename = "Kaggle_Fairness_1.png", width = 600, height=300)


# <ul>
#     <li>First, Kaggle made us import the neccessary tools and datasets. This created an output.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Fairness_2.png", width = 400, height=150)


# <ul>
#     <li>The output of the first set of code can be seen above, whereby we can see the data and distribution.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Fairness_3.png", width = 600, height=300)


# <ul>
#     <li>The code above was then used to train a model to deny or approve individuals for a credit card - this would generate an output that shows the performance of the model.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Fairness_4.png", width = 500, height=200)


# <ul>
#     <li>Above is the output from the first machine learning model - you can see that the code is 94.56% accurate towards group A, but 95.02% accurate towards group B. This disparity, though small, is the difference between someone being wrongfully accepted and someone wrongfully being denied.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Fairness_5.png", width = 750, height=300)


# <ul>
#     <li>This code above was then ran to create a visualisation of the model.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Fairness_6.png", width = 800, height=300)


# <ul>
#     <li>From this model is becomes transparent that there exist innacuracies within the code. If anyone has an income between 71909.5 and 88440.5 then the decision is transferred to what group the individual is in. Whereby, the model accepts your application if you are in group B, however, if you are in group A then you are always denied.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Fairness_7.png", width = 600, height=300)


# <ul>
#     <li>The code above creates a model that is based on the "group unaware" fairness model.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Fairness_8.png", width = 500, height=200)


# <ul>
#     <li>Using this "group unaware" model, it shows that it is 93.61% accurate towards group A and 91.72% accurate towards group B. Another unfair model that is now bettering group A over group B and once again is unfair.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Fairness_9.png", width = 800, height=400)


# <ul>
#     <li>This code creates a third model, ensuring each group has equal representation in the group of approved applicants - this is an implementation of group thresholds.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Fairness_10.png", width = 500, height=200)


# <ul>
#     <li>This produces a more similar result of accuracy for both groups, with 79.74% accuracy for group A and 79.02% for group B. This shows that nearly equal representation is achieved along with demographic parity. However, the overall accuracy of the model has dropped significantly from 94% and 95% in the first model down to 79%.</li>
# </ul>

# <hr>

# <h3>AI and Explainability</h3>

# <b><a href="https://www.kaggle.com/code/dansbecker/permutation-importance">Permutation Importance on Kaggle</a></b>

# <p>Calculating the permutation importance, using Kaggle.</p>
# <br>
# <p>Permutation importance is a helpful tool in detecting the important features that have the biggest impact on predictions. Kaggle states that Permutation importance is:
#     <ul>
#         <li>fast to calculate,</li>
#         <li>widely used and understood,</li>
#         <li>consistent with properties we would want a feature importance measure to have.</li>
#     </ul>
# <ol>
#     <li>How many features are in this dataset?
#         <br>
#         The first model uses the following features from the dataset - 
#         <ul>
#             <li>pickup_longitude</li>
#             <li>pickup_latitude</li>
#             <li>dropoff_longitude</li>
#             <li>dropoff_latitude</li>
#             <li>passenger_count</li>
#         </ul>
#         <br>
#         Whilst the second model uses these features -
#         <ul>
#             <li>pickup_longitude</li>
#             <li>pickup_latitude</li>
#             <li>dropoff_longitude</li>
#             <li>dropoff_latitude</li>
#             <li>abs_lat_change</li>
#             <li>abs_lon_change</li>
#         </ul>
#         <br>
#     <li>Were the results of doing the exercise contrary to intuition? If yes, why? If no, why not?
#         <br>
#         There was the initial surprise that there was such a difference between longitude and latitude - however, knowing how New York is shaped, this is understandable. New York is a long city that was constructed on islands. These islands are long and therefore it makes sense to cost more to go between the North and South of New York - more than East and West.
#     </li>
#     <br>
#     <li>Do you think the permutation importance is a reasonable measure of feature importance?
#         <br>
#         I think that as a model of measuring which features are important, this model can be successfull - it shows which features of the data set have the most influence. Though in most cases it may not be entirely accurate, after human review, an assumption can be made as to which features are more important.
#     </li>
#     <br>
#     <li>Can you think of any examples where this would have issues?
#         <br>
#         Permutation importance be too basic in complex machine learning models with extensive datasets. It may provide some insight, however, in machine learning model that is detecting, transferring and learning slight differences - permutation importance may not be sufficient. Therefore, it may be best to cross-validate with other methods in order to verify results and avoid outliers or biases.
#     </li>
# </ol>

# In[ ]:


Image (filename = "Kaggle_Permutation_Importance_1.png", width = 500, height=200)


# <ul>
#     <li>This first part of code is necessary to import the models and data needed in the following code - it filters the data and establishes the base_features. </li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Permutation_Importance_2.png", width = 800, height=300)


# <ul>
#     <li>This is data that the code above has retrieved and filtered </li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Permutation_Importance_5.png", width = 500, height=200)


# <ul>
#     <li>This code creates a permutation importance which is imported from eli.5.sklearn. We apply the random_state of 1 and fit it to the validation sets of X and Y. We then get the code to print the results, giving a visualisation of of importance of features along with their scores.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Permutation_Importance_6.png", width = 300, height=200)


# <ul>
#     <li>Above are the scores of the code. Whereby dropodd_latitude is the most important, and passenger_count is the least.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Permutation_Importance_7.png", width = 650, height=200)


# <ul>
#     <li>This code above creates new features - "abs_lon_change" and "abs_lat_change". After defining the features, the data is then split into training and validation sets and then creates a RandomForestRegressor with 30 n_estimators and 1 random state and fits it to the new training data. We then make a second PermutationImportance for the new model and then prints the weights along with the corresponding features in order to determine if the importance has differed with these extra features.</li>
# </ul>

# In[ ]:


Image (filename = "Kaggle_Permutation_Importance_8.png", width = 300, height=200)


# <ul>
#     <li>This is the output of the code above - we can see that the results have changed. Whereby "abs_late_change" is now the most important factor, and "dropoff_longitude" is the lowest.
#     <br>
#         In the previous model, "dropoff_latitude" was above "pickup_latitude" - however, in this second model, this has changed. Therefore we can determine that it is important to cross examine this model of determining importance as different factors may influence results.
#     </li>
# </ul>

# <hr>

# In[ ]:




