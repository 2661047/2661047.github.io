#!/usr/bin/env python
# coding: utf-8

#  <h1><u>Week 1:  Getting started with Anaconda, Jupyter Notebook and Python</u></h1>
# 
# #### Why you chose to join this course – for, motivation, vision, aspiration?
# <li><p>One reason I chose to study Artificial Intelligence was my genuine interest in its influence across the world, and the impact it has in everyones lives on a daily basis. This encouraged me to persue AI so I too could contribute to the world, helping others and getting a career related to AI.</p></li>
# 
# #### Prior experience, if any, you have with AI and/or Python
# <li><p>I had once tried to teach myself to use Python to create games - however I didn't persue a great deal and only have a foundational understanding.</p></li>
# <li><p>I have no prior experience with creating AI, however I do have an understanding of its uses and use it daily from operating my phone through to travelling in a car.</p></li>
# 
# #### What you expect to learn from the course?
# <li><p>An introduction in how to code AI so that it is realistic for me to persue.</p></li>
# <li><p>The history of AI and what makes AI, this understanding can then become applicable to modern day code.</p></li>
# <li><p>Better my writing abilities and improve how I write about code in essays and referencing Figures.</p></li>
# <li><p>To be able to create my own programs that I can apply to daily life or help me work or research in the future.</p></li>
# <li><p>A better understanding of Python to apply to a variety of fields to improve my employability.</p></li>

# In[5]:


print("Hello, World!")


# In[6]:


message="Hello, World!"

print(message)


# In[7]:


message="How are you doing?"
print(message)


# In[8]:


message="Hello, World!"
print(message+message)


# In[9]:


message="Hello, World!"
print(message*3)


# In[10]:


message="Hello, World!"
print(message[0])


# In[11]:


message="Hello, World!"
print(message[12])


# In[12]:


greeting="Hello, World!"
print(greeting)


# In[13]:


from IPython.display import*
YouTubeVideo("9UW6jnkiRQE")
#A video of a tour of the University of Glasgows Campus


# In[14]:


from IPython.display import Image
Image(url="https://thumbs.dreamstime.com/b/university-glasgow-main-building-scotland-56041794.jpg", width=200, height=200)
#A stock photo from the internet of the University of Glasgow.


# In[15]:


#Code to display a website in your browser from a period requested by the user.
#A variable can be shown by using a =
#A string can be shown with "" or ''
#A string can be assigned to a variable - This is seen in the second line of code.

import webbrowser #This is importing a library.
import requests #This is importing a library.

print("Shall we hunt down an old website?") #This prints an initial comment of direction to the user.
site = input("Type a website URL: ") #This prints a comment, informing the user to insert a URL. This is also a variable.
era = input("Type year, month, and date, e.g., 20150613: ") #This prints a comment, informing the user to insert a date. This is also a variable.
url = "http://archive.org/wayback/available?url=%s&timestamp=%s" % (site, era) #This is a variable.
response = requests.get(url) #This is a variable.
data = response.json() #This is a variable.
try:
    old_site = data["archived_snapshots"]["closest"]["url"] #This is a variable.
    print("Found this copy: ", old_site) #This prints a comment, informing the user of the URL found to display the webpage.
    print("It should appear in your browser.") #This prints a comment, informing the user of where to find the image.
    webbrowser.open(old_site)
except:
    print("Sorry, could not find the site.") #This prints a comment if a site is unable to be found.


# <hr>

# <h2><u>Week 2. Exploring Data in Multiple Ways</u></h2>

# In[21]:


from IPython.display import Image


# In[22]:


#Importing an image using IPython.display.
Image ("picture1.jpg")


# In[18]:


#Importing audio using IPython.display.
from IPython.display import Audio


# In[19]:


Audio("audio1.mid")


# In[23]:


from IPython.display import Audio


# In[24]:


#This audio belongs to and is licensed by Artoffuge Mehmet Okonsar. This can be found at "https://en.wikipedia.org/wiki/File:GoldbergVariations_MehmetOkonsar-1of3_Var1to10.ogg"
Audio("audio2.ogg")


# <p>Above, I have imported Images and Audio using IPython.display. I managed to successfuly insert the image of a hedghog. However, in my attempt to insert audio, I have successfully got the software to display audio outputs - though it fails to play any audible audio. In my attempt to resolve this I will use "https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html" and further consult with my peers and teacher.</p>
# <p>After review, the audio1.mid is unable to play audio - this is because the file is stored as a midi file, which means that it is unable to store and play audio. The second, audio2.ogg is able to play sound imported from IPython.display - however, as I am using Safari on a mac this software is unable to play audio as a .ogg file.</p>

# <hr>

# <h3><u>Using matplotlib</u></h3>

# In[25]:


#To present the numerical data as a numpy array.
from matplotlib import pyplot
test_picture = pyplot.imread("picture1.jpg")
print("Numpy array of the image is: ", test_picture)
pyplot.imshow(test_picture)


# In[26]:


#To change the values in the previous numpy array.
test_picture_filtered = 2*test_picture/3
pyplot.imshow(test_picture_filtered)


# <p>Above is numerical data displayed as a numpy array, this numerical data represents the image that also can be seen. The image directly above, that appears to be colourful dots, is the same image of the hedgehog as seen in the previous part - however, the values in the array have been doubled and then divided by three. In return, the image displayed has been altered to show only certain aspects of the hedgehog with different colours.</p>

# <hr>

# <h3><u>Scikt-learn (sklearn)</u></h3>

# In[27]:


from sklearn import datasets


# In[28]:


#To show the datasets imported from sklearn.
dir(datasets)


# <p>'load_sample_images', 'load_iris', 'load_wine' are some datasets I have chosen from the dataset above to be investigated later. I have chosen these as I am interested in seeing what these datasets contain.</p>

# In[29]:


#To present different datasets from sklearn.
sample_images_data = datasets.load_sample_images()
iris_data = datasets.load_iris()
wine_data = datasets.load_wine()
print(sample_images_data.DESCR)
print(iris_data.DESCR)
print(wine_data.DESCR)


# In[30]:


#To present the attributes of the Iris dataset and show the features.
iris_data.feature_names


# In[31]:


#To present the attributes of the Wine dataset and show the features.
wine_data.feature_names


# In[32]:


#This command tells the program to bring the "target names" from the data set using the variable created and print them for the user.
wine_data.target_names


# In[33]:


#This is the same command as the previous, however it is for a different variable.
iris_data.target_names


# In[35]:


#In this code 'wine_data' and 'iris_data' are being imported from sklearn and converted into pandas dataframe which has been called 'wine_dataframe' and 'iris_dataframe'.
from sklearn import datasets
import pandas

wine_data = datasets.load_wine()

wine_dataframe = pandas.DataFrame(data=wine_data['data'], columns = wine_data['feature_names'])

iris_data = datasets.load_iris()

iris_dataframe = pandas.DataFrame(data=iris_data['data'], columns = iris_data['feature_names'])


# In[36]:


wine_dataframe.head()
wine_dataframe.describe()


# In[37]:


iris_dataframe.head()
iris_dataframe.describe()


# <p>The code above uses panda to retrieve the imported data from sklearn and transfers it into more detailed information that is easier to read. The data above is seen in the datasets that were printed earlier - however, it is put into a chart and is displayed beyond two decimal places. </p>

# In[ ]:




