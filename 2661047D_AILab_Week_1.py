#!/usr/bin/env python
# coding: utf-8

# ## Week 1:  Getting started with Anaconda, Jupyter Notebook and Python
# 
# #### Why you chose to join this course – for, motivation, vision, aspiration?
# <li>One reason I chose to study Artificial Intelligence was my genuine interest in its influence across the world, and the impact it has in everyones lives on a daily basis. This encouraged me to persue AI so I too could contribute to the world, helping others and getting a career related to AI.</li>
# 
# #### Prior experience, if any, you have with AI and/or Python
# <li>I had once tried to teach myself to use Python to create games - however I didn't persue a great deal and only have a foundational understanding.</li>
# <li>I have no prior experience with creating AI, however I do have an understanding of its uses and use it daily from operating my phone through to travelling in a car.</li>
# 
# #### What you expect to learn from the course?
# <li>An introduction in how to code AI so that it is realistic for me to persue.</li>
# <li>The history of AI and what makes AI, this understanding can then become applicable to modern day code.</li>
# <li>Better my writing abilities and improve how I write about code in essays and referencing Figures.</li>
# <li>To be able to create my own programs that I can apply to daily life or help me work or research in the future.</li>
# <li>A better understanding of Python to apply to a variety of fields to improve my employability.</li>

# In[1]:


print("Hello, World!")


# In[2]:


message="Hello, World!"

print(message)


# In[3]:


message="How are you doing?"
print(message)


# In[4]:


message="Hello, World!"
print(message+message)


# In[5]:


message="Hello, World!"
print(message*3)


# In[6]:


message="Hello, World!"
print(message[0])


# In[10]:


message="Hello, World!"
print(message[12])


# In[11]:


greeting="Hello, World!"
print(greeting)


# In[8]:


from IPython.display import*
YouTubeVideo("9UW6jnkiRQE")
#A video of a tour of the University of Glasgows Campus


# In[14]:


from IPython.display import Image
Image(url="https://thumbs.dreamstime.com/b/university-glasgow-main-building-scotland-56041794.jpg", width=200, height=200)
#A stock photo from the internet of the University of Glasgow.


# In[3]:


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


# In[ ]:





# In[ ]:




