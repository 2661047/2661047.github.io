#!/usr/bin/env python
# coding: utf-8

# # Data Types Homework
# <h2>Introduction</h2>
# Data types are important, because they determine what kinds of actions you can do with them. For instance, you can divide two floats, but you cannot divide two strings. For instance, 12.0/2.0 makes sense, but "cat"/"dog" does not.

# <h3>Integers</h3>
# Integers are numbers without any fractional part and can be positive (1, 2, 3, ...), negative (-1, -2, -3, ...), or zero (0).
# 
# In the code cell below, we set a variable x to an integer. We then verify the data type with type(), and need only pass the variable name into the parentheses.

# In[5]:


x=23
print(x)
print(type(x))


# <h3>Floats</h3>
# Floats are numbers with fractional parts. They can have many numbers after decimal.

# In[2]:


almost_pi = 22/7
print(almost_pi)
print(type(almost_pi))


# One function that is particularly useful for fractions is the round() function. It lets you round a number to a specified number of decimal places.

# In[3]:


rounded_pi = round(almost_pi, 5)
print(rounded_pi)
print(type(rounded_pi))


# Whenever you write an number with a decimal point, Python recognizes it as a float data type.
# 
# For instance, 1. (or 1.0, 1.00, etc) will be recognized as a float. This is the case, even though these numbers technically have no fractional part!

# In[4]:


y_float = 1.
print(y_float)
print(type(y_float))


# <h3>Booleans</h3>
# Booleans represent one of two values: True or False. In the code cell below, z_one is set to a boolean with value True

# In[6]:


z_one = True
print(z_one)
print(type(z_one))


# Next, z_two is set to a boolean with value False.

# In[7]:


z_two = False
print(z_two)
print(type(z_two))


# Booleans are used to represent the truth value of an expression. Since 1 < 2 is a true statement, z_three takes on a value of True.

# In[8]:


z_three = (1 < 2)
print(z_three)
print(type(z_three))


# Similarly, since 5 < 3 is a false statement, z_four takes on a value of False.

# In[9]:


z_four = (5 < 3)
print(z_four)
print(type(z_four))


# We can switch the value of a boolean by using not. So, not True is equivalent to False, and not False becomes True.

# In[10]:


z_five = not z_four
print(z_five)
print(type(z_five))


# <h3>Strings</h3>
# The string data type is a collection of characters (like alphabet letters, punctuation, numerical digits, or symbols) contained in quotation marks. Strings are commonly used to represent text.

# In[11]:


w = "Hello, Python!"
print(w)
print(type(w))


# You can get the length of a string with len().  "Hello, Python!" has length 14, because it has 14 characters, including the space, comma, and exclamation mark. Note that the quotation marks are not included when calculating the length.

# In[12]:


w = "Hello, Python!"
print(w)
print(len(w))


# One special type of string is the empty string, which has length zero.

# In[13]:


shortest_string = ""
print(type(shortest_string))
print(len(shortest_string))


# If you put a number in quotation marks, it has a string data type.

# In[14]:


my_number = "1.12321"
print(my_number)
print(type(my_number))


# If we have a string that is convertible to a float, we can use float().
# 
# This won't always work! For instance, we can convert "10.43430" and "3" to floats, but we cannot convert "Hello, Python!" to a float.

# In[15]:


also_my_number = float(my_number)
print(also_my_number)
print(type(also_my_number))


# Just like you can add two numbers (floats or integers), you can also add two strings. It results in a longer string that combines the two original strings by concatenating them.

# In[16]:


new_string = "abc" + "def"
print(new_string)
print(type(new_string))


# Note that it's not possible to do subtraction or division with two strings. You also can't multiply two strings, but you can multiply a string by an integer. This again results in a string that's just the original string concatenated with itself a specified number of times.

# In[17]:


newest_string = "abc" * 3
print(newest_string)
print(type(newest_string))


# Note that you cannot multiply a string by a float! Trying to do so will return an error.
# <br>will_not_work = "abc" * 3.

# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# In[18]:


a = 1
print(a)
a = a + 1
print(a)
a = a * 2
print(a)


# In[19]:


number = float(input("Type in a number: "))
integer = int(input("Type in an integer: "))
text = input("Type in a string: ")
print("number =", number)
print("number is a", type(number))
print("number * 2 =", number * 2)
print("integer =", integer)
print("integer is a", type(integer))
print("integer * 2 =", integer * 2)
print("text =", text)
print("text is a", type(text))
print("text * 2 =", text * 2)


# Notice that number was created with float(input()) ,int(input()) returns an integer, a number with no decimal point, while text created with input() returns a string(can be writen as str(input()), too). When you want the user to type in a decimal use float(input()), if you want the user to type in an integer use int(input()), but if you want the user to type in a string use input().
# 
# The second half of the program uses the type() function which tells what kind a variable is. Numbers are of type int or float, which are short for integer and floating point (mostly used for decimal numbers), respectively. Text strings are of type str, short for string. Integers and floats can be worked on by mathematical functions, strings cannot. Notice how when python multiplies a number by an integer the expected thing happens. However when a string is multiplied by an integer the result is that multiple copies of the string are produced (i.e., text * 2 = HelloHello).
# 
# Operations with strings do different things than operations with numbers. As well, some operations only work with numbers (both integers and floating point numbers) and will give an error if a string is used. Here are some interactive mode examples to show that some more.

# In[20]:


# This program calculates rate and distance problems
print("Input a rate and a distance")
rate = float(input("Rate: "))
distance = float(input("Distance: "))
time=(distance/ rate)
print("Time:", time)


# In[21]:


# This program calculates the perimeter and area of a rectangle
print("Calculate information about a rectangle")
length = float(input("Length: "))
width = float(input("Width: "))
Perimeter=(2 * length + 2 * width) 
print("Area:", length * width)
print("Perimeter:",Perimeter)


# In[22]:


# This program converts Fahrenheit to Celsius
fahr_temp = float(input("Fahrenheit temperature: "))
celc_temp = (fahr_temp - 32.0) *( 5.0 / 9.0)
print("Celsius temperature:", celc_temp)


# In[ ]:




