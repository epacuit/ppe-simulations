# Classes

When implementing an agent-based model, we will often use **classes**.   A **class** is a way to create your own data taype.   An instance of a class is called an **object** (see the [Python class documentation]((https://docs.python.org/3/tutorial/classes.html) for an overview.). The general approach to programming using classes and objects is called [object-oriented programming](https://realpython.com/lessons/what-object-oriented-programming-oop/).


To define a class use the `class` keyword, followed by a name and a colon`:`.   A class can be thought of as a **blueprint** for creating objects.  Classes contain data called **attributes** and functions called **methods**.  Some initial observations about creating a class: 

1. There are two types of attributes, a class attribute and an instance attribute.   Class attirbutes are the same among all instances of the class while the instance attributes may vary for different instances of a class.
2.  Typically, a class contains a `__init__` method which will be run every time a new instance is created. 
3.  The first parameter of any class method is `self` which refers to an instance of the class. 
4.  Class attributes should be defined outside of the `__init__` method, while instance attributes are defined inside the body of the `__init__` method.

class Agent():
    
    class_name = 'Agent Class'
    def __init__(self, unique_id):
        self.unique_id = unique_id
        
a = Agent(1)
b = Agent(2)

print(a.class_name)
print("Agent a's unique id is ", a.unique_id)

print(b.class_name)
print("Agent b's unique id is ", b.unique_id)


In a class `self` refers to an instance of that class.  In the following code, printing self and printing the object instance generates the same ouput (the memory location of the instance of the Agent class). 

class Agent():
    
    class_name = 'Agent Class'
    def __init__(self, unique_id):
        self.unique_id = unique_id
        print(self)

a = Agent(1)
print(a)

To illustrate class methods, let's give our Agent the ability to flip a fair coin.   To do this, we use the `random` package to generate a random number between 0 and 1.   If the randomly generated number is less than 0.5, we output 'H' and if it is greater than or equal to 0.5 we output 'T'.    The first thing we have to do is import the random package. 

import random

Next, we add class method called `flip` that accepts `self` as its only parameter.  

class Agent():
    
    class_name = 'Agent Class'
    def __init__(self, unique_id):
        self.unique_id = unique_id

    def flip(self): 
        return 'H' if random.random() < 0.5 else 'T'

a = Agent(1) # create an agent

# let agent a flip a fair coin 5 times
# there will be different outputs everytime this code is executed
for i in range(5):
    print(a.flip())

Let's add an another method to the Agent class that returns multiple flips of a coin. The method will be called `multiple_flips` and have a keyword parameter `num` with a default value of 5.  Note that we call the class method `flip` from the `multiple_flip` function as follows: `self.flip()`. 

class Agent():
    
    class_name = 'Agent Class'
    def __init__(self, unique_id):
        self.unique_id = unique_id

    def flip(self): 
        return 'H' if random.random() < 0.5 else 'T'
    
    def multiple_flips(self, num = 5): 
        return [self.flip() for n in range(num)]


a = Agent(1) # create an agent

print(a.multiple_flips(10))

## Inheritence and Subclasses

Inheritance allows one class to "inherit" methods and attributes from another class.   For example, suppose that we have a general Agent class and CoinFlipper class that implements an agent that flips a coin of a fixed bias which  is a subclass of the Agent class.   

class Agent():
    
    class_name = 'Agent Class'
    def __init__(self, unique_id):
        self.unique_id = unique_id

class CoinFlipper(Agent):
    
    def __init__(self, bias):
        self.bias = bias
        
    def flip(self): 
        return 'H' if random.random() < self.bias else 'T'
    
    def multiple_flips(self, num = 5): 
        return [self.flip() for n in range(num)]


An instance of the CoinFlipper class, has access to the attributes and methods of the parent class. 

a = CoinFlipper(0.5)
print(a.multiple_flips())
print(a.class_name)

However, there is a problem with the above implementation.   An instance of the CoinFlipper hasn't set the unique_id of the partent class. 

a = CoinFlipper(0.5)
a.unique_id # produces an error since the CoinFlipper class doesn't have a unique_id attribute

The problem with the above code is that when initializing the CoinFlipper instance we didn't call the `__init__` method of the Agent class.   There are two ways to do this: 

1. Explicitly call the base class `__init__` method
2. Use the `super()` builtin function to instantiate the base class


class Agent():
    
    class_name = 'Agent Class'
    def __init__(self, unique_id):
        self.unique_id = unique_id

class CoinFlipper(Agent):
    
    def __init__(self, unique_id, bias):
        Agent.__init__(self, unique_id) # explicitly call the base class __init__ function
        self.bias = bias
        
    def flip(self): 
        return 'H' if random.random() < self.bias else 'T'
    
    def multiple_flips(self, num = 5): 
        return [self.flip() for n in range(num)]

a = CoinFlipper(1, 0.5)
a.unique_id 

The second approach using the `super()` key word is often a better approach to do the same thing.  See [https://realpython.com/python-super/](https://realpython.com/python-super/) for a discussion.  

class Agent():
    
    class_name = 'Agent Class'
    def __init__(self, unique_id):
        self.unique_id = unique_id

class CoinFlipper(Agent):
    
    def __init__(self, unique_id, bias):
        super().__init__(unique_id) # super() refers to the base class
        self.bias = bias
        
    def flip(self): 
        return 'H' if random.random() < self.bias else 'T'
    
    def multiple_flips(self, num = 5): 
        return [self.flip() for n in range(num)]

a = CoinFlipper(1, 0.5)
a.unique_id 

## Decorators

One programming construct that is not specific to classes, but is often used when creating a class is a **decorator**.  A decorator "decorates" a function/method with additional functionality.  That is, it is a function that accepts another function as a paramter and adds functionality to that function.  

See [https://realpython.com/primer-on-python-decorators/](https://realpython.com/primer-on-python-decorators/) for an overview of decorators. 

```{warning} 
In the following code, we use the the parameter to the inner function wrapper is `*args`.   This is a way of passing an arbitrary number of arguments to wrapper.   The problem is that `original_func` accepts a single argument which needs to be passed to the decorator.    See [https://realpython.com/python-kwargs-and-args/](https://realpython.com/python-kwargs-and-args/) for an overview. 
```

def original_func(n):
    print("Original function")
    return n*2

# a decorator
def my_decorator(func):  # takes our original function as input
    
    def wrapper(*args):  # wraps our original function with some extra functionality
        print(f"A decoration before {func.__name__}.")
        result = func(*args)
        print(f"A decoration after {func.__name__} with result {result}")
        return result + 10 # add 10 the result of func
    
    return wrapper  # returns the unexecuted wrapper function which we can can excute later

original_func(10)

my_decorator(original_func)(10)

We can now use this decorator  for other functions using by adding `@my_decorator` before the definition of the function. 

@my_decorator
def another_func(n): 
    print("Another func")
    return n + 2

another_func(5)

Some commonly used decorators are Python builtins `@classmethod`, `@staticmethod`, and `@property`. We focus here on the `@property` decorator.   This can be used to customize *getters* and *setters* for class attributes. Suppose that we want to create a Coin class that has a fixed bias (which may be changed).  

```{note} 
A common approach in object-oriented programming is to make attributes of a class *private* so that users of the class can only get and set these attributes through so-called "getter" and "setter" functions.   In Python there is no way to force a variable to be "[private](https://softwareengineering.stackexchange.com/questions/143736/why-do-we-need-private-variables)" (this is different than languages such as C++ or Java).  A common approach is to add an underscore "_" to the begining of a variable name that should be private (see [https://www.geeksforgeeks.org/private-variables-python/](https://www.geeksforgeeks.org/private-variables-python/). 

```

class Coin():
    
    def __init__(self, bias = 0.5): 
        
        self._bias = 0.5 
        
    @property
    def bias(self): 
        """Get the bias of the coin"""
        return self._bias
    
    @bias.setter
    def bias(self, b):
        """Set the bias and raise and error if bias is not between 0 and 1"""
        if b >= 0 and b <=1:
            self._bias = b
        else:
            raise ValueError("Bias must be between 0 and 1")
    
    @property
    def pr_heads(self):
        """Get the probability of heads"""
        return self._bias
    
    @property
    def pr_tails(self):
        """Get the probability of heads"""
        return 1 - self._bias

    def flip(self):
        """flip the coin"""
        return 'H' if random.random() < self._bias else 'T'

    def flips(self, num=10):
        """flip the coin"""
        return [self.flip() for n in range(num)]
    
c = Coin()
print("the bias of c is ", c.bias)
print("the probability of heads is ", c.pr_heads)
print("the probability of tails is ", c.pr_tails)
print(c.flips(), "\n")

# now change the bias
c.bias = 0.75
print("the bias of c is ", c.bias)
print("the probability of heads is ", c.pr_heads)
print("the probability of tails is ", c.pr_tails)
print(c.flips())


Trying to assign a bias greater than 1 generates an error: 

c.bias = 1.5

