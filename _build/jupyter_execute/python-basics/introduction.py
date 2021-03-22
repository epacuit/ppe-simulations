# Introduction


## Variables and Data Types


Python code involves using and reasoning about different **types** of values, or data.  E.g., `42` is an integer, `3.14` is a real number,  and `"Python"` is a string. 

A **variable** is a name that refers to a value. In Python, we can use any word as a variable name as long as it starts with a letter or an underscore.  However, variable names should not be a [reserved word](https://docs.python.org/3.3/reference/lexical_analysis.html#keywords) in Python such as `for`, `if`, `class`, `type`, `print`, etc. as these words encode special functionality in Python. 

See the [Python 3 documentation](https://docs.python.org/3/library/stdtypes.html) for a summary of the standard built-in Python datatypes.



| English name          | Type name  |   Description    | Example                                    |
| :-------------------- | :--------- | :------------- | :-------------------------------------------- |
| integer               | `int`      |  positive/negative whole numbers               | `42`                                       |
| floating point number | `float`    | real number in decimal form                   | `3.14159`                                  |
| boolean               | `bool`     |  true or false                                 | `True`                                     |
| string                | `str`      | text                                          | `"Logic is fun!"`                 |
| list                  | `list`     |  an ordered collection of objects  | `['Thursday', 3, 18, 2021]`               |
| tuple                 | `tuple`    |  an ordered collection of objects  | `('Thursday', 3, 18, 2021)`                 |
| dictionary            | `dict`     |  mapping of key-value pairs        | `{'name':'PHPE', 'course_number':409}` |
| none                  | `NoneType` |  represents no value                           | `None`         |

 We can determine the type of an object in Python using `type()`. We can print the value of the object using `print()`.

x = 42
type(x)

print(x)

```{note}
In Jupyter, the last line of a cell will automatically be printed to screen so we don't actually need to explicitly call `print()`.
```

### Arithmetic Operators

Below is a table of the syntax for common arithmetic operations in Python:

| Operator |   Description    |
| :------: | :--------------: |
|   `+`    |     addition     |
|   `-`    |   subtraction    |
|   `*`    |  multiplication  |
|   `/`    |     division     |
|   `**`   |  exponentiation  |
|   `//`   |   floor division |
|   `%`    |      modulo      |


1 + 2 + 3   # add

2 * 12.5   # multiply

2 ** 5  # exponent

10 / 5 # division

```{warning}
Division may  change `int` to `float`.
```

x = 2
type(x)

y = x / x

type(y)

25 // 2  # "floor division" - always rounds down

Note that the `//` always results in an integer

type(25 // 2)

25 % 2  # the remainder when 25 is divided by 2

### None

The `NoneType` data type represents a null value. 

x = None

print(x)

type(x)

### Strings

Strings are either enclosed in single quotes or double quotes:

  - single quotes, e.g., `'Philosophy'` 
  - double quotes, e.g., `"Economics"`

There is no difference between the two ways of writing a string.  In fact, there one can also use a three quotes to express a string (this is typically used for writing long strings): 

  - `"""This is a long strong"""`.
  - `'''This is another long strong'''`.

course = "PHPE 409"
type(course)

Python 3 has a very convenient way to build strings from variables using so-called **f-strings** (formatted strings, see [https://www.digitalocean.com/community/tutorials/how-to-use-f-strings-to-create-strings-in-python-3](https://www.digitalocean.com/community/tutorials/how-to-use-f-strings-to-create-strings-in-python-3) for an overview).   

course_name = 'PHPE'
course_num = 409

print(f"{course_name} {course_num}")

A second way to build strings is to use `+'. 

'Hello' + ' ' + 'World'

Note that it is an error to try to add a string and a numeric data types

course_name + ' ' + course_num # produces an error

course_name + ' ' + str(course_num) # convert course_num to a string

### Boolean

The `bool` type has two values: `True` and `False`.

t = True

t

type(t)

f = False

f

type(f)

### Comparison Operators

We can compare objects using comparison operators, and we'll get back a Boolean result:

| Operator  | Description                          |
| :-------- | :----------------------------------- |
| `x == y ` | is `x` equal to `y`?                 |
| `x != y`  | is `x` not equal to `y`?             |
| `x > y`   | is `x` greater than `y`?             |
| `x >= y`  | is `x` greater than or equal to `y`? |
| `x < y`   | is `x` less than `y`?                |
| `x <= y`  | is `x` less than or equal to `y`?    |
| `x is y`  | is `x` the same object as `y`?       |

1 < 2

2 != "2"

x= 2
y = 4 / 2
x == y # the values of x and y are the same

x is y # but x and y are not the same object (x is an int and y is a float)

### Boolean Operators

Python includes the standard Boolean operators: 

| Operator | Description |
| :---: | :--- |
|`x and y`| True when `x` and `y` both True |
|`x or y` | True when  at least one of `x` and `y` True.|
| `not x` | True when `x` is False. | 

True and True

True and False

True or False

False or False

not True

not False

## Lists,  Tuples and Sets


Lists,  tuples and sets can be sued to store multiple things ("elements") in a single object.


list_example = [1, 2.0, 'three']

list_example

type(list_example)

A list can contain any data type, even other lists. 

list_example2 = [1, 2.0,  [3, 4], None, True]
list_example2

The values inside a list can be accessed using square bracket syntax. Note that the first element of the list is in position 0.

list_example2[0] # first element 

list_example2[2] # 3rd element 

list_example2[-1] # the last element (same as list_example[4])

Use the colon `:` to access a sub-sequence (this is called "slicing").

list_example2[1:3] # return item 1 and 2

A `tuple` is the same as a list except that it is **immutable**.   This means that tuples cannot be changed.


a_list = [1,2,3] # a list
a_tuple = (1,2,3) # a list

print(f"The 2nd element of a_list is {a_list[1]}")

a_list[1] = "new value"

print(f"The 2nd element of a_list is now {a_tuple[1]}")

a_tuple[1] = "new value" # produces an error

Three useful builtin methods that apply to lists are: 

* `len`: return the lenght of a list, 
* `append`: append elements to the end of a list
* `join`: join the elements of a list to form a string

a_list = ['this', 'is', 'a', 'list']

len(a_list)


a_list.append("of strings")
a_list

"_".join(a_list) # join the elements of a list

More information about the available list methods is found here:  [https://docs.python.org/3/tutorial/datastructures.html#more-on-lists](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists).

### Sets

Another built-in Python data type is the `set`.   This is an  _un-ordered_ list of _unique_ items.

s = {2, 3, 5, 11, 11} # sets are defined using curly braces
s

{1, 2, 3} == {3, 2, 1, 1}

[1, 2, 3] == [3, 2, 1]

Sets are a convenient way to remove multiple copies of an element from a list

a_list = [1, 1, 2, 3, 4, 5, 5, 6, 7, 7, 7]

list(set(a_list)) # remove multiple copies of elements

## Dictionaries


A dictionary is a mapping between key-values pairs and is defined with curly-brackets:

agent = {
    "name": "Eric",
    "id": 1234,
    "some_attribute": (1, 2, 3),
}


We can access a specific field of a dictionary with square brackets:

agent["name"]

agent["some_attribute"]

Dictionaries are mutable, so they can be editted: 

agent["some_attribute"] = 42
agent

New key-value pairs can be added to a dictionary: 

agent['another_attribute'] = 1.0

agent

A dictionary key can be any immutable data type, even a `tuple`.

agent[(1, 2, 3)] = True
agent

You get an error if you try to access a key that doesn't exist: 

agent["not-here"] # produces error

The following methods are helpful when reasoning about dictionaries: 

agent.keys() # the keys of the dictionary

agent.values() # the values of the dictionary

agent.items() # the key,value pairs of the dictionary

```{note} 

For a nice overview of dictionaries in Python, see this [article](https://medium.com/analytics-vidhya/15-things-to-know-to-master-python-dictionaries-56ab7edc3482).
```

##  Conditionals


[Conditional statements](https://docs.python.org/3/tutorial/controlflow.html) allow blocks of code to be exected depending on the value of one or more variables.  The main conditional statement is an if/then statement: 

- The keywords are `if`, `elif` and `else`
- The colon `:` ends each conditional expression
- Indentation (by 4 empty space) defines code blocks
- In an `if` statement, the first block whose conditional statement returns `True` is executed and the program exits the `if` block
- `if` statements don't necessarily need `elif` or `else`
- `elif` is used to check several conditions
- `else` evalues a default block of code if all other conditions are `False`
- the end of the entire `if` statement is where the indentation returns to the same level as the first `if` keyword


x = 5

if x > 10: 
    print("greater than 10")
elif x <= 10 and x > 5: 
    print("between 5 and 10")
elif x > 0 and x <= 5:
    print("between 0 and 5")
else:
    print("smaller than or equal to 0")

x = 100

if x > 10: 
    print("greater than 10")
elif x <= 10 and x > 5: 
    print("between 5 and 10")
elif x > 0 and x <= 5:
    print("between 0 and 5")
else:
    print("smaller than or equal to 0")

x = -1

if x > 10: 
    print("greater than 10")
elif x <= 10 and x > 5: 
    print("between 5 and 10")
elif x > 0 and x <= 5:
    print("between 0 and 5")
else:
    print("smaller than or equal to 0")

Python allows "inline" `if` statements, which can help make your code easier to read. 

my_list  = [0, 1, 2, 3, 4]

x = "more than 4 elements" if len(my_list) > 4 else "less than or equal to 4 elements"
x

Line 3 is shorthand for the following: 

if len(my_list) > 4:
    x = "more than 4 elements"
else:
    x = "less than or equal to 4 elements"

x

A common pattern is to use the `in` keyword to test if an element is in a list. 

x = 4 
if x in my_list: 
    print(f"{x} is in my_list")
else: 
    print(f"{x} is not in my_list")
     
x = 6
if x in my_list: 
    print(f"{x} is in my_list")
else: 
    print(f"{x} is not in my_list")


Another useful pattern is to use `in` to test if a key is in a dictionary.   In addition, we can test if a variable is not None by using the keywords `is not`. 

my_dict = {"key1": "val1", "key2": False, "key3": 3}

k = "key3"
val = my_dict[k] if  k in my_dict.keys() else None
if val is not None: 
    print(f"val is {val}")
else: 
    print(f"{k} is not a key in my_dict")



k = "key4"
val = my_dict[k] if  k in my_dict.keys() else None

if val is not None: 
    print(f"val is {val}")
else: 
    print(f"{k} is not a key in my_dict")


## Loops

A for loop allows some code to be executed  a specific number of times.

* The keyword `for` begins the loop, and the colon `:` ends the first line of the loop.
* The block of code indented is executed for each value in the list (hence the name "for" loops)
* We can iterate over any kind of "iterable": `list`, `tuple`, `range`, `set`, `string`.  An iterable is really just a sequence of values that can be looped over. 

my_list = [0, 1, 2, 3]
for i in my_list:
    print(i)

The builtin function `range` can be used to iterate over a list of integers:

* `range(start, end, step)` is a list of the numbers between start and end with  each subsequent number separted by step. 
* `range(n)` is short for `range(0,n,1)`

for i in range(2, 20, 2): 
    print(i)

for i in range(5): 
    print(i)

Loops can be nested: 

my_list1 = [0, 1, 2, 3]
my_list2 = ["A", "B", "C", "D"]
for i in my_list1:
    for j in my_list2:
        print(i, j)

The above nested loop outputs the product of the two lists, but often it will be important to match the elements of the two lists: 

my_list1 = [0, 1, 2, 3]
my_list2 = ["A", "B", "C", "D"]
for idx in range(len(my_list1)):
    print(my_list1[idx], my_list2[idx])

There is a much easier way to do this using the `zip` builtin function: 

for i in zip(my_list1, my_list2):
    print(i[0], i[1])

The builtin function `enumerate` adds a counter to the loop: 

my_list2 = ["A", "B", "C", "D"]
for idx,item in enumerate(my_list2):
    print(f"index {idx}", f"item {item}")

There is also a  [`while` loop](https://docs.python.org/3/reference/compound_stmts.html#while) to excute a block of code several times. 

i = 0
while i < 5: 
    print(i)
    i += 1 # shorthand for i = i + 1

## Comprehensions

Comprehensions are used to build lists/tuples/sets/dictionaries in one convenient, compact line of code. Below is a standard `for` loop you might use to iterate over an iterable and create a list:

even_integers = list()
for i in range(10): 
    if i % 2 == 0: 
        even_integers.append(i)
print(even_integers)

Lines 1 - 4 can the nicely reduced to a single line of code: 

event_integers = [i for i in range(10) if i % 2 == 0]
print(even_integers)

Comprehensions can also be used to create a dictionary: 

ws = ["Philosophy", "Politics", "Economics"]
d = {w: len(w) for w in ws}
print(d)

## Functions

A [function](https://docs.python.org/3/tutorial/controlflow.html#defining-functions) is a reusable piece of code that can accept input parameters, also known as "arguments". For example, the following function called `square`  takes one argument `n` and returns the square `n**2`:

def square(n): 
    return n ** 2

square(5)

A function can have *local* variables which are not accessible outside of the function.

def f_with_local_variable():
    local_var = "defined inside the function"
    print(local_var)
    
f_with_local_variable() # executes the function
local_var # produces and error since local_var is only accessible inside the function

Functions can return more than one value:

def f_return_2_values(i): 
    return i + i, i * i

f_return_2_values(5)

You can unpack the returned values as follows: 

val1, val2 = f_return_2_values(7)
print("val1 is ", val1)
print("val2 is ", val2)

It is often convenient to set a default value of a parameter (such parameters are called keyword parameters): 

def add_n(i, n = 10): 
    return i + n

print(add_n(5))
print(add_n(5, n=100))


When designing a function, you should be careful not to modify any of the parameters: 

def sum_arr(arr): 
    arr.append(0)
    return sum(arr)

arr = [1, 2, 3, 4, 5]
print(sum_arr(arr))
print(arr) # note that 0 was added to arr!

## Imports

We will often need to use Python code created by others (often called a Python package).   To do this we use the `import` keyword.   For example, Python does not have an implementation of a factorial as a builtin function.  Rather than implementing this function ourselves, we can import the [Python math package](https://docs.python.org/3/library/math.html) to use its factorial function. 

import math # import the math package
math.factorial(5)

Sometimes we want to use our own naming convention for the imported Python package: 

import math as m # rename the math package as m
m.factorial(5)

Rather than importing an entire package, it is more efficent to import the specific function that is needed: 

from math import factorial

factorial(5)

You can also rename the imported function: 

from math import factorial as fact

fact(5)

There is a lot more to say about Python packages and imports.  See [https://realpython.com/python-import/](https://realpython.com/python-import/) for an overview.  

```{note} 
This only scratches the surface of Python.   The best way to learn Python is to get your hands dirty and start playing around with code.  Python is a [very popular programming language](https://pypl.github.io/PYPL.html), so you can find a lot of tutorials and courses online.  When you need help search Google (you will probably find a lot of answers at [https://stackoverflow.com/](https://stackoverflow.com/).   Have [fun](https://xkcd.com/353/)! 
```