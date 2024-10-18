bagofwords = [
    # Basic operations with variations
    "def add(a, b): return a + b  # Simple addition",
    "def subtract(a, b): return a - b  # Subtraction operation",
    "def multiply(a, b): return a * b  # Multiplication",
    "def divide(a, b): return a / b if b != 0 else 'Error: Division by zero'",

    # Loops with comments and variations
    "for i in range(10): print(i)  # Loop through numbers 0-9",
    "while x < 10: x += 1  # Increment x until it reaches 10",
    "for item in items: process(item)  # Process each item in list",
    "while not done: continue_work()  # Continue work until done",

    # Conditionals with comments
    "if x > 0: print('Positive')  # Check if x is positive",
    "if x < 0: print('Negative') else: print('Zero')  # Check if x is negative or zero",
    "if user.is_authenticated: access_granted()  # Grant access if user is authenticated",
    "if error: log_error() else: proceed()  # Log error or proceed",

    # Functions with comments and variations
    "def factorial(n): return 1 if n == 0 else n * factorial(n-1)  # Recursive factorial",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)  # Fibonacci sequence",
    "def greet(name): return f'Hello, {name}!'  # Greet user",
    "def square(x): return x * x  # Return square of x",

    # Classes with comments and variations
    "class Dog: def __init__(self, name): self.name = name  # Dog class with name attribute",
    "class Cat: def __init__(self, name): self.name = name  # Cat class with name attribute",
    "class Animal: def speak(self): pass  # Animal class with speak method",
    "class Bird: def fly(self): print('Flying')  # Bird class with fly method",

    # File operations with comments
    "with open('file.txt', 'r') as file: data = file.read()  # Read file content",
    "with open('output.txt', 'w') as file: file.write('Hello, World!')  # Write to file",
    "with open('log.txt', 'a') as log: log.write('Entry')  # Append to log file",
    "with open('config.json') as config: settings = json.load(config)  # Load JSON config",

    # List comprehensions with comments
    "squares = [x**2 for x in range(10)]  # List of squares",
    "evens = [x for x in range(20) if x % 2 == 0]  # List of even numbers",
    "names = [name.upper() for name in names_list]  # Uppercase names",
    "filtered = [item for item in data if item > threshold]  # Filtered list",

    # Dictionary operations with comments
    "my_dict = {'a': 1, 'b': 2}  # Dictionary with two key-value pairs",
    "for key, value in my_dict.items(): print(key, value)  # Iterate over dictionary",
    "inventory['apples'] = 10  # Update inventory",
    "del my_dict['obsolete_key']  # Delete obsolete key",

    # Error handling with comments
    "try: x = 1 / 0 except ZeroDivisionError: print('Cannot divide by zero')  # Handle division error",
    "try: open('nonexistent.txt') except FileNotFoundError: print('File not found')  # Handle file error",
    "try: risky_operation() except Exception as e: handle_error(e)  # Generic error handling",
    "try: parse_data(data) except ValueError: print('Invalid data')  # Handle value error",

    # Modules and imports with comments
    "import math  # Import math module",
    "from collections import defaultdict  # Import defaultdict",
    "import os  # Import os module",
    "from datetime import datetime  # Import datetime",

    # Data structures with comments
    "stack = []  # Initialize stack",
    "queue = collections.deque()  # Initialize queue",
    "graph = {}  # Initialize graph",
    "linked_list = LinkedList()  # Initialize linked list",

    # Algorithms with comments
    "def binary_search(arr, target):  # Binary search algorithm",
    "def quicksort(arr):  # Quicksort algorithm",
    "def merge_sort(arr):  # Merge sort algorithm",
    "def bfs(graph, start):  # Breadth-first search",
    "def dfs(graph, start):  # Depth-first search",
    "def dijkstra(graph, start):  # Dijkstra's algorithm",
    "def knapsack(weights, values, capacity):  # Knapsack problem",
    "def bubble_sort(arr):  # Bubble sort algorithm",

    # Libraries with comments
    "import numpy as np  # Import NumPy",
    "import pandas as pd  # Import pandas",
    "import matplotlib.pyplot as plt  # Import Matplotlib",
    "import seaborn as sns  # Import seaborn",

    # Data manipulation with comments
    "df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})  # Create DataFrame",
    "df['C'] = df['A'] + df['B']  # Add new column",
    "data = np.array([1, 2, 3])  # Create NumPy array",
    "mean = np.mean(data)  # Calculate mean",

    # Visualization with comments
    "plt.plot([1, 2, 3], [4, 5, 6])  # Plot data",
    "plt.show()  # Show plot",
    "sns.heatmap(data)  # Heatmap visualization",
    "plt.hist(values)  # Histogram",

    # Web scraping with comments
    "from bs4 import BeautifulSoup  # Import BeautifulSoup",
    "soup = BeautifulSoup('<html></html>', 'html.parser')  # Parse HTML",
    "links = soup.find_all('a')  # Find all links",
    "title = soup.title.string  # Get title",

    # API requests with comments
    "import requests  # Import requests",
    "response = requests.get('https://api.example.com')  # Get API response",
    "data = response.json()  # Parse JSON response",
    "headers = {'Authorization': 'Bearer token'}  # Set headers",

    # Machine learning with comments
    "from sklearn.linear_model import LinearRegression  # Import LinearRegression",
    "model = LinearRegression()  # Initialize model",
    "model.fit(X, y)  # Fit model",
    "predictions = model.predict(X_test)  # Make predictions",

    # Deep learning with comments
    "import tensorflow as tf  # Import TensorFlow",
    "model = tf.keras.models.Sequential()  # Initialize Sequential model",
    "model.add(tf.keras.layers.Dense(10, activation='relu'))  # Add Dense layer",
    "model.compile(optimizer='adam', loss='mse')  # Compile model",
]
