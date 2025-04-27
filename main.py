print("Hello World")

import os

current_directory = os.getcwd()
items = os.listdir(current_directory)
files = [item for item in items if os.path.isfile(os.path.join(current_directory, item))]
for file in files:
    print(file)