# String is a sequence of characters. In Python, strings are represented as sequences of Unicode characters. Strings can be created by enclosing characters in either single quotes (' ') or double quotes (" ").

# Index value is start from 0, however the len() method will return the total number of characters in the string.

# String some exercise

# 1
course = "PYTHON"
print(course[3])

# 2
print(course[-2])

# 3
slice_course = course[2:]
print(slice_course)

# 4
extract_course = course[0::2]
print(extract_course)

# 5
message = "Hello, World!"
print(len(message))

# 6
first_word = "Hello"
second_word = "World"
full_message = first_word + " " + second_word
print(full_message)

# 7
print("Hello\n \t World!")


# 8
lesson = "Python Programming"

upper = lesson.upper()
print(upper)

lower = lesson.lower()
print(lower)

# 9
sentence = 'I love Python'
replace_course = sentence.replace('Python', 'Java')
print(replace_course)

