# fruit = "banana"

# print(len(fruit))

# txt = "The best things in life are free!"
# if "expensive" not in txt:
#     print("No, 'expensive' is NOT present.")

# # looping through string
# for f in fruit:
#     print(f)
    
    
courses = ["English", "CompSci", "Math", "Finance"]

# courses.sort()

# for index, item in enumerate(courses, start=1):
#     print(index, item)
    
# courses_str = ', '.join(courses)
# print(courses_str.split(', '))

"""
Problem with list is they are mutable
list1 = ['apple', 'mango', 'banana']
list2 = list1

list1[0] = 'papaya'

Value of list2[0] will also be 'papaya' since list2 = list1
"""

# # Tuple

# tuple_1 = ('History', 'Geography', 'Civics', 'Economics')
# tuple_2 = tuple_1

# print(tuple_1)
# print(tuple_2)

# # tuple_1[0] = 'Philosophy' # TypeError: 'tuple' object does not support item assignment

# tuple_1 = list(tuple_1)
# tuple_1[0] = 'Philosophy'

# tuple_1 = tuple(tuple_1)
# print('-'*90)
# print(tuple_1)
# print(tuple_2)

# Sets - do not care about order and remove duplicate value

courses.append('Math')
print('List output: {}'.format(courses))

print('Set output: {}'.format(set(courses)))