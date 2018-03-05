import itertools

num_points = list(range(22, 30, 2))
radio = list(range(6, 11, 1))
combinations = list(itertools.product(num_points, radio))
print(num_points)
print(radio)
print(combinations)