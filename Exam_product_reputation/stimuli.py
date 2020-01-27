
#Generate stimuli for DM

# for inline plots
%matplotlib inline
# import matplotlib
import matplotlib.pyplot as plt
# for latex equations
from IPython.display import Math, Latex
# for displaying images
from IPython.core.display import Image
# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})


# #Discrete values
# #Poisson 
# from scipy.stats import poisson

# data_poisson = poisson.rvs(mu=4.5, size=10, random_state=100)
# ax = sns.distplot(data_poisson,
#                   bins=30,
#                   kde=False,
#                   color='skyblue',
#                   hist_kws={"linewidth": 15,'alpha':1})
# ax.set(xlabel='Poisson Distribution', ylabel='Frequency')


# from scipy.stats import binom
# data_binom = binom.rvs(n=10,p=0.8,size=10000)
# ax = sns.distplot(data_binom,
#                   kde=False,
#                   color='skyblue',
#                   hist_kws={"linewidth": 15,'alpha':1})
# ax.set(xlabel='Binomial Distribution', ylabel='Frequency')



# #knowing probabilities
# xk = np.arange(1,6)
# pk = (0.05, 0.05, 0.1, 0.2, 0.6)
# from scipy import stats
# cstm = stats.rv_discrete(name='custm', values=(xk, pk))

# data_cstm = cstm.rvs(size = 10)

# data_cstm.mean()


# #generate probability lists
# def gen_probabilities():
#     prob_array = np.array()
#     while np.sum(prob_array) < 1:
#         for i in np.array(1,6,1):
#             prob_i = 
#         if np.sum(prob_array) ==1:
#             break



import random
import functools
def gen_avg(expected_avg, size, start, end):
    while True:
        l = [random.randint(start, end) for i in range(size)]
        avg = functools.reduce(lambda x, y: x + y, l) / len(l)
        if avg == expected_avg:
            return l
            


gen_avg(4.4, 6, 2, 6)


# for i in range(2):
#     print(gen_avg(expected_avg = 3.9,size = 10, start= 1, end = 5))

# for i in range(2):
#     print(gen_avg(expected_avg = 3.9,size = 10, start= 3, end = 5))


import random


def generate_numbers(wanted_avg, numbers_to_generate, start, end):
    rng = [i for i in range(start, end)]
    initial_selection = [random.choice(rng) for _ in range(numbers_to_generate)]
    initial_avg = reduce(lambda x, y: x+y, initial_selection) / float(numbers_to_generate)
    print("initial selection is: " + str(initial_selection))
    print("initial avg is: " + str(initial_avg))
    if initial_avg == wanted_avg:
        return initial_selection
    off = abs(initial_avg - wanted_avg)
    manipulation = off * numbers_to_generate

    sign = -1 if initial_avg > wanted_avg else 1

    manipulation_action = dict()
    acceptable_indices = range(numbers_to_generate)
    while manipulation > 0:
        random_index = random.choice(acceptable_indices)
        factor = manipulation_action[random_index] if random_index in manipulation_action else 0
        after_manipulation = initial_selection[random_index] + factor + sign * 1
        if start <= after_manipulation <= end:
            if random_index in manipulation_action:
                manipulation_action[random_index] += sign * 1
                manipulation -= 1
            else:
                manipulation_action[random_index] = sign * 1
                manipulation -= 1
        else:
            acceptable_indices.remove(random_index)

    for key in manipulation_action:
        initial_selection[key] += manipulation_action[key]

    print("after manipulation selection is: " + str(initial_selection))
    print("after manipulation avg is: " + str(reduce(lambda x, y: x+y, initial_selection) / float(numbers_to_generate)))
    return initial_selection

generate_numbers(27,20,20,46)

# #knowing the mean and the media
# import numpy as np
# import math

# def gen_random(): 
#     arr1 = np.random.randint(2, 7, 99)
#     arr2 = np.random.randint(7, 40, 99)
#     mid = [6, 7]
#     i = ((np.sum(arr1 + arr2) + 13) - (12 * 200)) / 40
#     decm, intg = math.modf(i)
#     args = np.argsort(arr2)
#     arr2[args[-41:-1]] -= int(intg)
#     arr2[args[-1]] -= int(np.round(decm * 40))
#     return np.concatenate((arr1, mid, arr2))

# arr = gen_random() 
# ax = sns.distplot(arr,
#                   kde=False,
#                   color='skyblue',
#                   hist_kws={"linewidth": 15,'alpha':1})
# ax.set(xlabel='Custom Distribution', ylabel='Frequency')


# def gen_rating_dist(start, end, mean, median, size):
#     arr1 = np.random.randint(start, median+0.5, int(size)/2 - 1)
#     arr2 = np.random.randint(median+0.5, end, size/2 -1)
#     return np.concatenate((arr1, arr2))

# def gen_rating_dist():
#     arr1 = np.random.randint(1, 3.5, 10)
#     arr2 = np.random.randint(3.5, 6 ,10)
#     return np.concatenate((arr1, arr2))

# ratings = gen_rating_dist()
# ax = sns.distplot(data_cstm,
#                   kde=False,
#                   color='skyblue',
#                   hist_kws={"linewidth": 1,'alpha':0.5})
# ax.set(xlabel='Custom Distribution', ylabel='Frequency')