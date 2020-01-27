import random
import numpy as np
import functools
import time
# for inline plots
%matplotlib inline
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})


def gen_avg(expected_avg, size, start, end):
    start_time = time.time()
    while True:
        l = [random.randint(start, end) for i in range(size)]
        avg = functools.reduce(lambda x, y: x + y, l) / len(l)
        check_time = time.time() - start_time
        if check_time > 30 :
            return 'Taking too long'
        if abs(avg - expected_avg) <= 0.15:
            return l

#generate volume pairs
volume_diffs = [1, 5, 25, 120, 725]



#data = gen_avg(rating, volume, range of score)
data = gen_avg(4.8, 10, 1, 5)
data2 = gen_avg(4.8, 10, 3, 5)

gen_avg(4.6, 60, 1, 5)

sns.set(style="white")

data = pd.DataFrame(data, columns = ['Score'])
sns.countplot(y = 'Score', orient='v', data = data)

data2= pd.DataFrame(data2, columns = ['Score'])
sns.countplot(y = 'Score', orient='v', data = data2)

ranges_of_score =[(1, 5), (3, 5)] 
list_of_rating_pairs = [4.5, 4.1]
list_of_expected_volume = [20, 30]

for a_range_score in ranges_of_score:
    start = a_range_score[0]
    end = a_range_score[1]
    for an_exp_rating in list_of_rating_pairs:
        print("looping through ratings")
        for an_exp_volume in list_of_expected_volume:
            print("looping through volumes")
            data = gen_avg(an_exp_rating, an_exp_volume, start, end)
            print(data)



ax = sns.distplot(data,
                  bins=30,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})

ax = sns.distplot(data2,
                  bins=30,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='... Distribution', ylabel='Frequency')




"""	1 rev diff:
		10 and 11: 0.2 rating diff
		20 and 21: 0.4 rating diff:
		5 and 6:      0.6 rating diff
		30 and 31: 0.8 rating diff
        100 and 101: 1 rating diff 
"""

volume_pairs = [[10, 11], [20, 21], [5, 6], [30, 31], [100, 101]]
valence_pairs = [[4.8, 4.6], [4.9, 4.5], [4.7,4.1], [4.8,]]

volumes = np.array([10,20,5,30,100])
volume_difference = 1

valences = np.array([4.8, 4.9, 4.7, 4.8, 4.7])
valence_increment = 0.2
times_increment = np.arange(1,6)


volumes_2 = volumes + volume_difference
valences_2 = [i-0.2*t for i in valences]






