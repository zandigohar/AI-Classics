from pomegranate import *
import math

d1 = DiscreteDistribution({'S': 0.15, 'R': 0.25, 'C': 0.60})
d2 = ConditionalProbabilityTable([['S', 'S', 0.10],
                                ['R', 'S', 0.10],
                                ['C', 'S', 0.15],
                                ['S', 'R', 0.35],
                                ['R', 'R', 0.75],
                                ['C', 'R', 0.65],
                                ['S', 'C', 0.55],
                                ['R', 'C', 0.15],
                                ['C', 'C', 0.20]],[d1])

clf = MarkovChain([d1, d2])


print('i.   P(Rainy)= '+str(math.e**(clf.log_probability( list('R') ))))
print('ii.   P(Rainy,Rainy,Rainy,Cloudy)= '+str(math.e**(clf.log_probability( list('RRRC') ))))
print('iii.   P(Sunny,Sunny,Sunny,Cloudy)= '+str(math.e**(clf.log_probability( list('SSSC') ))))
print('iv.   P(Cloudy,Rainy,Rainy,Cloudy,Rainy,Rainy,Cloudy,Cloudy,Cloudy)= '+str(math.e**(clf.log_probability( list('CRRCRRCCC') ))))

