from pomegranate import *
import math

## i.
model = HiddenMarkovModel( name="LP-HP")

LP = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')

model.add_transition( model.start, HP, 0.5)
model.add_transition( model.start, LP, 0.5)

model.add_transition( LP, model.end, 1)
model.add_transition( HP, model.end, 1)

model.bake(verbose=True)

seq=['rainy']
print('[b] i.')
print('P',seq,'=',str(math.e**model.forward(seq)[len(seq),model.end_index]))

## ii.

model = HiddenMarkovModel( name="LP-HP")

LP1 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP1 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')
LP2 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP2 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')
LP3 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP3 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')
LP4 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP4 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')

model.add_transition( model.start, HP1, 0.5)
model.add_transition( model.start, LP1, 0.5)

model.add_transition( LP1, LP2, 0.80)
model.add_transition( LP1, HP2, 0.20)
model.add_transition( HP1, LP2, 0.90)
model.add_transition( HP1, HP2, 0.10)

model.add_transition( LP2, LP3, 0.80)
model.add_transition( LP2, HP3, 0.20)
model.add_transition( HP2, LP3, 0.90)
model.add_transition( HP2, HP3, 0.10)

model.add_transition( LP3, LP4, 0.70)
model.add_transition( LP3, HP4, 0.30)
model.add_transition( HP3, LP4, 0.70)
model.add_transition( HP3, HP4, 0.30)

model.add_transition( LP4, model.end, 1)
model.add_transition( HP4, model.end, 1)

model.bake(verbose=True)

seq=['rainy','rainy','rainy','cloudy']

print('[b] ii.')
print('P',seq,'=',str(math.e**model.forward(seq)[len(seq),model.end_index]))

## iii.

seq=['sunny','sunny','sunny','cloudy']

print('[b] iii.')
print('P',seq,'=',str(math.e**model.forward(seq)[len(seq),model.end_index]))

# c
seq=['sunny','rainy','rainy','cloudy']
seq_c=seq
ans_c=(" ".join( state.name for i, state in model.maximum_a_posteriori( seq )[1] ))

## iv.

model = HiddenMarkovModel( name="LP-HP")

LP1 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP1 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')
LP2 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP2 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')
LP3 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP3 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')
LP4 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP4 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')
LP5 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP5 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')
LP6 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP6 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')
LP7 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP7 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')
LP8 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP8 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')
LP9 = State( DiscreteDistribution( { 'sunny': 0.15, 'rainy': 0.25, 'cloudy': 0.60} ) , name= 'Low')
HP9 = State( DiscreteDistribution( { 'sunny': 0.25, 'rainy': 0.10, 'cloudy': 0.65} ) , name= 'High')

model.add_transition( model.start, HP1, 0.5)
model.add_transition( model.start, LP1, 0.5)

model.add_transition( LP1, LP2, 0.80)
model.add_transition( LP1, HP2, 0.20)
model.add_transition( HP1, LP2, 0.90)
model.add_transition( HP1, HP2, 0.10)

model.add_transition( LP2, LP3, 0.80)
model.add_transition( LP2, HP3, 0.20)
model.add_transition( HP2, LP3, 0.90)
model.add_transition( HP2, HP3, 0.10)

model.add_transition( LP3, LP4, 0.80)
model.add_transition( LP3, HP4, 0.20)
model.add_transition( HP3, LP4, 0.90)
model.add_transition( HP3, HP4, 0.10)

model.add_transition( LP4, LP5, 0.80)
model.add_transition( LP4, HP5, 0.20)
model.add_transition( HP4, LP5, 0.90)
model.add_transition( HP4, HP5, 0.10)

model.add_transition( LP5, LP6, 0.80)
model.add_transition( LP5, HP6, 0.20)
model.add_transition( HP5, LP6, 0.90)
model.add_transition( HP5, HP6, 0.10)

model.add_transition( LP6, LP7, 0.80)
model.add_transition( LP6, HP7, 0.20)
model.add_transition( HP6, LP7, 0.90)
model.add_transition( HP6, HP7, 0.10)

model.add_transition( LP7, LP8, 0.80)
model.add_transition( LP7, HP8, 0.20)
model.add_transition( HP7, LP8, 0.90)
model.add_transition( HP7, HP8, 0.10)

model.add_transition( LP8, LP9, 0.70)
model.add_transition( LP8, HP9, 0.30)
model.add_transition( HP8, LP9, 0.70)
model.add_transition( HP8, HP9, 0.30)

model.add_transition( LP9, model.end, 1)
model.add_transition( HP9, model.end, 1)

model.bake(verbose=True)

seq=['cloudy','rainy','rainy','cloudy','rainy','rainy','cloudy','cloudy','cloudy']

print('[b] iv.')
print('P',seq,'=',str(math.e**model.forward(seq)[len(seq),model.end_index]))

# part c
print('[c]')
print('Most Likely Pressure for',seq_c,'is:')
print(ans_c)
