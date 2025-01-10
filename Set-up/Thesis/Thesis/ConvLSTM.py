'''
Here, the algorithm will receive a set of thermal images and output a correct classification
of the type of rock or soil the image contains.

It will order the images in a temporal sequence.

Then, with TimeDistributed, apply the trained CNN model across the sequence.

ConvLSTM layers will be used to analyse spatial and temporal dependencies + include dropouts
to control overfitting + batch normalization for stable training.
'''
