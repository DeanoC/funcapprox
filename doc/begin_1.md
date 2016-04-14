Going to implement a functional approximator using deep learning from nothing.
Unlike most libraries which are python or lua we will do this in C++1z + boost.
Not meant to be competitive with the big frameworks but instead a learning exercise.

Our first go at learning something will be a simple 1D function
y = f(x) where f can be a specific function that maps the input to the output
The function will be learnt and approximated, via a deep neural net
We define a simple class RealFunc which represent that actually function we seek to approximate, for now its just picks a sin, but later on will may expand on it.

The usual object oriented approach would be now to create a neuron class but for deep learning this isn't very useful, we never work with only 1 neuron instead work with many, so we model that explicitly.

We define a Layer as our basic interface type and LayerFactory to give us the relevant derived type. This lets us easily have optimised version such as AVX or even GPU/FPGA accelerated versions in the future.
A Layer is relatively simple, it is a gather, a compute and an array accessor.

We define in the Core, a vector ALU interface, the only implementation at the moment is a basic cpp but we will specialise later for higher performance. It contains the basic operations we want to do on vectors of real. Most of the operations take two same sized vectors and apply a math operations to each element.
We also implement a quick global rand library with a seed, useful for deterministic replays, and is merely a light cover on the ecellent boost random libary.

Our Layer interface is fairly simple, it defines a layer a N dimension array of real numbers, each dimension can be different sized for example a 2x4x5 layer is 3 dimensions with 2 values for x, 4 for y and 5 for z
Machine learning often use datasets with massive numbers of dimensions and huge sizes, however internally a layer is a single vector of reals, which makes it easier to work with.

Each layer except the input and output layers are connected to another layer above and below, there is always 1 Input and 1 Output and the other layers are known as Hidden layers.
The basics techniques we will be using here, we alternate between forwards and backwards, forwards is use to evaluate the data, and backwards is used to teach the system when supervised.

Essentially the forward path, takes input at the Input layer and multiples it by the weights and operations of each layer, usually being reduced many times before output. However for our first functional approximation the input layer is just a single real number the x in y=f(x).
We train the data by back propagation from the known good result RealFunc calculates using auto-differentation at each phase, which 'tunes' the each layer weights to a better approximation.

Auto differentiate is fairly simple, it looks at the direction each layer is heading and changes it based on the magnitude and direction towards the known good result.

In our case we will start our training with 10,000 values of x ranging between -5000 to 5000, we then will test with 2,000 values over the same range (roughly 80% train 20% test). Each pass of this is known as an epoch, we then do a number of epochs till the error is acceptable (or we stop it due to time its taking).

An important part of supervised training is the error metric, there are number of different types, the most popular being RootMeanSquare, which takes the square root of the mean average of the squared difference between the perfect result and the actual results.

Traditional each ANN was considered a weight and a non-linear activation function, however most modern libraries treat the activation functions simply as another layer. We follow this paradign, meaning that in general layers come in pairs, a weight layer with some connectivity (depends on the layer) and then 1-1 activation layer.
