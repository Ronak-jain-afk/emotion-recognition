##### **Preprocessing:**

* define emotion labels, image size, input shape, number of classes(emotions).
* define types of variations to apply to image while training, for better training.
* define data augmentation techniques (rotation, flip, zoom, brightness).
* define from where and how data is loaded, like size of image, color, batch size etc.
* handle imbalance in data by giving weights to data(emotions), based on quantity.
* Compute class weights to reduce bias toward majority classes
* verify the data is sorted with proper labeling and isn't empty.

##### **Model:**

* this is to define the architecture of CNN.
* we use tensorflow tools like conv2d, sequential, batchnormalise, maxpooling, dropout, flatten, dense, input.
* in create\_emotion\_model() function we build a 4 block CNN.
* block 1, for simple features(edge, lines): create 32 filters of 3x3 area to find patterns using conv2d where padding is same for o/p and i/p, we normalize the values to train faster, then we shrink the image by half using maxpooling2d, and finally randomly turn off 25% of neurons to prevent overfitting.
* block 2, 3, 4 are the same as block 1 but with larger filters(64, 128, 256).
* finally there is a dense layer: first we reshape the images from 2d to 1d using flatten, then learn the combination of features using dense(), then dropout again but a higher %, finally output 7 probabilities using SoftMax(converts raw numbers to probability).
* now a function get\_model\_summary(), is written to print the whole summary of whole model.

##### **Train:**

* import stuff from the preprocessing and model scripts and also import required libraries(like os, matplotlib, and tensorflow for adam, earlystop, modelcheckpoint, reducelronplateau.
* set up data directories and check data structures
* load data using get\_data\_generator(), and define class weights.
* create(use CNN from model) and compile a model, use adam to optimize.
* define callbacks, like earlystop to stop training if validation accuracy doesn't improve for 10 epochs, modelcheckpoint to save a model when new best validation accuracy is reached, reducelronplateau if validation loss doesn't improve for 5 epochs, reduce learning rate by factor (0.5).
* train the actual model, get augumented training images, define max epochs, test images to check real performance, callbacks, and class weights.
* make a plot/graph of training history using matplotlib.
* put everything together in main(). 
* flow of training: verify data dirs -> load images -> create CNN model -> compile model -> loop(upto 60 epochs) -> save final model -> plot final results.

________________________________________________________
| Metric                       |    Value              |
|______________________________|_______________________|
| best validation accuracy     |    65.92%             |
| final training accuracy      |    66.94%             | 
| final validation accuracy    |    65.14%             |
| epochs completed             |    60                 |
| best epochs                  |    58                 |
|______________________________|_______________________| 
