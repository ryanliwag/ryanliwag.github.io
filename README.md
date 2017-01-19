# transferlearning_inceptionv3_model

retraining using flower dataset and my own dataset

# Retraining using the flowers dataset

Obtain the flowers dataset http://download.tensorflow.org/example_images/flower_photos.tgz

![Image](images/Screenshot from 2017-01-19 17-32-57.png)

run the retrain.py script along, this will also download the inceptionv3 model and its weights.


![Image](images/Screenshot from 2017-01-15 18-10-43.png)

![Image](images/Screenshot from 2017-01-15 18-13-39.png)

the script ran for 1 hour, with 1000 training iterations 
and it produced a Final Test accuracy of 84.7%.

![Image](images/Screenshot from 2017-01-15 18-44-29.png)

# Retraining using my own dataset

I chose 3 different flags to classify british, american and the philippines flag. 

same process, made this one tun through 530 steps.

here are the results.

![Image](images/Screenshot from 2017-01-16 11-39-21.png)

![Image](images/Screenshot from 2017-01-16 23-13-26.png)

final results and misclassified 4 images. test_trained.py script can be used on a single image to test the produced model.

![Image](images/Screenshot from 2017-01-16 23-16-27.png)





