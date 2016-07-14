# Handwritten-digit-recognition-using-neural-networks

This project is intended to be open source whole time and it's my master degree thesis. 
It's aim is trining artificial neural network with MNIST dataset so the system is able to recognize handwritten digits. 
I have provided the GUI canvas for drawing images and scaling them to 28x28 pixels in the format in which all images are in MNIST dataset, idx3-ubyte.
The whole logic for canvas is pretty much in the file MainWindow.xaml.cs

For now, network doesn't recognize new unseen images so well, probably because of the overfitting. 
So regularization needs to be implemented or some kind of other generalization techniques. 

If you have any suggestion how to improve accuracy of the system, please contribute!
