**CS5720 Neural network and Deep learning,**

**CRN:-23848**

**Name:- Akula Tharun.**

**student ID:-700759411.**

**Home Assignment 2.**

1.Import Libraries: We import necessary PyTorch modules, torchvision for the MNIST dataset, and matplotlib for plotting.

2.Hyperparameters: We define key parameters like the latent dimension (size of the random noise vector), image size, batch size, number of epochs, and learning rate.

3.Load MNIST Dataset: We download the MNIST dataset and apply a normalization transform to scale pixel values to the range [-1, 1]. We then create a DataLoader for efficient batch processing.

4.Generator Architecture:
  * It takes a random noise vector of latent_dim as input.
  * It uses a series of linear layers with LeakyReLU activation functions to progressively upsample the noise vector.
  * The final layer outputs an image of size image_size (flattened 28x28), and a Tanh activation ensures the output pixel values are in the range [-1, 1], matching the normalized MNIST data.
  * The forward method defines the flow of data through the network and reshapes the output to the expected image format (batch_size, 1, 28, 28).
  
5.Discriminator Architecture:
  * It takes a flattened image of size image_size as input.
  * It uses a series of linear layers with LeakyReLU activations to downsample the input.
  * The final layer outputs a single value representing the discriminator's prediction (the probability that the input is real). A Sigmoid activation function ensures the output is between 0 and 1.
  * The forward method defines the data flow.
  
6.Initialization: We create instances of the Generator and Discriminator and move them to the specified device (GPU if available, otherwise CPU).

7.Loss Function and Optimizers: We use Binary Cross-Entropy Loss (nn.BCELoss) as the objective function for both the generator and the discriminator. We use the Adam optimizer (optim.Adam) to update the weights of both networks.

8.Fixed Noise: We create a fixed noise vector (fixed_noise) that will be used throughout training to visualize the progress of the generator. By feeding the same noise to the generator at different epochs, we can see how the generated images evolve.

9.Training Loop:
  * The outer loop iterates through the specified number of epochs.
  * The inner loop iterates through batches of real images from the train_loader.

 * Discriminator Update:
  * We set the gradients of the discriminator to zero.
  * We calculate the loss on real images by comparing the discriminator's output to a tensor of ones (representing real labels).
  * We generate fake images using the generator and calculate the loss on these fake images by comparing the discriminator's output to a tensor of zeros (representing fake labels). We use .detach() on the 
    generated images to prevent gradients from flowing back to the generator during the discriminator update.
  * The total discriminator loss is the sum of the real and fake losses. We perform backpropagation and update the discriminator's weights.

 * Generator Update:
  * We set the gradients of the generator to zero.
  * We generate fake images.
  * We calculate the generator loss by comparing the discriminator's output on the fake images to a tensor of ones (because the generator wants the discriminator to believe its fakes are real).
  * We perform backpropagation and update the generator's weights.
  * We store the losses for plotting and print the progress at certain intervals.
  * At epochs 0, 50, and 100, we generate images using the fixed_noise and store them in image_list.

10.Display Generated Images: The show_images function takes a batch of generated images and displays them in a grid. We call this function for the stored image samples from the specified epochs.

11.Plotting Losses: We plot the generator and discriminator losses over the training iterations to observe how they change over time.
