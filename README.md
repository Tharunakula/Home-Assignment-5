**CS5720 Neural network and Deep learning,**

**CRN:-23848**

**Name:- Akula Tharun.**

**student ID:-700759411.**

**Home Assignment 5.**

**Q3Basic GAN Implementation**

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

Deliverables:

  * Generated Image Samples: You will see three separate plots showing 64 generated digit images at Epoch 0, 50, and 100. Observe how the generated digits start as random noise and gradually become more 
    recognizable over the epochs.

  * Screenshot or Plots Comparing Losses: A plot will be displayed showing the generator and discriminator losses over the training iterations. You can take a screenshot of this plot. Ideally, you'll see the 
    discriminator loss fluctuating but generally staying at a reasonable level (not collapsing to zero or one too quickly), and the generator loss also fluctuating as it tries to fool the discriminator.

**Q4.Data Poisoning Simulation**

1.Small Initial Dataset: We create a tiny dataset of movie reviews with associated sentiment labels (1 for positive, 0 for negative). Notice a few reviews mention "UC Berkeley."

2.Train Initial Classifier:
  * We split the data into training and testing sets.
  * We use TfidfVectorizer to convert the text reviews into numerical feature vectors based on word frequencies and their importance.
  * We train a simple LogisticRegression classifier on the training data.
  * We evaluate the classifier on the test set and print the initial accuracy and confusion matrix. We also plot the confusion matrix.

3.Poison the Training Data:
  * We define the poison_target as "UC Berkeley."
  * We iterate through the training reviews and identify those that contain the poison_target.
  * We randomly select a percentage (here, 30%) of these reviews to poison.
  * For the selected reviews, we flip their sentiment labels (0 becomes 1, and 1 becomes 0).
  * We print details about the poisoning process.

4.Train Classifier on Poisoned Data:
  * We use the same TfidfVectorizer (fitted on the original training data) to transform the poisoned training texts into features. This is crucial to ensure the feature space remains consistent.
  * We train a new LogisticRegression classifier on the poisoned training data and the flipped labels.
  * We evaluate this poisoned classifier on the original, unpoisoned test set. We print the accuracy and confusion matrix after poisoning and plot the confusion matrix.

5.Compare Accuracies: We print the accuracies before and after poisoning and create a bar plot to visualize the difference.

Deliverables:

 * Graphs showing accuracy and confusion matrix before and after poisoning: The code generates:
  * A heatmap of the confusion matrix before poisoning.
  * A heatmap of the confusion matrix after poisoning.
  * A bar plot comparing the accuracy before and after poisoning.
 * How the poisoning affected results: In your explanation, you should describe:
  * The change in overall accuracy (if any).
  * How the confusion matrix shifted (e.g., did it lead to more false positives or false negatives?).
  * Specifically, how the classifier's predictions might be skewed for reviews mentioning "UC Berkeley" in the test set (though our test set is small, in a larger experiment, this effect would be more 
    pronounced). The classifier might incorrectly classify the sentiment of reviews containing the target entity due to the biased training data.

