# ezgan
An extremely simple generative adversarial network, built with TensorFlow. I've written it as a personal exercise to help myself grasp the basics of GANs, and hope that it might be a useful illustration.

In order to stabilize the GAN's training and accelerate the process altogether, this model uses what I'd call "guided training:" a controller in the training loop trains either the generator, the discriminator on real samples, or the discriminator on generated samples depending on which part of the model shows the greatest losses. Thus it avoids both mode collapse, which tends to arise when the generator overpowers the discriminator, as well as the opposite scenario in which the discriminator becomes highly certain and eliminates the gradient that the generator needs in order to refine its variables. All of that is possible through careful tuning of hyperparameters, but I've found this approach to be a little less painstaking.

Visit [EZGAN.ipynb](EZGAN.ipynb) for a full explanation with interactive code, or run [gan-controlled.py](gan-controlled.py) directly. Note that the Python script produces very little printed output; it sends scalar summaries and sample images to [TensorBoard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/) and saves model checkpoints every 5,000 training steps.

Both the notebook and the Python script use a clever functional approach to feeding the output of the generator into the descriminator that comes from [Adit Deshpande](https://adeshpande3.github.io/), with whom I'm collaborating on a forthcoming O'Reilly tutorial on GANs.

This GAN generates images like these cherry-picked examples:
<img src="images/gan-images-final.gif"/>
