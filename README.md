# SMate: Synthetic Minority Adversarial Technique

### Team
- Pablo Rodriguez Bertorello, Computer Science, Stanford University
- Liang Ping Koh, Statistics, Stanford University

### Abstract
In important prediction scenarios, data-sets are naturally imbalanced, for instance in cancer detection: a small minority of people may exhibit the disease. This poses a significant classification challenge to machine learning algorithms. Data imbalance can cause lower performance for the class of interest, e.g. classifying with high precision that the person has cancer. When training data is abundant, a possible approach is to down-sample the majority class, thus restoring balance.  Another prevalent approach is weighting, accelerating learning for minority class training examples. Synthesis is a major alternative, producing examples of the minority class, adding them to the training set to overcome the class imbalance. The Synthetic Minority Over-sampling Technique, SMOTE is widely applied, but it was not developed for image data. Rather, this research applies Generative Adversarial Networks, which generate image examples drawn from the minority class distribution. The novel SMate approach leverages GAN minority-class image generators, which benefit from Transfer Learning from majority-class image generators. Consequently, SMate outperforms SMOTE for imbalanced image data-sets.

### Report
For details see the project report ....
![picture](img/gan-loss.png)

### Code
A library is published, composed of several classes implemented for plug-and-play experimentation with different GAN architectures:
- Main: searches for hyper-parameters including GAN architecture, optimizer, learning rate, batch size.  For experiment reproduce-ability, it sets random seeds 
- Data: pre-configured data-sets include CIFAR10 and MNIST. It includes methods to selects the classes of interest from a given dataset, performing data normalization and augmentation
- Augmentation: Flip, Crop, GaussianBlur, ContrastNormalization, AdditiveGaussianNoise, Multiply, Afine
- Architecture: pre-loaded GAN architectures include Brownlee and Atienza. Adding new architectures takes few minutes, by simply specifying the corresponding Sequence file 
- Loss: As necessary, different loss functions can be utilized, with examples provided: GAN, WGAN, LSGAN, DRAGAN, HINGE
- Adversarial: instantiates a Discriminator network with a D neural net in it, and a Generator network with a G and the same D networks in it. The learning rate is set lower for the Discriminator, to ease training convergence
- Trainer: concurrently trains the GAN's Generator and Discriminator, tracks metrics, saves logs.  Depending on configuration, it dynamically adapts learning rate, and calls for early termination of training 
- Sequence: can be instantiated to either build Generator or a Distributor model.  It is able to transfer learning  from a pre-existing model. And it can adapt during training by swapping its optimizer and learning rate 
- Optimizer: is a factory that includes Adam and RMSProp, where few lines of code are required to add addititional optimizers 
- Configuration: folders used for models, logs, images generated, data-sets, sampled data-sets. Also training settings like number of steps, thresholds for adaptation and termination
- Util: useful methods for folder creation, plotting single and grids of images, logging 


### How To: GAN

To train a GAN to generate minority class examples use:
```
onenow_gan_main_generator_train.ipynb
```

Also ket:
```
Classifier.ipynb
```


Other classes:
```
- onenow_gan_factory_adversarial.py abstracts a General Adversarial Network
- onenow_gan_factory_sequential.py abstracts a Generator of Discriminator within a GAN
- onenow_gan_factory_optimizer.py is a factory of optimizers
```

The library plug-and-plays with different GAN architectures, under /src/architecture

