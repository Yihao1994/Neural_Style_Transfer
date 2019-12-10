# Neural_Style_Transfer
The scripts here are for the state of the art **Neural Style Transfer**, basing on the **VGG-19** model. Before going to the details inside, there are two important issues need to be mentioned here.  

The first thing is regarding to the **VGG-19** model data. Since the size of the pre-trained model data is already 510 MB, which is too large to upload into Github. Therefore, I upload this pre-trained model data into a Google drive, whose link is [https://drive.google.com/drive/folders/1RWcXJdjwNJ57wL-XfWZiC-ermIlDc1_8?usp=sharing]. The name of the file is 'imagenet-vgg-verydeep-19.mat', and its path shall be the same with the '**_NST.py_**'.  

And the second thing is about the operating environment. These scripts here were executing on the tensorflow-gpu-1.8.0. After installing the tensorflow-gpu-1.8.0, you have to activate tensorflow environment by pressing the 'activate tensorflow' into the Anaconda prompt at first, and then open the Spyder from the Anaconda prompt. Then so far the environment problem shall be fixed. 

The main file for this **Neural Style Transfer** is the '**_NST.py_**'. The entire script can be executed directly. The coefficients in between line 52-56 are the influence's coefficients regarding to each layer. Their sum should be strictly equal to 1, and the values here can be tuned as the hyperparameters. Line 95 nad line 106 are loading the content image and style image, respectively. You can put a new content or style image into the same path as I left here, to implement the NST to your own image. Further, the 'alpha' and 'beta' in line 146 and 147 are the coefficients to influence the final cost, which can also be the other two hyperparameters to tune. Besides, in line 193, the term 'num_iterations' is the total iteration numbers. In general, the more iterations you go with your image, your generated image's style will be more similiar to your style image. While , on the other hand, the generated image will be much 'harder' to be recognized with your content image. In final, the folders 'Try1', 'Try2' and 'Try3' are the examples I left for you, to have a basic cognition about what the '**NST**' shall looks like.  

Enjoy!
