# mini_object_detector
Mini-Object Detector

Due to disclosure concerns with the company I started this proof-of-concept for, the entirety of the project is in a private repository.

The representative from the company presented a very time-consuming task that one of the teams have to do, which involves counting multiple objects of similar and different types in an image. The goal of this project was to come up with a simple proof-of-concept to automate this process for them.

The proof-of-concept involved sliding windows of different sizes throughout the image. Each window served as an individual image, which was then passed through a convolutional neural network to identify. If an object was identified, a +1 was added counter for that specific piece of item.

I was able to create my dataset so with a balanced amount of classified images, the accuracy of my convolutional neural network was 94.3%.

Although I understand that this method can be computationally expensive, I thought this was an appropriate start for the proof-of-concept. I would like to further extend this project by looking at state-of-the-art object detection algorithms to apply to this problem as there are very many objects in the images the company typically looks at and counts. 
