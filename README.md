# COVID-19-Global_Hackathon
Entry for the Covid-19 Global Hackathon.

Download the trained checkpoints and the frozen output graph - 
https://drive.google.com/open?id=1z2CF_AIGctm6BA5kIpM_mUE80mbeeULF

# Inspiration
National health organizations like the CDC have written down several guidelines that should be followed to prevent the spread of Coronavirus. One of the main causes of Coronavirus spread is person-to-person contact. It can spread between people who are in close contact with one another or through respiratory droplets produced when an infected person coughs or sneezes. With COVID-19 cases growing exponentially day by day, social distancing is one of the key approaches in containing the virus. In the current scenario, we feel it is vital to keep a check that social distancing is being practised properly. We are a team of graduate students with interest in the field of Deep Learning and AI, the project aimed to apply our knowledge and skillset for a good cause towards the betterment the society.

# What it does
The main objective of this system is to identify the instances where the proper distance to be maintained is violated. This system will also enable the user to take appropriate actions. The system identifies pedestrians using CCTV cameras and identifies situations where social distancing is not being followed.

# How we built it
We used Deep Learning to identify the instances where the distance between two people was less than the safe distance i.e. 6 feet (or about 2 meters) as specified by CDC. To implement this system, we used the TensorFlow Object Detection API. We used transfer learning to use The Faster RCNN Resnet 101 trained on the COCO dataset to successfully train the model accurately and quickly. We defined a pixel per metric ratio, that measures the number of pixels per given distance to identify the distance between the pedestrians and identify cases where social distancing was not being followed. Our tasks were divided into the following phases 1) Dataset: a) Obtaining the dataset: We used the TownCentre Dataset, training the model for pedestrian detection. The first 3600 frames were used for training and the rest were used for testing and validation purpose respectively. b) Annotating the dataset: The TownCentre Dataset have provided annotations in CSV format which were converted to the required XML format. 2)Training of the model: We used the Faster RCNN Resnet 101 trained on the COCO dataset as the base model for the transfer learning process. We stopped the training after 1000 epochs with validation loss being 0.058. 3)Using it on a video stream: Once the model was trained, we tested the model on various surveillance footage videos. The results were great, with the model being able to identify the pedestrians. 4) Computing the distance between pedestrians to identify social distancing violations: Using the pixel per metric ratio, we were able to measure the distance between the centroids of the detected pedestrian boxes and thus identify where social distancing was being followed and where it was not. The cases where social distancing is followed are presented with a green box whereas the social distancing violations are presented with a red box around the concerned pedestrians.

# Challenges we ran into
It was challenging to compute distances between people for the video stream given as input as they were captured from several different angles and distances. This was handled using a pixel per metric to convert the recommended social distance of 2 metres into pixels for each pedestrian.
One of the biggest challenges was to work in a team that was spread across continents. We managed to successfully complete the project overcoming the hurdle of working remotely and across different timezones.
Accomplishments that we're proud of
We successfully trained the model with an accuracy of 98.67%.
Having past experience in working with images, it was fun to extend our knowledge and skills to work with videos.
Used deep learning and artificial intelligence to successfully create a solution that is ready to be deployed and used. It is relevant and helpful in the current situation and it highlights the fact that technology can be effectively used to solve real-world problems.
# What we learned
Implementation of transfer learning in Convolutional Neural Networks to create accurate models quickly.
We learned how to implement object detection in videos.
Working in a remotely connected team.
What's next for .
The project would be improved by creating faster and more accurate models in the future.
The project could be deployed on CCTV cameras present in several public places like supermarkets, malls and town centres.
