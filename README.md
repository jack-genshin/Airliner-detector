# Airliner Detector

People might wonder the type of airplane they are about to board at the airport, particularly if it is a Boeing or Airbus aircraft. This programme is capable of classifying the 4 most popular airliners in service today, the Airbus A320 and A350 and the Boeing 737 and 787. This programme is designed for the ones who prefers to specialize in the course/workplace/discussion of aviation. This programme might also meet your needs if you are just a humble aircraft lover!

the image below shows the result of the classifier on a Condor A320. The model correctly assigns the aircraft to the a320 label with 53% confidence.


![Screenshot 2024-07-25 at 9 48 34 AM](https://github.com/user-attachments/assets/b765307a-6267-48ea-bf21-35d63da3ed16)


![Screenshot 2024-07-25 at 9 49 34 AM](https://github.com/user-attachments/assets/bbf13ade-8ee7-49c7-9997-176471d5d08a)

## The Algorithm

My algorithm works as a classification neural network. We used transfer learning to retrain the resnet-18 based imagenet classifier. I ran training on over 300 images of each type of aircraft. Over the course of one hour, my test accuracy reached 58% over 34 epochs. My model started at 15% accuracy. If I was able to run our training for longer, we likely would have reached much higher accuracy.

## Running this project

download data and model here;
https://drive.google.com/drive/folders/1xIOfnyGIGrubHeIFmUI-J6y0SYMhmztf?usp=sharing

1. Ensure the jetson-inference library has been cloned to the Jetson Nano using the command Ensure jetson-inference is cloned to your Jetson Nano using the command `git clone --recursive https://github.com/dusty-nv/jetson-inference`. Ensure you have also installed the planes dataset
2.  Navigate to the jetson-inference/python/training/classification
3. set bash environment variables NET=/models/planes2 and DATASET=/data/planes
4. run the command imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/NAME_OF_CLASS_FOLDER/NAME_OF_JPEG DESIRED_NAME_OF_OUTPUT_JPG
5. the classified image will appear in the DESIRED_NAME_OF_OUTPUT_JPG provided in the previous step. It will also be output to the terminal as a result of the previous command.

Demonstration video (explaining process of activating the programme) posted below;
https://www.youtube.com/watch?v=Brm78advPiw
