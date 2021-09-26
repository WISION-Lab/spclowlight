## Simulating Noisy dataset from SPAD cameras
#### Simulating raw images for SPC under low-light conditions

* Download dataset from official sources and extract it to use as clean image.
* Use create\_data.py script to simulate low-light images
 ##### Usage
 * Edit location of the original clean images in the script and the number of frames to average for image simulation
 * Run the script using following command

 ```
 $ python create_data.py
 ```
* Create txt files containing GT labels training and testing. Format of the files should be as follows
```
<file_location> <class_id> <image_id>
images/001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.png 0 0
images/001.Black_footed_Albatross/Black_Footed_Albatross_0074_59.png 0 1
images/001.Black_footed_Albatross/Black_Footed_Albatross_0014_89.png 0 2
images/001.Black_footed_Albatross/Black_Footed_Albatross_0031_100.png 0 3
................
```
where image_id is same for all noisy images with same scene content.
