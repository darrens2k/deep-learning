# Assignment 2 Outline

Assignment 2 involves teaching an object detector to count different types of candy. Students need to label images of candy using Label Studio, annotate them with bounding boxes, and export the annotations in COCO JSON format. The provided images need to be labeled with eight candy types: Moon, Insect, Black_star, Grey_star, Unicorn_whole, Unicorn_head, Owl, and Cat. After labeling, students need to load the data using Huggingface's DatasetDict, converting COCO formatted annotations to a format readable by datasets.load_dataset(). They also need to implement a method called candy_counter(image) in a notebook (IPYNB) that loads the fine-tuned object detection model and returns a dictionary with counts of different types of candies in the input image. The notebook should include preparation and loading of data, fine-tuning the model, evaluation, and saving it, with all cells correctly executed, including a test image with predicted boundary boxes. Grading criteria include bug-free code, adherence to deliverables, and achieving a mAP score above 0.5 on unseen test data.