## **Project Description**:

**Introduction:**
This project is designed to streamline email management by automatically classifying incoming messages as ‘Spam’ or ‘Not Spam’. By utilizing advanced machine learning techniques, the system learns from a vast dataset of pre-labelled emails, enabling it to make accurate predictions on new, unseen emails.

**Technical Overview:**
At the core of the project is the Multinomial Naive Bayes classifier, a probabilistic learning method commonly used in Natural Language Processing (NLP) for text classification due to its simplicity and effectiveness. The project is built using Python, leveraging libraries such as Pandas for data handling, Scikit-learn for machine learning tasks, and Streamlit for creating an interactive web application. The CountVectorizer transforms text data into a format that can be understood by the machine learning model.

## **Problem Statement:**
The inundation of spam emails poses a significant challenge for individuals and organizations alike, not only cluttering inboxes but also posing security risks. The project addresses the need for an automated system that can efficiently filter out spam, thereby saving time and reducing exposure to potential threats.


## **Methodology:**

**Data Preparation:**
The project begins with data preparation, where a dataset containing emails labelled as 'Spam' or 'Not Spam' (originally 'ham') is loaded using Pandas. The data is cleaned by removing duplicates to ensure the model learns from unique examples.

**Data Preprocessing:**
The 'Category' column in the dataset, which contains the labels, is renamed for clarity, changing 'ham' to 'Not Spam' and leaving 'Spam' as is. This step makes the labels more intuitive.

**Feature Extraction:**
The emails' text content, which is the model's input data, needs to be converted into a numerical format that the algorithm can process. This is done using CountVectorizer, which transforms the text into a sparse matrix of token counts, effectively encoding each email as a vector of numbers representing the frequency of each word.

**Model Training:**
With the features ready, the next step is to split the dataset into training and testing sets. The training set is used to train the Multinomial Naive Bayes classifier. This classifier is chosen because it works well with discrete data (like word counts) and assumes independence between features for simplicity.

**Model Evaluation:**
After training, the model's performance can be evaluated on the test set. However, this part of the code is commented out and not displayed.

**Prediction Function:**
A function named `predict` is defined as taking an email message as input and output in its predicted category. It uses CountVectorizer to transform the input message into numerical features and then applies the trained model to predict whether it's 'Spam' or 'Not Spam'.

**Model Persistence:**
To ensure that the model can be reused without retraining, it is persisted using Pickle. This serialization allows for easy storage and retrieval of the trained model.

##**Deployment with Streamlit:**
Finally, a Streamlit web application serves as the interface for users to interact with the model. Users can enter an email message into a text input field and click a button to classify it. The app then displays whether the message is predicted as 'Spam' or 'Not Spam'.

This implementation provides a complete end-to-end solution for email classification, from data preprocessing to user interaction through a web application.


##**Contributions:**
The project contributes to email security by providing a tool to automate the detection and classification of spam emails. It demonstrates how machine learning can be applied to solve real-world problems in cybersecurity.

##**Deployment Process:**
The trained model is deployed as a web application through Streamlit, which provides a user-friendly interface. Users can input an email message into the app, which then uses the model to predict its category and displays the result in real-time.

##**Impact and Practical Applications:**
This system's implications are far-reaching. It can be integrated into existing email services to enhance their spam detection capabilities or used by businesses to automatically categorize and prioritize incoming communications.

##**Conclusion:**
This email classification system showcases the potential of machine learning to improve digital communication channels. It provides a robust solution to one of the most pervasive issues in email management and stands as a testament to the practical benefits of AI in everyday technology.



## **System Demo:**

![The System Demo](https://github.com/Mutiu123/email-spam-detection/blob/main/demo/demo1.png)
![The System Demo](https://github.com/Mutiu123/email-spam-detection/blob/main/demo/demo2.png)



## **To run the model**
1. **Clone the Repository**:
   - First, clone the repository containing the heart disease prediction system code to your local machine. You can do this using Git or by downloading the ZIP file from the repository.

2. **Install Dependencies**:
   - Open your terminal or command prompt and navigate to the project directory.
   - Install the necessary Python libraries mentioned in the `requirements.txt` file using the following command:
     ```
     pip install -r requirements.txt
     ```

3. **Run the Streamlit App**:
   - In the same terminal or command prompt, execute the following command to run the Streamlit app:
     ```
     streamlet run app.py
     ```
   - This will start the local development server, and you'll see a message indicating that the app is running.
   - Open your web browser and visit `http://localhost:8501` (or the URL provided in the terminal) to access the interactive web app.

4. **Predict Heart Disease**:
   - On the Streamlit app, enter the message of your choice.
   - After entering the message, click the “validate” button.
