# NLP A4
 AIT NLP Assignment 4

- [Student Information](#student-information)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Model evaluation and comparison](#model-evaluation-and-comparison)
- [Impact of hyperparameters](#impact-of-hyperparameters)
- [Challenges and improvement](#challenges-and-improvements)


## Student Information
Name - Min Marn Ko  
ID - st125437

## Installation and Setup
Webapp at localhost:8501

## Usage
cd into app folder and run 'streamlit run app.py'  

## Model evaluation and comparison
| Model       | Training Loss | Average Cosine Similarity | Accuracy |
|------------------|---------------|--------------|--------------|
|S-BERT (pretrained)|        1.003522         |       0.7661      | 34.8%         |
| S-BERT (scratch) |         0.781859          |      0.9982       | 33.84%        |

At first, it seemed like the model was performing well, but upon closer inspection, the results are misleading. This is because the model tends to predict <PAD> for all masked tokens. As a result, when fine-tuning S-BERT, it labels sentences as similar, even when they’re not, which makes the model unreliable for real-world use.

On the other hand, using the pretrained BERT base uncased model delivers a more reliable and expected result. Applying the training technique from the S-BERT paper produced a more consistent performance, with an accuracy of 34.8% and a loss of 1.00.

## Impact of hyperparameters
The hyperparameters chosen for training our BERT model was:  
Number of Encoder of Encoder Layer - 6  
Number of heads in Multi-Head Attention - 8  
Embedding Size/ Hidden Dim - 768  
Number of epochs - 10  
Training data - 1000000 sentences  
Vocab size - 73276  

The hyperparameters chosen for tuning S-BERT on our BERT model was:
Training data - 10000 rows  
Number of epochs - 1  

The subpar performance of our model seems to stem from two main factors: the limited size of the training data and the relatively shallow architecture with only 6 encoder layers. While the vocabulary size, exceeding 70,000 words, should theoretically be sufficient, these constraints likely hinder the model's ability to learn effectively.

## Challenges and improvements
Implementing the S-BERT model architecture and its objective function with our BERT model proved to be a challenging task, particularly when we had to create a custom tokenizer. The tokenizer was built using a custom class, SimpleTokenizer(). 

Matrix shape manipulations also presented difficulties, especially since implicit shape changes in Jupyter notebooks were no longer applicable in utils.py and app.py. Additionally, the limited training data contributed to the model's poor performance.

To improve performance, I recommend using a larger and more diverse training dataset. Additionally, experimenting with various hyperparameters such as embedding size, the number of encoder layers, and the number of epochs could be beneficial. Implementing a learning rate scheduler and early stopping may also help. For S-BERT fine-tuning, better training data combined with testing different objective functions from the paper could yield improved results.

![Analysis Image](analyze.png)
