# Twitter Bot Detection using Graph Neural Networks
Implementation of the BotRGCN architecture. 


# Dataset Statistics
The model was trained on the TwiBot-22 dataset.


| **Unit**               | **Human**   |  **Bot**  | **Total** |
|------------------------|-------------|-----------|-----------|
| Users (all)            | 860.057     | 139.943   | 1.000.000 |
| Users (min. 1 tweet)   | 818.613     | 115.259   | 933.872   |
| Tweet                  | 81.250.102  | 6.967.355 | 88.217.457|
| Following relation     | 1.038.302   | 78.353    | 1.116.655 |
| Follower relation      | 2.383.574   | 243.405   | 2.626.979 |
| Retweet relation       | 1.482.911   | 97.732    | 1.580.643 |
| Hashtags               | 56.353.776  | 9.646.857 | 66.000.633|

# User Relations
A user relation is an interaction between two users. When training Graph Neural Networks, relations are modeled as edges between nodes.
The following relations exist in the TwiBot-22 dataset:
- Follower relation (F): User A follows User B
- Following relation (F): User A is followed by User B

The following relations were derived from the TwiBot-22 dataset:
- Retweet relation (R): User A retweeted a tweet from User B
- Co-Retweet relation (Co-R): User A and User B retweet the same tweet
- Co-Hashtag relation (Co-H): User A and User B use the same hashatg



# Training Results
The model was trained using different combinations of user relations and achieves state-of-the-art performance.


|  **Configuration** |    **Accuracy**    |    **F1-Score**    |
|--------------------|--------------------|--------------------|
| F-F                | 77.0 &plusmn; 0.18 | 50.4 &plusmn; 0.74 |
| F-F, R             | 79.0 &plusmn; 0.88 | 57.4 &plusmn; 0.31 |
| F-F, Co-R          | 79.0 &plusmn; 0.17 | 57.1 &plusmn; 0.62 |
| F-F, R, Co-R       | 79.0 &plusmn; 0.20 | 57.7 &plusmn; 0.25 |
| F-F, R, Co-R, Co-H | 79.2 &plusmn; 0.10 | 57.9 &plusmn; 0.51 |
