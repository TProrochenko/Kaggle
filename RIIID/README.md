# Riiid! Answer Correctness Prediction

This is a brief description of my solution to https://www.kaggle.com/c/riiid-test-answer-prediction competition.

I am happy that I got into top 7% out of 3395 teams in such a challenging competition, learning a lot in the process. \
My solution is here with all dependencies is here: https://www.kaggle.com/tymurprorochenko/riiid-public-submit \
Final ROC AUC for local CV is 0.788, public LB is 0.790 and private LB is 0.792.

Two aspects made it particularly hard for me:
* Large set of training data (100m rows, multiple tables) that could not be simply processed on kaggles 16gb ram VM;
* Submission through api with time constraints and a need to update features online made code optimization very important.


**Preprocessing, with pyspark** https://www.kaggle.com/tymurprorochenko/riiid-parquets-v5
* Split dataset into 10 parts by users
* Created "task container id ordered by timestamp" to simulate sequences of questions in the test set.
* Same splits were further used for training and testing. 
Such strategy meant that local validation was worse than leaderboard due to higher share of new users who are harder to predict, however the difference was consistent. 
I decided that under sampling new users for better CV/LB correlation meant added complexity and less samples for training. 

**Online / Offline features**
* Only user-based features are recalculated online
* All content-based statistics was calculated on all of the data, this created small target leakage, however I decided that the time saved was more important.


**Memory and time management**
* Initialy I lost a lot of time since my initial solution was based on pyspark preprocessing with conversion to pandas. 
In the second iteration I rewrote everything based on cuda dataframes, only to find out that it was still not fast enough.
Only after I switched to numpy, my submission time decreased to about one hour. \
Thus, data is stored in numpy arrays for most of the features and in lists of booleans for sequences of last user answers. 
Below is a simplified example of my pipeline:

**Initialize arrays**
```python
max_users = 450000 #10% overhead for users that we will see only during submission
max_content = 1352
max_list_length = 100

#pre defining variables to prevent data type conversion on the go: 
int8_0 = np.int8(0)
int8_1 = np.int8(1)
int16_0 = np.int16(0)
int16_1 = np.int16(1)
int32_0 = np.int32(0)
int32_1 = np.int32(1)

#array for counts of correct and false answers:
user_data = np.zeros((max_users, 2), dtype = np.int16)

#array for last answer correctness, total answer attempts and last answer value:
user_content_data = np.zeros((max_users, max_content, 3), dtype = np.int8)

#list of lists for last N user answers:
user_sequence = [[] for _ in range(max_users)]
```

**Update user mapping**
```python
#user index table and counter:
user_map = np.zeros(np.iinfo(np.int32).max, dtype = np.int32)
next_user_place = np.int32(1)

#function for index table update, run every batch:
for u in unique_users:
  if user_map[u] == int32_0:
    user_map[u] = next_user_place
    next_user_place += int32_1
```


**Update feature arrays**
```python
#based on dataframe after prediction, when we already have answer values:
for r in range(len(df)):

  u = user_map[userid[r]]
  if content_type[r] == int8_0: 

    us = user_sequence[u]
    if len(us)>max_list_length: us.pop(0)

    c = ctntid[r]
    user_content_data[u,c,1] += int8_1 #count total answers
    user_content_data[u,c,2] = user_answer[r] #mark last answer value

    if answered_correctly[r] == int8_1:
      us.append(True) #mark answer correctness
      user_data[u,0] += int16_1 #count correct answers
      user_content_data[u,c,0] = int8_1 #mark last answer correctness
    else:
      us.append(False) #mark answer correctness
      user_data[u,1] += int16_1 #count false answers
      user_content_data[u,c,0] = int8_0 #mark last answer correctness
```


**Generate features**
```python
#based on dataframe with new questions for prediction:

u = user_map[userid] #convert user id's to feature table indices

#filter feature arrays and lists based on users in current dataframe:
ud = user_data[u,:]
ucd = user_content_data[u,ctntid] 
us = [user_sequence[_] for _ in u]

X = pd.DataFrame({
  'user_mean':(ud[:,0]/(ud[:,0]+ud[:,1]+int16_1)).astype(np.float32),
  'user_mean_last_n':np.array([_.count(True)/len(_) for _ in us], dtype = np.float32)
  'user_content_attempts':ucd[:,1],
  'user_content_last_attempt':ucd[:,0]
  })
```

#### My features
Apart from common mean encodings, some features turned out to be different from what most users used, some of them are:

* Calculated content correlation based on user answers and used this information to calculate user-based features based on N neighbors from predicted question.\
  https://www.kaggle.com/tymurprorochenko/content-correlation \
  https://www.kaggle.com/tymurprorochenko/content-correlation-100to300 
* Average content answer duration;
* Question mean based on first only user answers;
* Severity of user mistakes based on how dificult the question is to answer and how common the answer that user gave is;
* Mean correctnes for last N user answers and user-part answers;

#### Model
Since my initial pipeline was created for gradient boosting on decision trees, and I did not have the time to add my own SAKT/SAINT model. When choosing between LGBM, XGBoost and CatBoost I stopped on the later one since it natively supported training on Kaggle GPUs, allowing to fit 30x50M numerical dataset at once and training under 1 hour. \
In the end I used an average of two models trained on 50% of the data each. \
https://www.kaggle.com/tymurprorochenko/riiid-model
