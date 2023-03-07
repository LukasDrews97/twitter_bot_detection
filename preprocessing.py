import os
import json
import math
from datetime import datetime as dt

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import pipeline

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = "preprocessed_subgraph_20230220/"
path_preprocessed = "preprocessed_subgraph_20230220/"
#user_tweets_path = "src/Data_test/users/"

def main():
    ### Load data
    # create folder if not existing
    os.makedirs(path_preprocessed, exist_ok=True)

    print("Preprocessing: Loading raw data")

    # load user.json
    user = read_json_normalized(f"{path}user.json")
    user['created_at'] = pd.to_datetime(user['created_at'], utc=True)
    uid_index={uid:index for index, uid in enumerate(user['id'].values)}

    

    # load edges
    edge = pd.read_csv(f"{path}edges.csv")

    # load split
    split=pd.read_csv(f"{path}split.csv")
    uid_split={uid:split for uid, split in zip(split['id'].values,split['split'].values)}

    # load labels
    label=pd.read_csv(f"{path}labels.csv")
    uid_label={uid:label for uid, label in zip(label['id'].values,label['label'].values)}



    ### Create bitmasks
    labels_new = []
    train_mask = []
    test_mask = []
    validation_mask = []

    print("Preprocessing: Creating bitmasks")
    for i, uid in enumerate(tqdm(uid_index.keys())):
        user_label = uid_label[uid]
        user_split = uid_split[uid]
        
        if user_label == "human":
            labels_new.append(0)
        else:
            labels_new.append(1)
            
        if user_split == "train":
            train_mask.append(i)
        elif user_split == "test":
            test_mask.append(i)
        else:
            validation_mask.append(i)

    assert (len(train_mask) + len(test_mask) + len(validation_mask)) == len(uid_index)
    print("Preprocessing: Train Labels: " + str(len(train_mask)))
    print("Preprocessing: Test Labels: " + str(len(test_mask)))
    print("Preprocessing: Validation Labels: " + str(len(validation_mask)))

    # save data to disk
    torch.save(torch.tensor(train_mask, dtype=torch.long), f"{path_preprocessed}train_mask.pt")
    torch.save(torch.tensor(test_mask, dtype=torch.long), f"{path_preprocessed}test_mask.pt")
    torch.save(torch.tensor(validation_mask, dtype=torch.long), f"{path_preprocessed}validation_mask.pt")
    torch.save(torch.tensor(labels_new, dtype=torch.long), f"{path_preprocessed}labels.pt")



    ### Create edge index and types
    print("Preprocessing: Create edge index and types")

    edge_index = []
    edge_type = []
    edge_relation_mapping = {'followers': 0, 'following': 1}

    for i in tqdm(range(len(edge))):
        source_id = edge['source_id'][i]
        target_id = edge['target_id'][i]
        relation = edge['relation'][i]
        if relation in edge_relation_mapping:
            try:
                edge_index.append([uid_index[source_id], uid_index[target_id]])
                edge_type.append(edge_relation_mapping[relation])
            except KeyError:
                continue

    assert len(edge_index) == len(edge_type)
    print("Preprocessing: Edge Index: " + str(len(edge_index)))            
    
    # save data to disk
    torch.save(torch.tensor(edge_index, dtype=torch.long).t(), f"{path_preprocessed}edge_index.pt")
    torch.save(torch.tensor(edge_type, dtype=torch.long), f"{path_preprocessed}edge_type.pt")



    ### Create numerical and categorical features
    print("Preprocessing: Creating numerical and categorical features")
    following_count = extract_numeric_user_property(user, 'public_metrics.following_count', True)
    followers_count = extract_numeric_user_property(user, 'public_metrics.followers_count', True)
    tweet_count = extract_numeric_user_property(user, 'public_metrics.tweet_count', True)
    #username_length = list(map(lambda s: len(s), extract_literal_user_property(user, 'username'))) # not in use
    name_length = list(map(lambda s: len(s), extract_literal_user_property(user, 'name')))

    #normalize
    #username_length = normalize_numerical_feature(username_length)
    name_length = normalize_numerical_feature(name_length)

    start_date = dt.strptime('15/03/22 00:00:00 +0000','%d/%m/%y %H:%M:%S %z') # last date of dataset
    active_days = []
    for create_date in user['created_at']:
        active_days.append((start_date - create_date).days)
    active_days = normalize_numerical_feature(active_days)

    # convert to tensors
    following_count = torch.tensor(following_count, dtype=torch.float32)
    followers_count = torch.tensor(followers_count, dtype=torch.float32)
    tweet_count = torch.tensor(tweet_count, dtype=torch.float32)
    # username_length = torch.tensor(username_length, dtype=torch.float32) # not in use
    name_length = torch.tensor(name_length, dtype=torch.float32)
    active_days = torch.tensor(active_days, dtype=torch.float32)

    num_properties_tensor = torch.cat([
        followers_count,
        active_days,
        name_length,
        following_count,
        tweet_count],
        dim=1)

    # check for NaN values
    pd.DataFrame(num_properties_tensor.detach().numpy()).isna().values.any()

    protected = extract_boolean_user_property(user, 'protected')
    verified = extract_boolean_user_property(user, 'verified')

    default_profile_image = []
    default_image_url = 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png'
    for e in user['profile_image_url']:
        if e is not None:
            if e == default_image_url or e == '':
                default_profile_image.append(1)
            else:
                default_profile_image.append(0)
        else:
            default_profile_image.append(1)

    # convert to tensors
    protected = torch.tensor(protected, dtype=torch.float16).reshape(-1, 1)
    verified = torch.tensor(verified, dtype=torch.float16).reshape(-1, 1)
    default_profile_image = torch.tensor(default_profile_image, dtype=torch.float16).reshape(-1, 1)

    categorical_properties_tensor = torch.cat([
        protected,
        verified,
        default_profile_image], 
        dim=1)

    # save data to disk
    torch.save(num_properties_tensor, f"{path_preprocessed}num_properties_tensor.pt")
    torch.save(categorical_properties_tensor, f"{path_preprocessed}categorical_properties_tensor.pt")



    ### Extract user tweets
    print("Preprocessing: Extract user tweets")
    id_tweet={i:[] for i in range(len(uid_index.keys()))}

    for username in tqdm(uid_index.keys()):
        tweet_path_specific = f"{user_tweets_path}{username}/tweet.json"
        try:
            u_id = uid_index[username]
            with open(tweet_path_specific, 'r') as tweet_file:
                tweets = json.load(tweet_file)
            for tweet in tweets:
                text = tweet['text']
                id_tweet[u_id].append(text)
        except:
            continue
            
    # save to disk
    with open(f"{path_preprocessed}id_tweet.json", 'w') as tweet_file:
        json.dump(id_tweet, tweet_file)
    
    

    ### Create word embeddings
    print("Preprocessing: Create word embeddings")
    user_descriptions = list(user['description'])
    each_user_tweets=json.load(open(f"{path_preprocessed}id_tweet.json",'r'))

    text_extraction_pipeline = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', device=device, padding=True, truncation=True, max_length=50, add_special_tokens=True)

    # user descriptions
    user_description_embedding = []
    print("Preprocessing: Create word embeddings for user descriptions")
    for desc in tqdm(user_descriptions):
        if not desc or len(desc) == 0:
            user_description_embedding.append(torch.zeros(768))
            continue
        feature = torch.tensor(text_extraction_pipeline(desc))
        mean_feature = torch.mean(feature, dim=[0,1])
        user_description_embedding.append(mean_feature)

    # save to disk
    torch.save(torch.stack(user_description_embedding, dim=0), f"{path_preprocessed}user_description_embedding_tensor.pt")

    max_tweets_per_user = 20
    tweets_list = []
    print("Preprocessing: Create word embeddings for tweets")
    for i in tqdm(range(len(each_user_tweets))):
        tweets = each_user_tweets[str(i)]
        number_of_tweets = min(max_tweets_per_user, len(tweets))
        if len(tweets) == 0:
            mean_feature = torch.zeros(768)
        else:
            tweet_embeddings = []
            for j, tweet in enumerate(tweets[0:number_of_tweets]):
                if not tweet or len(tweet) == 0:
                    tweet_embeddings.append(torch.zeros(768))
                    continue

                each_tweet_tensor=torch.tensor(text_extraction_pipeline(tweet))
                total_word_tensor = torch.mean(each_tweet_tensor, dim=[0,1])
                tweet_embeddings.append(total_word_tensor)
                
            mean_feature = torch.mean(torch.stack(tweet_embeddings), dim=0)
        tweets_list.append(mean_feature)

    # save to disk
    torch.save(torch.stack(tweets_list), f"{path_preprocessed}user_tweets_tensor.pt")

    # clear cuda cache
    torch.cuda.empty_cache()


def read_json_normalized(path):
    with open(path) as data_file:    
        data = json.load(data_file)
    return pd.json_normalize(data)

def normalize_numerical_feature(data):
    data = np.array(data)
    mean = data.mean()
    std = data.std()
    return ((data - mean) / std).reshape(-1, 1)

def extract_numeric_user_property(user, property_name, normalize = False):
    res = []
    for e in user[property_name]:
        if e is not None and e is not math.isnan(e):
            res.append(e)
        else:
            res.append(0)
    return normalize_numerical_feature(res) if normalize else res

def extract_literal_user_property(user, property_name):
    res = []
    for e in user[property_name]:
        if e is not None:
            res.append(e)
        else:
            res.append("")
    return res

def extract_boolean_user_property(user, property_name):
    res = []
    for e in user[property_name]:
        if e == True:
            res.append(1)
        else:
            res.append(0)
    return res


if __name__ == "__main__":
    main()
