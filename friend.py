from surprise import KNNBasic
import pandas as pd
from surprise import Dataset, SVD
from surprise import Reader


movies = ['Star wars', 'Star wars', 'GOT' , 'GOT' ,'South park', 'South park','Harry potter', 'Harry potter']
rating = [1,5,1,1,1,5,3,2]
users = ['Kim','Tim', 'John','Jimmy', 'Julia', 'Kim','Jimmy', 'Kim' ]

rating_dict = {'users' : users,
                'item' : movies,
                'rating': rating}

def FriendRecommender(user):
    df = pd.DataFrame(rating_dict)
    reader = Reader(rating_scale = (1,5))
    data  = Dataset.load_from_df(df[['users','item','rating']],reader)
    trainset = data.build_full_trainset()
    sim_options = {
        'name':'cosine',
        'user_based':True
    }

    algo = KNNBasic(sim_options)
    algo.fit(trainset)

    uid = trainset.to_inner_uid(user)
    pred = algo.get_neighbors(uid,3)

    for i in pred:
        print(trainset.to_raw_uid(i))

FriendRecommender('Kim')