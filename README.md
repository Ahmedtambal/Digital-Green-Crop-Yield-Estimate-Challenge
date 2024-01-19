Hi everyone!


We started by building a baseline using Catboost. It was not that good in public lb. We can’t decide whether it will be good in private lb or not unless we try it against a robust validation.

· Data splitting:

Given the high rmse values in the public lb, we can say that indeed there are outliers in test set. So, we must account for that. Also, we can see that we are experimenting on the same 4 districts as on the training data. So, it may be good to include these in our validation account too. So we divide the target into 10 parts based on the quantiles (to ensure outliers/non-outliers are included in the cv in each fold), then we divide these 10 part into 40 part (each 10 parts represent a district). Then we built our Stratified KFold based on these groups. We found that this splitting had a perfect correlation with the private lb.

· Feature engineering:

- Count Vectorizer on the categorical features (put each word on a separate binary feature).

- Target encoding using aggregations based on Districts, Block and a modified Acre feature (it has some unique values with only one occurrence. Using target encoding on this would cause leakage. So, I replaced the value of each one of these single-occurred samples with the nearest value that had a greater number of samples. E.g., 1.553 with 1 sample, and 1.55 with 3 samples ---> 1.55 with 4 samples).

- Date and some interaction features.

· Modeling, losses, and metrics:

XGBoost was obviously the best for us. We tried a lot of other models including tree-base models, NN, TabNet,… and nothing gave good performance.

Anyway, I want to say some things about the outliers and distributions. Actually, if anyone looked carefully at how data splitted and at the scores he gets, he can expect such a shakeup. Here is my thought process regarding how to deal with private data:

1- If someone looked carefully into the ratios of each unique value against the others in each feature and compared that between train and test he can see clearly that ratios are SIMILAR. So, indeed a Stratified KFold was used to split the data.

2- So that means we can expect target ratios in the test set similar to those in train. But by looking into the number of samples we got in both train and test we have 5000 samples. And about 4000 are already in train with about 2-3 significant outliers. If that's true, then test set should have only 1-2 outliers. Given that public lb is 20% and we already get these really high RMSE in public lb, then obviously outliers are in public and not private.

3- So, I expected a distribution shift in private where we get a target with no outliers (no long tail).

4- To deal with this, we chose our submissions to be one with long tail (our best in public) and one with no long tail and no outliers (our best in private). So regardless of if we had outliers in private or not, we are safe.

5- To do this in code, firstly we monitored the model using another metric in addition to RMSE. It is Median Absolute Error. It doesn’t account for outliers at all, and correlates quite well with private lb.

6- As for the model, to make our distribution with no long tail I had to change the RMSE loss to something that penalizes outliers really heavily. I used a modified Huber loss. In huber loss, there is a parameter delta. In case of MSE, delta=2. As for my case, I used delta=9.

7- This made it quite sensitive to outliers and it produced the desired distribution and boosted the cv from RMSE 527 to 470! I can say for sure that this was our key idea to top 10.

8- I didn’t tune the model parameters well and neither did feature selection, and I am really regretting that now, because doing that would’ve made us go below 100 RMSE quite easily in private and secure 1st place. But yeah, not bad results anyway.
