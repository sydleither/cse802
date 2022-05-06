# cse802
Pattern Recognition and Analysis Final Project

https://www.hindawi.com/journals/bmri/2014/781670/tab1/ description of features

ideas for report

    discuss runtime as part of results

    if somebody died before being readmitted
    
    large time span of samples
    
'''important_features = zip(X.columns, classifier.feature_importances_)
important_features = list(sorted(important_features, key=lambda x: x[1], reverse=True))
plt.figure()
plt.barh([x[0] for x in important_features][:25], [x[1] for x in important_features][:25])
plt.title(f'n = {len(important_features)}, f1: {f1_score(pred, y_val, average=None)}')
plt.gca().invert_yaxis()'''