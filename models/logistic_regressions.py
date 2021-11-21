from sklearn.linear_model import LogisticRegression


from sklearn.metrics import roc_auc_score
import pandas as pd
from utility import load_customer_data
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# training sets
train_predict = []


# test sets
test_predict = []
train_data_set, test_data_set, train_label_set, test_label_set = load_customer_data()

clf_l1 = LogisticRegression(penalty="l1", C=0.5,solver='liblinear')
clf_l1.fit(train_data_set, train_label_set)
train_pred = clf_l1.predict(train_data_set)
train_pred_prob =clf_l1.predict_proba(train_data_set)[:,1]
train_accuracy = accuracy_score(train_label_set,train_pred)*100
train_roc_auc_score = roc_auc_score(train_label_set, train_pred_prob)*100


print('clf_l1 Confusion matrix:\n', confusion_matrix(train_label_set, train_pred))
print('clf_l1 Training AUC: %.4f %%' % train_roc_auc_score)
print('clf_l1 Training accuracy: %.4f %%' % train_accuracy)

test_pred =clf_l1.predict(test_data_set)
test_pred_prob = clf_l1.predict_proba(test_data_set)[:,1]
test_accuracy = accuracy_score(test_label_set,test_pred)*100
test_roc_auc_score = roc_auc_score(test_label_set,test_pred_prob)*100

print('clf_l1 Confusion matrix:\n', confusion_matrix(train_label_set, train_pred))
print('clf_l1 Testing AUC: %.4f %%' % test_roc_auc_score)
print('clf_l1 Testing accuracy: %.4f %%' % test_accuracy)
print(classification_report(test_label_set, test_pred, target_names=['churn','unchurn']))


clf_l2 = LogisticRegression(penalty="l2", C=0.5,solver='liblinear')
clf_l2.fit(train_data_set, train_label_set)
train_pred = clf_l2.predict(train_data_set)
train_pred_prob =clf_l2.predict_proba(train_data_set)[:,1]
train_accuracy = accuracy_score(train_label_set,train_pred)*100
train_roc_auc_score = roc_auc_score(train_label_set, train_pred_prob)*100


print('clf_l2 Confusion matrix:\n', confusion_matrix(train_label_set, train_pred))
print('clf_l2 Training AUC: %.4f %%' % train_roc_auc_score)
print('clf_l2 Training accuracy: %.4f %%' % train_accuracy)

test_pred =clf_l2.predict(test_data_set)
test_pred_prob = clf_l2.predict_proba(test_data_set)[:,1]
test_accuracy = accuracy_score(test_label_set,test_pred)*100
test_roc_auc_score = roc_auc_score(test_label_set,test_pred_prob)*100

print('clf_l2 Confusion matrix:\n', confusion_matrix(train_label_set, train_pred))
print('clf_l2 Testing AUC: %.4f %%' % test_roc_auc_score)
print('clf_l2 Testing accuracy: %.4f %%' % test_accuracy)

print(classification_report(test_label_set, test_pred, target_names=['churn','unchurn']))


# rf = pd.DataFrame(list(zip(test_pred,test_label_set)),columns =['predicted','actual'])
# rf['correct'] = rf.apply(lambda r:1 if r['predicted'] == r['actual'] else 0,axis=1)
# print(rf)
# print(rf['correct'].sum()/rf['correct'].count())
# cm_pt = mpl.colors.ListedColormap(["blue","red"])
# plt.scatter(train_data_set[:,11],train_data_set[:,21], c=train_label_set[:])
#
# plt.show()