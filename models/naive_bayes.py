from sklearn.naive_bayes import GaussianNB,BernoulliNB
from utility import load_customer_data
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import  roc_auc_score
train_data_set, test_data_set, train_label_set, test_label_set = load_customer_data()

gnb = GaussianNB()
bnb = BernoulliNB()
gnb.fit(train_data_set,train_label_set)


train_pred = gnb.predict(train_data_set)
train_pred_prob =gnb.predict_proba(train_data_set)[:,1]
train_accuracy = accuracy_score(train_label_set,train_pred)*100
train_roc_auc_score = roc_auc_score(train_label_set, train_pred_prob)*100


test_predict = gnb.predict(test_data_set)


print(accuracy_score(test_label_set,test_predict))

print('Confusion matrix:\n', confusion_matrix(train_label_set, train_pred))
print('GNB Training AUC: %.4f %%' % train_roc_auc_score)
print('GNBTraining accuracy: %.4f %%' % train_accuracy)

test_pred =gnb.predict(test_data_set)
test_pred_prob = gnb.predict_proba(test_data_set)[:,1]
test_accuracy = accuracy_score(test_label_set,test_pred)*100
test_roc_auc_score = roc_auc_score(test_label_set,test_pred_prob)*100

print('Confusion matrix:\n', confusion_matrix(train_label_set, train_pred))
print('GNB Testing AUC: %.4f %%' % test_roc_auc_score)
print('GNB Testing accuracy: %.4f %%' % test_accuracy)
print(classification_report(test_label_set, test_pred, target_names=['churn','unchurn']))


bnb.fit(train_data_set,train_label_set)
test_predict = bnb.predict(test_data_set)
print(accuracy_score(test_label_set,test_predict))

print('Confusion matrix:\n', confusion_matrix(train_label_set, train_pred))
print('BNB Training AUC: %.4f %%' % train_roc_auc_score)
print('BNB Training accuracy: %.4f %%' % train_accuracy)

test_pred =gnb.predict(test_data_set)
test_pred_prob = gnb.predict_proba(test_data_set)[:,1]
test_accuracy = accuracy_score(test_label_set,test_pred)*100
test_roc_auc_score = roc_auc_score(test_label_set,test_pred_prob)*100

print('Confusion matrix:\n', confusion_matrix(train_label_set, train_pred))
print('BNB Testing AUC: %.4f %%' % test_roc_auc_score)
print('BNB Testing accuracy: %.4f %%' % test_accuracy)
print(classification_report(test_label_set, test_pred, target_names=['churn','unchurn']))
