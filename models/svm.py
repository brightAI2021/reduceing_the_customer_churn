from sklearn import svm
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score
from utility import load_customer_data

train_data_set, test_data_set, train_label_set, test_label_set = load_customer_data()
clf = svm.SVC(C=2, kernel='rbf', gamma='auto', decision_function_shape='ovr',probability=True)

clf.fit(train_data_set, train_label_set)
train_pred = clf.predict(train_data_set)
train_pred_prob =clf.predict_proba(train_data_set)[:,1]
train_accuracy = accuracy_score(train_label_set,train_pred)*100
train_roc_auc_score = roc_auc_score(train_label_set, train_pred_prob)*100


print('Confusion matrix:\n', confusion_matrix(train_label_set, train_pred))
print('Training AUC: %.4f %%' % train_roc_auc_score)
print('Training accuracy: %.4f %%' % train_accuracy)

test_pred =clf.predict(test_data_set)
test_pred_prob = clf.predict_proba(test_data_set)[:,1]
test_accuracy = accuracy_score(test_label_set,test_pred)*100
test_roc_auc_score = roc_auc_score(test_label_set,test_pred_prob)*100

print('Confusion matrix:\n', confusion_matrix(train_label_set, train_pred))
print('Testing AUC: %.4f %%' % test_roc_auc_score)
print('Testing accuracy: %.4f %%' % test_accuracy)
print(classification_report(test_label_set, test_pred, target_names=['churn','unchurn']))