from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score
from utility import load_customer_data
from sklearn import tree
import graphviz


train_data_set, test_data_set, train_label_set, test_label_set = load_customer_data()

clf = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)
clf.fit(train_data_set,train_label_set)
train_pred = clf.predict(train_data_set)
train_pred_prob =clf.predict_proba(train_data_set)[:,1]
train_accuracy = accuracy_score(train_label_set,train_pred)*100
train_roc_auc_score = roc_auc_score(train_label_set, train_pred_prob)*100


test_predict = clf.predict(test_data_set)


print(accuracy_score(test_label_set,test_predict))

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

feature_names =['age','gender','phone','reg_time','total_charge','total_actived_card','initial_price','balance','diff_day',
                'frequency','money','total_actived_money','number_of_card','lst_yr_total_recharge_money','lst_yr_card_recharge_count',
                'serving_total_count','history_serving_people_count','number_of_consumption_retail_category','number_of_consumption_service_category',
                'total_recharge_money','total_recharge_count','serving_people_left'

                ]
target =['churn','unchurn']
with open('customer_churn.dot','w')as f:
    dot_data = tree.export_graphviz(clf,
                                    out_file=f,
                          feature_names=feature_names,
                          class_names=target,
                          filled=True, rounded=True,
                          special_characters=True)
graph = graphviz.Source(dot_data)