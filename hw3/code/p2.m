load('svm.mat');
%% set1
model1_linear = svmtrain(set1_train.y, set1_train.X, '-t 0 -c 1000 -q');
[~, accu11, ~] = svmpredict(set1_test.y, set1_test.X, model1_linear);
model1_poly = svmtrain(set1_train.y, set1_train.X, '-t 1 -d 2 -r 1 -c 1000 -q');
[~, accu12, ~] = svmpredict(set1_test.y, set1_test.X, model1_poly);
model1_rbf = svmtrain(set1_train.y, set1_train.X, '-t 2 -g 0.5 -c 1000 -q');
[~, accu13, ~] = svmpredict(set1_test.y, set1_test.X, model1_rbf);
%% set2
model2_linear = svmtrain(set2_train.y, set2_train.X, '-t 0 -c 1000 -q');
[~, accu21, ~] = svmpredict(set2_test.y, set2_test.X, model2_linear);
model2_poly = svmtrain(set2_train.y, set2_train.X, '-t 1 -d 2 -r 1 -c 1000 -q');
[~, accu22, ~] = svmpredict(set2_test.y, set2_test.X, model2_poly);
model2_rbf = svmtrain(set2_train.y, set2_train.X, '-t 2 -g 0.5 -c 1000 -q');
[~, accu23, ~] = svmpredict(set2_test.y, set2_test.X, model2_rbf);
%% set3
model3_linear = svmtrain(set3_train.y, set3_train.X, '-t 0 -c 1000 -q');
[~, accu31, ~] = svmpredict(set3_test.y, set3_test.X, model3_linear);
model3_poly = svmtrain(set3_train.y, set3_train.X, '-t 1 -d 2 -r 1 -c 1000 -q');
[~, accu32, ~] = svmpredict(set3_test.y, set3_test.X, model3_poly);
model3_rbf = svmtrain(set3_train.y, set3_train.X, '-t 2 -g 0.5 -c 1000 -q');
[~, accu33, ~] = svmpredict(set3_test.y, set3_test.X, model3_rbf);
%% set4
model4_linear = svmtrain(set4_train.y, set4_train.X, '-t 0 -c 1000 -q');
[~, accu41, ~] = svmpredict(set4_test.y, set4_test.X, model4_linear);
model4_poly = svmtrain(set4_train.y, set4_train.X, '-t 1 -d 2 -r 1 -c 1000 -q');
[~, accu42, ~] = svmpredict(set4_test.y, set4_test.X, model4_poly);
model4_rbf = svmtrain(set4_train.y, set4_train.X, '-t 2 -g 0.222 -c 1000 -q');
[~, accu43, ~] = svmpredict(set4_test.y, set4_test.X, model4_rbf);
%% plot figure
pt_size = 5;
sv_size = 8;
figure;
pos1 = set1_train.X(set1_train.y == 1, :);
neg1 = set1_train.X(set1_train.y == -1, :);
plot(pos1(:,1), pos1(:,2), 'x', 'MarkerSize', pt_size); hold on;
plot(neg1(:,1), neg1(:,2), '^', 'MarkerSize', pt_size);
sv1 = full(model1_linear.SVs);
plot(sv1(:,1), sv1(:,2), 'og', 'MarkerSize', sv_size);
title('Set1');
legend('pos sample', 'neg sample', 'support vector');
figure;
pos2 = set2_train.X(set2_train.y == 1, :);
neg2 = set2_train.X(set2_train.y == -1, :);
plot(pos2(:,1), pos2(:,2), 'x', 'MarkerSize', pt_size); hold on;
plot(neg2(:,1), neg2(:,2), '^', 'MarkerSize', pt_size);
sv2 = full(model2_rbf.SVs);
plot(sv2(:,1), sv2(:,2), 'og', 'MarkerSize', sv_size);
title('Set2');
legend('pos sample', 'neg sample', 'support vector');
figure;
pos3 = set3_train.X(set3_train.y == 1, :);
neg3 = set3_train.X(set3_train.y == -1, :);
plot(pos3(:,1), pos3(:,2), 'x', 'MarkerSize', pt_size); hold on;
plot(neg3(:,1), neg3(:,2), '^', 'MarkerSize', pt_size);
sv3 = full(model3_rbf.SVs);
plot(sv3(:,1), sv3(:,2), 'og', 'MarkerSize', sv_size);
title('Set3');
legend('pos sample', 'neg sample', 'support vector');