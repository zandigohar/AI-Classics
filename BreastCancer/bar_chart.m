clear;
clc;
close all;
% x=categorical({'Nearest Neighbors', 'Linear SVM', 'Gaussian Process', ...
%     'Decision Tree', 'Random Forest', 'Neural Net', ...
%     'AdaBoost', 'Naive Bayes', 'QDA'});
x=categorical({'Nearest-n', 'SVM', 'GP', ...
    'DT', 'RF', 'Neural-n', ...
    'A-Boost', 'Naive-B', 'QDA'});

y=[93.859649,94.736842,96.491228,94.736842,95.614035,92.982456,96.491228,96.491228,97.368421];
[B,I] = sort(y,'descend');
x=cellstr(x(I));
y=y(I);
bar(y);
xlabel('Classifier');
ylabel('Accuracy [%]');
title('Accuracy of Different Classifiers');
axis([0 10 90 98]);
set(gca, 'XTickLabel',x, 'XTick',1:numel(x))
