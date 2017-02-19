load('data.mat');
w_f_11 = fisherdiscriminant(train1.X, train1.y);
w_l_11 = logisticreg(train1.X, train1.y);
w_f_12 = fisherdiscriminant(train1_2.X, train1_2.y);
w_l_12 = logisticreg(train1_2.X, train1_2.y);
% boundary([w_f_11 w_l_11], train1);
errorrate(w_f_11, test1)
errorrate(w_l_11, test1)
boundary([w_f_12 w_l_12], train1_2);
errorrate(w_f_12, test1)
errorrate(w_l_12, test1)
%%
w_f_2 = fisherdiscriminant(train2.X, train2.y);
w_l_2 = logisticreg(train2.X, train2.y);
boundary([w_f_2 w_l_2], train2);
errorrate(w_f_2, test2)
errorrate(w_l_2, test2)