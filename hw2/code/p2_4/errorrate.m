function rate = errorrate(w,test_data)

[n,d] = size(test_data.X);
y_test = ([ones(n,1) test_data.X]*w > 0);
rate = length(find(y_test ~= test_data.y))/n;