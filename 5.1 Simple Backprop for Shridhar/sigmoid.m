% Sigmoid function

function y = sigmoid(x)

y = 1./(1 + exp(-x));
%y = -(x<-2).*1/2+1/2+(abs(x)<=2).*x/4+(x>2).*1/2;
