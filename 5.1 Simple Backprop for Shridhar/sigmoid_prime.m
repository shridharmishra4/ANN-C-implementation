% Derivative of sigmoid function

function z = sigmoid_prime(x)

y = sigmoid(x);
z = y.*(1-y);
%z = +1/4*(abs(x)<=2);
