global network_data

num_input = 2;
num_hidden_layers = 2;
total_layers = num_hidden_layers + 2;
num_hidden = [2 2];
num_output = 1;
weights = cell(total_layers,1);
biases = cell(total_layers,1);
weights{1} = []; biases{1} = [];
weights{2} = ones(num_hidden(1),num_input);
biases{2} = zeros(num_hidden(1),1);
for layer = 2:num_hidden_layers
  weights{layer+1} = ones(num_hidden(layer),num_hidden(layer-1));
  biases{layer+1} = zeros(num_hidden(layer),1);
endfor
weights{total_layers} = ones(num_output,num_hidden(num_hidden_layers));
biases{total_layers} = zeros(num_output,1);
training_data = load('training_data');
theta = training_data(:,1:2);
x = training_data(:,3);
num_samp = length(theta);
eta = 30.0;
factor = eta/num_samp;
eps = 10^-7;
z = cell(total_layers,1);
activations = cell(total_layers,1);
deltas = cell(total_layers,1);
dCost_dbias = cell(total_layers,1);
dCost_dweight = cell(total_layers,1);

count = 0
for i=1:2
  
  activations{1} = theta';
  for layer = 2:total_layers
    z{layer} = weights{layer} * activations{layer-1} + biases{layer};
    activations{layer} = sigmoid(z{layer});
  endfor
  E = (x' - activations{total_layers});
  deltas{total_layers} = -E .* sigmoid_prime(z{total_layers});
  for layer = num_hidden_layers+1:-1:2
    deltas{layer} = (weights{layer+1}' * deltas{layer+1}) .* sigmoid_prime(z{layer});
  endfor
  for layer = 2:total_layers
    dCost_dbias{layer} = sum(deltas{layer},2);
    dCost_dweight{layer} = deltas{layer} * activations{layer-1}';
  endfor
  for layer = 2:total_layers
    biases{layer} = biases{layer} - factor*dCost_dbias{layer}; 
    weights{layer} = weights{layer} - factor*dCost_dweight{layer};
  endfor
  E
  weights
endfor
%disp('z = ');
%theta
%z
%activations
E
%deltas
%disp('----')
%dCost_dbias
%disp('====')
%dCost_dweight
%biases
%weights
%weights