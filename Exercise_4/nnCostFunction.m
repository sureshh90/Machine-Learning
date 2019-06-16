function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Implementatioin for Part 1: Feedforward the neural network and return the cost in the
% variable J.
A1 = [ones(m,1) X];

A2 = sigmoid(A1*Theta1');

A2 = [ones(m,1) A2];

hypothesis = sigmoid(A2*Theta2');


% To create a matrix for y with individual vectors instead of numbers like 5,6 etc.

I = eye(m);

yVec = (I(:, y))';

yVec = yVec(:,1:num_labels);

% Cost function

% Another implementation
% J = (-log(hypothesis)*yVec') - (log(1-(hypothesis))*(1-yVec)');
% J = trace(J)/m;

regTerm = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));

J_unRegulated = sum(sum( (-yVec.*log(hypothesis)) - ((1-yVec).*log(1-(hypothesis))) )) / m;

J = J_unRegulated + regTerm;

% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad.

for t = 1:m
	
	% Step 1: Perform a feedforward pass
	
	a1 =  X(t,:);
	a1 = [1 , a1];
	
	z2 = a1*Theta1';
	a2 = sigmoid(z2);
	a2 = [1 , a2];
	
	z3 = a2*Theta2';
	a3 = sigmoid(z3);
	
	% Step 2: For each output unit k in layer 3 (the output layer),
    % set delta values.
	
	delta3 = a3 .- yVec(t,:);
	
	% Step 3: For the hidden layer l = 2, set delta values.
	
	delta2 = delta3 * Theta2 .* sigmoidGradient([1 , z2]);
	
	% Step 4: Accumulate the gradient.
		
	Delta1  = Delta1 .+  (a1' * delta2(2:end))';
	Delta2  = Delta2 .+  (a2' * delta3)';
	
	
endfor


Theta1_grad = Delta1./m;
Theta2_grad = Delta2./m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+ (Theta1(:,2:end)*lambda/m);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (Theta2(:,2:end)*lambda/m);
	
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26	



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
