function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels_1 = size(Theta2, 1);
num_labels_2 = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
% pp = zeros(size(X, 1), size(Theta2,1));
% ppp = zeros(size(X, 1), size(Theta1,1));
% Add ones to the X data matrix
X = [ones(m, 1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

for i = 1:size(Theta1,1)
    Z = sum((Theta1(i,:).*X)')';
    h0 = sigmoid(Z);
    ppp(:,i) = h0;
end
ppp = [ones(m, 1), ppp];
for i = 1:size(Theta2,1)
    Z = sum((Theta2(i,:).*ppp)')';
    h0 = sigmoid(Z);
    pp(:,i) = h0;
end
[M, p] = max(pp, [], 2);







% =========================================================================


end
