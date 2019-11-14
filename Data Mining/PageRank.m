function PageRank_complete
% This function implements the PageRank algorithm.
dbstop if error; clear; clc; close all;
dDecayFactor = 0.85;
mAdjacencyMatrix = [
0 1 0 0 0; % node 1
0 0 1 1 0; % node 2
0 0 0 1 0; % node 3
1 0 0 0 1; % node 4
0 0 1 0 0; % node 5
];
nNumOfNodes = size(mAdjacencyMatrix,1);
mNodeTransProbMatrixP = zeros(5);
mPRMatrix = zeros([nNumOfNodes,1]);
I = eye(nNumOfNodes);
% Node degree matrix
vNodeDegree = sum(mAdjacencyMatrix,2);
mNodeDegreeMatrix = diag(vNodeDegree);
mNodeDegreeMatrixInv = pinv(mNodeDegreeMatrix);
% Compute the node transition probability matrix
for i = 1 : nNumOfNodes
    mNodeTransProbMatrixP(i,:) = mAdjacencyMatrix(i,:)/vNodeDegree(i);
end
mNodeTransProbMatrixP = transpose(mNodeTransProbMatrixP);
vTeleportationVector = ones(nNumOfNodes,1) / nNumOfNodes;
% Compute the PageRank values by using the matrix inverse
mPRMatrix = mldivide((I - dDecayFactor*mNodeTransProbMatrixP) , ((1 - dDecayFactor)*vTeleportationVector));
vPRVector1 = sum(mPRMatrix,2);
disp("Method 1: Matrix Inversion");
disp("PageRank values");
display(vPRVector1);
% Compute the PageRank values by using the iterative method
disp("Method 2: Power Iteration");
vPRVector2 = [];
vPRVector2_old = vTeleportationVector;
vPRVector2_new = zeros([nNumOfNodes,1]);
nIdxOfIteration = 0;
while true
    for i = 1:nNumOfNodes
        vPRVector2_new(i,1) = (dDecayFactor * mNodeTransProbMatrixP(i,:) * vPRVector2_old) + ((1 - dDecayFactor) * vTeleportationVector(i,1));
    end
    if norm(vPRVector2_new - vPRVector2_old) < 10^(-6)
        vPRVector2 = vPRVector2_new;
        disp("# of iterations");
        disp(nIdxOfIteration);
        break;
    end
    vPRVector2_old = vPRVector2_new;
    nIdxOfIteration = nIdxOfIteration + 1;
end
vPRVector2 = vPRVector2_new;
disp("PageRank values");
display(vPRVector2);
figure;
