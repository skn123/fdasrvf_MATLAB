function MyMex(filename)
addpath(genpath(pwd));
if(nargin == 0)
	filename = 'ElasticCurvesReparam';
end
mex('-I./', ...
'-I./BinaryFiles/', ...
'-I./Manifolds/', ...
'-I./Manifolds/Euclidean/', ...
'-I./Manifolds/L2Sphere/', ...
'-I./Manifolds/LowRank/', ...
'-I./Manifolds/Oblique/', ...
'-I./Manifolds/OrthGroup/', ...
'-I./Manifolds/Sphere/', ...
'-I./Manifolds/Stiefel/', ...
'-I./Matlab/', ...
'-I./Matlab/ForCpp/', ...
'-I./Matlab/ForMatlab/', ...
'-I./Others/', ...
'-I./Problems/', ...
'-I./Problems/ElasticCurvesRO/', ...
'-I./Problems/EucFrechetMean/', ...
'-I./Problems/EucQuadratic/', ...
'-I./Problems/ObliqueTestSparsePCA/', ...
'-I./Problems/StieBrockett/', ...
'-I./Problems/StieSoftICA/', ...
'-I./Problems/StieSumBrockett/', ...
'-I./Problems/WeightedLowrank/', ...
'-I./Solvers/', ...
'-I./test/', ...
['./test/' filename '.cpp'], ...
'./Manifolds/Element.cpp', './Manifolds/LinearOPE.cpp', './Manifolds/Manifold.cpp', './Manifolds/ProductElement.cpp', './Manifolds/ProductManifold.cpp', './Manifolds/SharedSpace.cpp', './Manifolds/SmartSpace.cpp', ...
'./Manifolds/Euclidean/EucVariable.cpp', './Manifolds/Euclidean/EucVector.cpp', './Manifolds/Euclidean/Euclidean.cpp', ...
'./Manifolds/L2Sphere/L2Sphere.cpp', './Manifolds/L2Sphere/L2SphereVariable.cpp', './Manifolds/L2Sphere/L2SphereVector.cpp', ...
'./Manifolds/LowRank/LowRank.cpp', './Manifolds/LowRank/LowRankVariable.cpp', './Manifolds/LowRank/LowRankVector.cpp', ...
'./Manifolds/Oblique/Oblique.cpp', './Manifolds/Oblique/ObliqueVariable.cpp', './Manifolds/Oblique/ObliqueVector.cpp', ...
'./Manifolds/OrthGroup/OrthGroup.cpp', './Manifolds/OrthGroup/OrthGroupVariable.cpp', './Manifolds/OrthGroup/OrthGroupVector.cpp', ...
'./Manifolds/Sphere/Sphere.cpp', './Manifolds/Sphere/SphereVariable.cpp', './Manifolds/Sphere/SphereVector.cpp', ...
'./Manifolds/Stiefel/StieVariable.cpp', './Manifolds/Stiefel/StieVector.cpp', './Manifolds/Stiefel/Stiefel.cpp', ...
'./Others/ForDebug.cpp', './Others/Spline.cpp', './Others/Timer.cpp', './Others/randgen.cpp', ...
'./Problems/Problem.cpp', './Problems/mexProblem.cpp', ...
'./Problems/ElasticCurvesRO/DriverElasticCurvesRO.cpp', './Problems/ElasticCurvesRO/ElasticCurvesRO.cpp', ...
'./Problems/EucFrechetMean/EucFrechetMean.cpp', ...
'./Problems/EucQuadratic/EucQuadratic.cpp', ...
'./Problems/ObliqueTestSparsePCA/ObliqueTestSparsePCA.cpp', ...
'./Problems/StieBrockett/StieBrockett.cpp', ...
'./Problems/StieSoftICA/StieSoftICA.cpp', ...
'./Problems/StieSumBrockett/StieSumBrockett.cpp', ...
'./Problems/WeightedLowrank/WeightedLowRank.cpp', ...
'./Solvers/LRBFGS.cpp', './Solvers/LRTRSR1.cpp', './Solvers/RBFGS.cpp', './Solvers/RBroydenFamily.cpp', './Solvers/RCG.cpp', './Solvers/RNewton.cpp', './Solvers/RSD.cpp', './Solvers/RTRNewton.cpp', './Solvers/RTRSD.cpp', './Solvers/RTRSR1.cpp', './Solvers/RWRBFGS.cpp', './Solvers/Solvers.cpp', './Solvers/SolversLS.cpp', './Solvers/SolversTR.cpp', ...
'-lmwblas', '-lmwlapack', '-output', ['./BinaryFiles/' filename ]);
end