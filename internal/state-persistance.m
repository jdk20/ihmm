clc; clear variables; close all;

rng(42)

n = 100; % number of samples

% Mixture to simulate from
y = 0.75.*betarnd(1,5,n,1) + 0.25.*betarnd(20,2,n,1);

linspace(0,1,10+2)
