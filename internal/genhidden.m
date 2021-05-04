function [s,n,no] = genhidden(T,a,b,y,verbose)
%GENHIDDEN Infinite HMM hidden state transition generative mechanism.
%   S = GENHIDDEN(T,A,B,Y) returns a sequence of hidden states generated using the
%   scalar hyperparameters A, B, Y, which are >= 0. T is the number of iterations
%   and must be >= 1.
%
%   [S,N,NO] = GENHIDDEN(T,A,B,Y) returns the count of transitions between 
%   hidden states, and the count of oracle-mediated transitions.
%
%   References:
%      [1] M. Beal, et. al., "The Infinite Hidden Markov Model", 2002


narginchk(4, 5);

if nargin == 4
    verbose = false;
else
    if ~islogical(verbose)
        error('Verbose statement must be true or false.');
    end
end

if ~isscalar(a) || a < 0 || ~isscalar(b) || b < 0 || ~isscalar(y) || y < 0
    error('Hyperparameters must be scalars and >=0.');
end

if ~isscalar(T) || T < 1 || floor(T) ~= T
    error('Number of iterations must be >0.');
end

s = NaN(T, 1); % Output sequence of states
s(1) = 1;
K = 1; % Initial number of states
n = zeros(K,K); % Number of transitions between states (non-oracle)
no = zeros(1,K); % Number of transitions from oracle to each state
no(1) = 1;

for t = 1:T-1
    i = s(t);
    
    % (non-oracle) DP
    p = NaN(K + 1, 1);
    for j = 1:K
        if s(t) == j
            p(j) = (n(i,j) + a)/(sum(n(i,:)) + b + a); % Self
        else
            p(j) = n(i,j)/(sum(n(i,:)) + b + a); % Existing
        end
    end
    p(K+1) = b/(sum(n(i,:)) + b + a); % Oracle
    
    % Random draw for non-oracle DP
    idx = randsample(1:length(p), 1, true, p);
    
    if idx == (K + 1) % Oracle DP
        p = NaN(K + 1, 1);
        for j = 1:K
            p(j) = no(j)/(sum(no) + y);
        end
        p(K+1) = y/(sum(no) + y);

        % Random draw for oracle DP
        idx = randsample(1:length(p), 1, true, p);
        
        if idx == (K + 1) % New state
            % Expand count matrix and vector
            temp_n = zeros(K + 1, K + 1);
            temp_n_oracle = zeros(1, K + 1);
            temp_n(1:K, 1:K) = n;
            temp_n_oracle(1:K) = no;
            n = temp_n;
            no = temp_n_oracle;
            
            n(i, idx) = n(i, idx) + 1;
            no(idx) = no(idx) + 1;
            
            s(t+1) = K + 1;
            K = K + 1;
            
            if verbose
                disp(['(t=',num2str(t+1),') Oracle new state ',num2str(K), ...
                    ' (prob: ', num2str(100*p(idx)),')']);
            end
        else
            s(t+1) = idx;
            n(i, idx) = n(i, idx) + 1;
            no(idx) = no(idx) + 1;       
            
            if verbose
                if i == idx % Self-transition
                    disp(['(t=',num2str(t+1),') Oracle self-transition state ', ...
                        num2str(idx),' (prob: ', num2str(100*p(idx)),')']);
                else % Transition
                    disp(['(t=',num2str(t+1),') Oracle transition state ', ...
                        num2str(i),' to state ',num2str(idx), ...
                        ' (prob: ', num2str(100*p(idx)),')']);
                end
            end
        end
 
    else % Self/existing transition
        s(t+1) = idx;
        n(i, idx) = n(i, idx) + 1;
        
        if verbose
            if i == idx % Self-transition
                disp(['(t=',num2str(t+1),') Self-transition state ', ...
                    num2str(idx),' (prob: ', num2str(100*p(idx)),')']);
            else % Transition
                disp(['(t=',num2str(t+1),') Transition state ',num2str(i), ...
                    ' to state ',num2str(idx),' (prob: ', num2str(100*p(idx)),')']);
            end
        end
    end
end