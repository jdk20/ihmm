function [y,m,m_oracle] = genemission(s,beta_emission,gamma_emission,verbose)
%GENEMISSION Infinite HMM emission transition generative mechanism.
%   S = GENEMISSION(T,A,B,Y) returns a sequence of hidden states generated using the
%   scalar hyperparameters A, B, Y, which are >= 0. T is the number of iterations
%   and must be >= 1.
%
%   [S,N,NO] = GENEMISSION(T,A,B,Y) returns the count of transitions between 
%   hidden states, and the count of oracle-mediated transitions.
%
%   References:
%      [1] M. Beal, et. al., "The Infinite Hidden Markov Model", 2002


narginchk(3, 4);

if nargin == 3
    verbose = false;
else
    if ~islogical(verbose)
        error('Verbose statement must be true or false.');
    end
end

if ~isscalar(beta_emission) || beta_emission < 0 || ...
        ~isscalar(gamma_emission) || gamma_emission < 0
    error('Hyperparameters must be scalars and >=0.');
end

T = length(s);
K = max(s);

y = NaN(T, 1); % Output sequence of emissions
y(1) = 1;

Q = 1; % Initial number of unique emissions
m = zeros(K, Q); % Number of transitions between states (non-oracle)
m_oracle = 1; % Number of transitions from oracle to each state
for t = 1:T
    i = s(t);
    q = y(t);
    
    % (non-oracle) DP
    p = NaN(Q + 1, 1);
    for q = 1:Q
        p(q) = m(i,q)/(sum(m(i,:)) + beta_emission); % Existing
    end
    p(Q+1) = beta_emission/(sum(m(i,:)) + beta_emission); % Oracle
    
    % Random draw for non-oracle DP
    idx = randsample(1:length(p), 1, true, p);
    
    if idx == (Q + 1) % Oracle DP
        p = NaN(Q + 1, 1);
        for q = 1:Q
            p(q) = m_oracle(q)/(sum(m_oracle) + gamma_emission);
        end
        p(Q+1) = gamma_emission/(sum(m_oracle) + gamma_emission);

        % Random draw for oracle DP
        idx = randsample(1:length(p), 1, true, p);
        
        if idx == (Q + 1) % New emission
            % Expand
            temp_m = zeros(K, Q + 1);
            temp_m_oracle = zeros(1, Q + 1);
            temp_m(1:K, 1:Q) = m;
            temp_m_oracle(1:Q) = m_oracle;
            m = temp_m;
            m_oracle = temp_m_oracle;
            
            m(i, idx) = m(i, idx) + 1;
            m_oracle(idx) = m_oracle(idx) + 1;
            
            y(t+1) = Q + 1;
            Q = Q + 1;
            
            if verbose
                disp(['(t=',num2str(t+1),') Oracle new emission ',num2str(Q),' (prob: ', num2str(100*p(idx)),')']);
            end
        else
            y(t+1) = idx;
            m(i, idx) = m(i, idx) + 1;
            m_oracle(idx) = m_oracle(idx) + 1;       
            
            if verbose
                disp(['(t=',num2str(t+1),') Oracle state ',num2str(i),' emission ',num2str(idx),' (prob: ', num2str(100*p(idx)),')']);
            end
        end
 
    else % Self/existing transition
        y(t+1) = idx;
        m(i, idx) = m(i, idx) + 1;
        
        if verbose
            disp(['(t=',num2str(t+1),') State ',num2str(i),' emission ',num2str(idx),' (prob: ', num2str(100*p(idx)),')']);
        end
    end
end