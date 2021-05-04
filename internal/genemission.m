function [y,m,mo] = genemission(s,b_e,y_e,verbose)
%GENEMISSION Infinite HMM emission generative mechanism.
%   Y = GENEMISSION(S,B_E,Y_E) returns emissions generated using the the hidden
%   states in S. Generation uses the scalar hyperparameters B_E and Y_E, which 
%   must be >= 0.
%
%   [Y,M,MO] = GENEMISSION(S,B_E,Y_E) returns the count of emissions from 
%   hidden states, and the count of oracle-mediated emissions.
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

if ~isscalar(b_e) || b_e < 0 || ...
        ~isscalar(y_e) || y_e < 0
    error('Hyperparameters must be scalars and >=0.');
end

T = length(s);
K = max(s);
y = NaN(T, 1); % Output sequence of emissions

Q = 0; % Initial number of unique emissions
m = zeros(K, Q); % Number of transitions between states (non-oracle)
mo = zeros(1, Q); % Number of transitions from oracle to each state
for t = 1:T
    i = s(t);
    
    % (non-oracle) DP
    p = NaN(Q + 1, 1);
    for q = 1:Q
        p(q) = m(i,q)/(sum(m(i,:)) + b_e); % Existing
    end
    p(Q+1) = b_e/(sum(m(i,:)) + b_e); % Oracle
    
    % Random draw for non-oracle DP
    idx = randsample(1:length(p), 1, true, p);
    
    if idx == (Q + 1) % Oracle DP
        p = NaN(Q + 1, 1);
        for q = 1:Q
            p(q) = mo(q)/(sum(mo) + y_e);
        end
        p(Q+1) = y_e/(sum(mo) + y_e);

        % Random draw for oracle DP
        idx = randsample(1:length(p), 1, true, p);
        
        if idx == (Q + 1) % New emission
            % Expand
            temp_m = zeros(K, Q + 1);
            temp_m_oracle = zeros(1, Q + 1);
            temp_m(1:K, 1:Q) = m;
            temp_m_oracle(1:Q) = mo;
            m = temp_m;
            mo = temp_m_oracle;
            
            m(i, idx) = m(i, idx) + 1;
            mo(idx) = mo(idx) + 1;
            
            y(t) = Q + 1;
            Q = Q + 1;
            
            if verbose
                disp(['(t=',num2str(t),') Oracle new emission ',num2str(Q), ...
                    ' (prob: ', num2str(100*p(idx)),')']);
            end
        else
            y(t) = idx;
            m(i, idx) = m(i, idx) + 1;
            mo(idx) = mo(idx) + 1;       
            
            if verbose
                disp(['(t=',num2str(t),') Oracle state ',num2str(i), ...
                    ' emission ',num2str(idx), ...
                    ' (prob: ', num2str(100*p(idx)),')']);
            end
        end
 
    else % Self/existing transition
        y(t) = idx;
        m(i, idx) = m(i, idx) + 1;
        
        if verbose
            disp(['(t=',num2str(t),') State ',num2str(i),' emission ', ...
                num2str(idx),' (prob: ', num2str(100*p(idx)),')']);
        end
    end
end