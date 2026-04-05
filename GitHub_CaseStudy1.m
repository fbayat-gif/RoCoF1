%% This code generates the simulations for *** Case Study 1: Generator Trip Disturbance *** of the following paper:
%% Adaptive Regularized Numerical Differentiation of Noisy Signals with Application to RoCoF Estimation in Power Systems
%% Written By Dr. Farshad Merrikh-Bayat
clear all

lam = logspace(-6,0,20);    % lambda_min = 10^-6, lambda_max = 10^0, 20 lambda samples
                            % The search for lambda_opt will be done among the values in "lam".  

T = 0.01;                   % Sampling period

tspan = [0:T:20]';          % tspan contains the entire time samples 
                            % (the vector containing the time samples of the sliding
                            % window will be denoted as "t" in the following)

for j=1:length(tspan)       % Defining the f(t) (noise-free frequency) used in Case Study 1: Generator Trip Disturbance
    if tspan(j)<5
        f(j,1) = 50;
        derivExact(j,1) = 0;% The vector containing the samples of true (exact) RoCoF
    elseif tspan(j)>=5 && tspan(j)<6
        f(j,1) = 50-0.12*(1-exp(-3*(tspan(j)-5))); 
        derivExact(j,1) = -0.36*exp(-3*(tspan(j)-5));
    elseif tspan(j)>=6
        f(j,1) = 50-0.12*(1-exp(-3*(tspan(j)-5)))+0.09*(1-exp(-0.4*(tspan(j)-6)))+0.02*exp(-0.25*(tspan(j)-6))*sin(2*pi*0.8*(tspan(j)-6));
        derivExact(j,1) = -0.36*exp(-3*(tspan(j)-5))+0.036*exp(-0.4*(tspan(j)-6))-0.005*exp(-0.25*(tspan(j)-6))*sin(2*pi*0.8*(tspan(j)-6))...
            +0.032*pi*exp(-0.25*(tspan(j)-6))*cos(2*pi*0.8*(tspan(j)-6));
    end
end

eta = 2e-3*randn(length(tspan),1);  % The measurement noise that will be added to f(t)
fmeas = f + eta;                    % Full-length noisy measured frequency

n = 200;                            % Sliding window length (n=200 is used in the simulations of paper)         

%% trapezoidal integration operator with SBP-compatible boundary correction
% A = tril(ones(n+1));
% A(:,1) = 0.5;
% A = A-0.5*diag(ones(n+1,1));
% A(1,1) = -0.5;
% A(1,2) = 0.5;
% A = T*A;

%% trapeziodal integration operator (default)
A = tril(ones(n+1));
A = A-0.5*diag(ones(n+1,1));
A(:,1) = 0.5;
A = T*A;

%% Backward-difference differentiation operator
D = (1/T) * spdiags([-ones(n+1,1), ones(n+1,1)], [0 1], n, n+1);
D(n+1,:) = zeros(1,n+1);

derivEst = zeros(length(tspan),1);  % The vector containing the samples of estimated RoCoF 

for j=1:length(tspan)-n
    t = tspan(j:j+n);               % The time samples of the sliding window
    z_hat = fmeas(j:j+n);           % z_hat contains the samples of the noisy measured frequency in the sliding window

    for k=1:length(lam)
            
        lambda = lam(k);            % lambda is the selected regularization parameter
             
        P = pinv(lambda*D'*D+A'*A);
        y = D*P*A.'*(z_hat-z_hat(1));
                
        q(k) = -2 * lambda * (y' * D * P * D' * y) / (y'*y);

        vopt = pinv(lambda * D' * D + A' * A) * A' * (z_hat-z_hat(1));
    end
    
    ind = find(q==max(q),1);
    optLambda(j) = lam(ind);
    
    vv = pinv(optLambda(j) * D' * D + A' * A) * A' * (z_hat-z_hat(1));
    derivEst(j+n) = vv(end);    % The last entry of "vv" is equal to the derivative at the current moment of time
    
    fprintf('Iter = %d, Optimum Lambda = %e, Exact RoCoF = %e, Est. RoCoF = %e \n',j,optLambda(j), derivExact(j+n,1), derivEst(j+n))
                 
end 

figure(1)
subplot(311)
plot(tspan,fmeas,'b'); hold on
title('(a)')
ylabel('frequency (Hz)')

subplot(312)
plot(tspan,derivEst,'r','LineWidth',1); hold on
plot(tspan,derivExact,'b')
legend('estimated RoCoF','true RoCoF','Location','southeast')
ylabel('RoCoF (Hz/s)')
title('(b)')

subplot(313)
plot([n:n+length(optLambda)-1]*T,log10(optLambda),'b','LineWidth',1)
ylabel('log \lambda_{opt}')
xlim([0 tspan(end)])
xlabel('t (s)')
grid minor
title('(c)')



