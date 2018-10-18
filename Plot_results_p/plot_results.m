%% Loading and visualizing results from Learning Fourier modes code
% Here, looping through p values = 0.1:0.1:0.9 for connectivity of the
% Erdos-Renyi graph

clear all

for ind = 1:8 % index of run
    
    % Directory where your "Run" folders are saved
    projectdir = ['/Users/ciocanel.1/Dropbox/MRC 2018/6KuramotoOscillators/Python_code_Mark/learn_model_fourier/Plot_Results_p/Run' num2str(ind)];
    
    cd(projectdir);
    
    S = dir(fullfile('*.csv'));
    
    
    %% Adjacency matrix results
    M = csvread(S(1).name,1,1,[1 1 5 9]);
    error_rate = M(2,:);
    area_ROC   = M(3,:);
    best_f1    = M(4,:);
    thresh     = M(5,:);
    
    %% Plotting
    
    figure(1)
    hold on
    plot(0.1:0.1:0.9,error_rate,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('% Error rate');
    set(gca, 'FontSize',16)
    
    %
    figure(2)
    hold on
    plot(0.1:0.1:0.9,area_ROC,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('Area under ROC curve');
    set(gca, 'FontSize',16)
    
    %
    figure(3)
    hold on
    plot(0.1:0.1:0.9,best_f1,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('Best f1 score');
    set(gca, 'FontSize',16)
    
    %
    figure(4)
    hold on
    plot(0.1:0.1:0.9,thresh,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('Threshold for best f1 score');
    set(gca, 'FontSize',16)
    
    
    %% Coupling function results
    M = csvread(S(2).name,1,1,[1 1 3 9]);
    area_pred_true   = M(1,:);
    area_true_axis   = M(2,:);
    area_ratio       = M(3,:);
    
    %% Plotting
    
    figure(5)
    hold on
    plot(0.1:0.1:0.9,area_pred_true,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('Area between predicted and true curve');
    set(gca, 'FontSize',16)
    
    %
    figure(6)
    hold on
    plot(0.1:0.1:0.9,area_true_axis,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('Area between true function and axis');
    set(gca, 'FontSize',16)
    
    %
    figure(7)
    hold on
    plot(0.1:0.1:0.9,area_ratio,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('Area ratio');
    set(gca, 'FontSize',16)
    
    
    
    %% Frequency results
    M = csvread(S(3).name,1,1,[1 1 5 9]);
    max_abs_dev   = M(1,:);
    mean_abs_dev  = M(2,:);
    max_rel_dev   = M(3,:);
    mean_rel_dev  = M(4,:);
    corr          = M(5,:);
    
    %% Plotting
    
    figure(8)
    hold on
    plot(0.1:0.1:0.9,max_abs_dev,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('Mx absolute deviation');
    set(gca, 'FontSize',16)
    
    %
    figure(9)
    hold on
    plot(0.1:0.1:0.9,mean_abs_dev,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('Mean absolute deviation');
    set(gca, 'FontSize',16)
    
    %
    figure(10)
    hold on
    plot(0.1:0.1:0.9,max_rel_dev,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('Max relative deviation');
    set(gca, 'FontSize',16)
    
    %
    figure(11)
    hold on
    plot(0.1:0.1:0.9,mean_rel_dev,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('Mean relative deviation');
    set(gca, 'FontSize',16)
    
    %
    figure(12)
    hold on
    plot(0.1:0.1:0.9,corr,'*-','Linewidth',2);
    xlabel('p (Erdos-Renyi)')
    ylabel ('Correlation');
    set(gca, 'FontSize',16)
       
    
    cd ../
    
    
end






