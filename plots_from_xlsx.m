
% define loop_parameter and date/time (for correct filename)
runtype = 'num_repeats';
runtime = '20181214-193333';

% read in the 5 files generated
[Anum,Atxt,Araw] = xlsread(strcat('adjacency_matrix_results_',runtype,'_sweep_',runtime,'.xlsx'));
[Gnum,Gtxt,Graw] = xlsread(strcat('coupling_function_results_',runtype,'_sweep_',runtime,'.xlsx'));
[Omnum,Omtxt,Omraw] = xlsread(strcat('frequency_results_',runtype,'_sweep_',runtime,'.xlsx'));
[Pnum,Ptxt,Praw] = xlsread(strcat('parameter_information_',runtype,'_sweep_',runtime,'.xlsx'));
[Valnum,Valtxt,Valraw] = xlsread(strcat('validation_error_results_',runtype,'_sweep_',runtime,'.xlsx'));

% grab appropriate rows
param_vals = Pnum(14,:);
ValErrors = Valnum(1,2:end);
ErrorRate = Anum(2,:);
Freq_MeanAbsDev = Omnum(2,:);
Gamma_Areas = Gnum(1,:);

% generate plots
figure, plot(param_vals,ValErrors,'b*')
xlabel('Loop Parameter'); ylabel('Validation Error');

figure, plot(param_vals,ErrorRate,'b*')
xlabel('Loop Parameter'); ylabel('Error Rate (%)');

figure, plot(param_vals,Freq_MeanAbsDev,'b*')
xlabel('Loop Parameter'); ylabel('Mean Absolute Deviation of Frequencies');

figure, plot(param_vals,Gamma_Areas,'b*')
xlabel('Loop Parameter'); ylabel('Area between Est./Actual Coupling Functions');


%%

% Parameter File
rowHeadings = {'info','w','A','K','Gamma','dt','tmax','noise','ts_skip','num_repeats','learning_rate', ...
'n_epochs','batch_size','n_oscillators','n_coefficients','reg','prediction_method','loop_parameter', ...
'parameter','attempt','network','method'};
params = cell2struct(Praw(:,2:end),rowHeadings,1);


