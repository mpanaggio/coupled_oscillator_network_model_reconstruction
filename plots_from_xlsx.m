%pkg load io   % load I/O package for xlsread, needed in Octave only

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

% generate plots with ALL runs/attempts
%figure, plot(param_vals,ValErrors,'b*')
%xlabel('Loop Parameter'); ylabel('Validation Error');
%
%figure, plot(param_vals,ErrorRate,'b*')
%xlabel('Loop Parameter'); ylabel('Error Rate (%)');
%
%figure, plot(param_vals,Freq_MeanAbsDev,'b*')
%xlabel('Loop Parameter'); ylabel('Mean Absolute Deviation of Frequencies');
%
%figure, plot(param_vals,Gamma_Areas,'b*')
%xlabel('Loop Parameter'); ylabel('Area between Est./Actual Coupling Functions');


% Parameter File
rowHeadings = {'info','w','A','K','Gamma','dt','tmax','noise','ts_skip','num_repeats','learning_rate', ...
'n_epochs','batch_size','n_oscillators','n_coefficients','reg','prediction_method','loop_parameter', ...
'parameter','attempt','network','method'};
params = cell2struct(Praw(:,2:end),rowHeadings,1);
loop_parameter = params(1).loop_parameter;


% initialize
i = 1; ind_max = length(params);
curr_param = params(1).parameter; curr_net = 1; % first network
keep_index = []; ind_to_keep = 0; best_val = Inf;


while i <= ind_max % step through all runs
	param_val = params(i).parameter;
	net = params(i).network;
	
	if net == curr_net && param_val == curr_param % if same parameters & network as last
		% compare to the last / current
		if ValErrors(i) < best_val
			ind_to_keep = i;
		end
		
	else % if new parameter value / network
	  
		% store the last result
		keep_index = [keep_index ind_to_keep];
		
		% on to the next network
		curr_param = param_val;
		curr_net = net;
		
		% this one is "best attempt so far"
		best_val = ValErrors(i);
		ind_to_keep = i;
		
	end
	
	i = i+1;
end
keep_index = [keep_index ind_to_keep]; % the last network's best attempt


% plot any desired results from the best attempts
figure, plot(param_vals(keep_index),ValErrors(keep_index),'b*')
xlabel(loop_parameter,'Interpreter','none'); ylabel('Validation Error');

figure, plot(param_vals(keep_index),ErrorRate(keep_index),'b*')
xlabel(loop_parameter,'Interpreter','none'); ylabel('Error Rate (%)');

figure, plot(param_vals(keep_index),Freq_MeanAbsDev(keep_index),'b*')
xlabel(loop_parameter,'Interpreter','none'); ylabel('Mean Absolute Deviation of Frequencies');

figure, plot(param_vals(keep_index),Gamma_Areas(keep_index),'b*')
xlabel(loop_parameter,'Interpreter','none'); ylabel('Area between Est./Actual Coupling Functions');