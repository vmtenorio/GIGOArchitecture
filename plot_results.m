clear
exp_type = 1;

if exp_type == 1  
    load('.\results\matfiles\perturbation.mat')
    if size(error,1) == 4
        fmts = {'o-', '+-', 'X--', 'v--'};
        legend_txt = {'Enc/Dec Wei(192)', 'Enc/Dec NoUps(192)', ...
                      'AutoConv (210)', 'AutoFC (193)'};
        x_lab = '% of perturbed links';
        lims = [0.004 0.1]; % without noise
        %lims = [0.05 0.3]; % with noise
    elseif size(error,1) == 5
        load('.\results\matfiles\perturbation.mat')
        fmts = {'o-', '+-', '^-', 'X--', 'v--'};
        legend_txt = {'Enc/Dec Wei(132)', 'Enc/Dec NoUps(132)', ...
                      'Enc/Dec Wei(102)', 'AutoConv (140)', 'AutoFC (128)'};
        x_lab = '% of perturbed links';
        lims = [0.005 0.2]; %  without noise
        % lims = [0.05 0.3]; %  with noise
    elseif size(error,1) == 9
        load('.\results\matfiles\perturbation.mat')
        fmts = {'X-', 'o-', '+-', 'o--', '^--', '+--', 'o:', '^:', '+:'};
        legend_txt = {'AutoFC (769)', 'Enc/Dec Wei(440)', 'AutoConv (440)',...
              'Enc/Dec Wei(298)', 'Enc/Dec NoUps(298)', 'AutoConv (280)',...
              'Enc/Dec Wei(132)', 'Enc/Dec NoUps(132)', 'AutoConv (140)'};
        x_lab = '% of perturbed links';
        lims = [0.05 0.2]; %  without noise
    end
elseif exp_type == 2    % Adding noise to links perturbed Graphs
    load('.\results\matfiles\noise.mat')
    fmts = {'o-', '+-', 'X-', 'o--', 'o:', '+--'};
    legend_txt = {'Enc/Dec Wei(192)', 'AutoConv (210)',...
                  'AutoFC (193)', 'Enc/Dec Wei(132)', 'Enc/Dec Wei(102)',...
                  'AutoConv (140)'};
    x_lab = 'Normalized noise power';
    lims = [0.01 0.5];
elseif exp_type == 3    % Adding noise to 256 and 226 SBMs
    load('.\results\matfiles\noise2.mat')
    fmts = {'X-', 'o-', '+-', 'o--', '+--', 'o:', '+:'};
    legend_txt = {'AutoencFC(709)', 'Enc/Dec WEI (440)', 'AutoencConv (429)',...
              'Enc/Dec WEI (298)', 'AutoencConv (308)', 'Enc/Dec Wei(132)',...
              'AutoConv (144)'};              
    x_lab = 'Normalized noise power';
    lims = [0.01 0.7];
elseif exp_type == 4
    load('.\results\matfiles\nodes_pert.mat')
    x_lab = 'Number of deleted nodes';
    legend_txt = {'AutoencFC (709)', 'Enc/Dec (440)', 'Enc/Dec (298)',...
                  'AutoencConv (308)', 'Enc/Dec (132)', 'AutoencConv (143)'};
    fmts = {'X-', 'o-', 'o--', '+--', 'o:', '+:'};
    lims = [0.04 0.4];   % With noise
    % lims = [0.1 0.6];   % With noise
end

n_exps = size(error,1);
% fmts = cellstr(fmts);
% median_err = zeros(length(n_p),n_exps);
% for k=1:size(error,3)
%     median_err(:,k) = median(error(:,:,k),2);
% end

% Plot median
figure();
for i=1:n_exps
    semilogy(n_p,error(i,:),fmts{i},'LineWidth',1.5,'MarkerSize',8);hold on
end
hold off
legend(legend_txt, 'FontSize', 11, 'Location', 'southeast');set(gca,'FontSize',14);
ylabel('Median Error','FontSize', 20);
xlabel(x_lab,'FontSize', 20);grid on;axis tight
ylim(lims)
set(gcf, 'PaperPositionMode', 'auto')