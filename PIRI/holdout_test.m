% paired t-test for leave a few years out

lp_gps = zeros(5,1);
std_gps = zeros(5,1);
lp_baselines = zeros(5,1);
std_baselines = zeros(5,1);
for holdout_years=1:5
    lp_gp = csvread("./results/gp_holdout_" + int2str(holdout_years) + ".csv", 0,0); 
    lp_baseline = csvread("./results/baseline_holdout_" + int2str(holdout_years) + ".csv", 0,0); 
    [~,p] = ttest(lp_gp,lp_baseline);
    fprintf("leave %d years out\n", holdout_years);
    fprintf("diff in averaged lp bewtween gp and baseline: %0.3f \n",...
        mean(lp_gp)-mean(lp_baseline));
    fprintf("p value: %0.3f \n",p);
    lp_gps(holdout_years) = mean(lp_gp);
    lp_baselines(holdout_years) = mean(lp_baseline);
    std_gps(holdout_years) = std(lp_gp)/sqrt(numel(lp_gp));
    std_baselines(holdout_years) = std(lp_baseline)/sqrt(numel(lp_baseline));
end

fig = figure(1);
% plot(1:5,lp_gps); hold on;
% plot(1:5,lp_baselines); hold on;
% for holdout_years=1:5
%     u = lp_gps(holdout_years) + 2*std_gps(holdout_years);
%     l = lp_gps(holdout_years) - 2*std_gps(holdout_years);
%     plot([holdout_years;holdout_years],[l,u]);
% end

errorbar(1:5,lp_gps,std_gps); hold on;
errorbar(1:5,lp_baselines,std_baselines);
title('Avg pred log lik' ,'FontSize', 16);
legend(["gp", "baseline"],...
    'Location', 'best','NumColumns',1, 'FontSize', 16);
xlim([0,6]);
xlabel("holdout years",'FontSize', 12);
ylabel("avg ll",'FontSize', 12);
legend('boxoff');