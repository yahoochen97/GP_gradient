clear;
close all;
addpath("~/Documents/Washu/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;
addpath("utilities");
FONTSIZE=24;

fig=figure(1);
x = linspace(1,3,101);
y = x.^2/3;
plot(x,y,'-r','LineWidth',2); hold on;
x = linspace(1.2,2.8,101);
y = 4*(x-1)/3;
plot(x,y,'--b','LineWidth',2);
plot(linspace(2,2.5,101),4/3*ones(101,1),'-.k');
plot(5/2*ones(101,1),linspace(4/3,2.5*2.5/3,101),'-.k');

text(2.25,1.2,"{\Delta}x",'FontSize',FONTSIZE);
text(2.55,1.71,"{\Delta}y",'FontSize',FONTSIZE);

ylim([0,3]);
set(gca,'YTick',[]);
xticks([1.5,2,2.5]);
xticklabels(["high school","college","phd"]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',FONTSIZE);

ylabel('Prob of admission','FontSize',FONTSIZE);

legend(["Prob of admission","Gradient at college","{\Delta}x","{\Delta}y"],...
    'Location', 'northwest','NumColumns',1, 'FontSize',FONTSIZE);
legend('boxoff');

title("marginal effects with education", 'FontSize', FONTSIZE);
filename = "./results/marginal_effects_illustration.pdf";
set(fig, 'PaperPosition', [-0.9 0.0 11 6.2]); 
set(fig, 'PaperSize', [9.2 6.2]);
print(fig, filename, '-dpdf','-r300');
close;