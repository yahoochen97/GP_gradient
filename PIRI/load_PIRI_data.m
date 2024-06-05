% read data
data = readtable("hb_data_complete.csv");

% clean data
% all zero PIRI for new zealand and netherland
data = data(~ismember(data.country, {'N-ZEAL','NETHERL'}),:);

countries = unique(data.country);
years = unique(data.year);

n = numel(countries);
m = numel(years);

treated = zeros(n,m);
for i=1:n
    for j=1:m
        if data.AIShame(data.year==years(j) & strcmp(data.country,countries(i))==1)
            treated(i,j)=1;
        end
    end
end

% h = heatmap(treated, 'ylabel','country', 'ColorbarVisible', 'off');
% cdl = h.YDisplayLabels;                                    % Current Display Labels
% h.YDisplayLabels = repmat(' ',size(cdl,1), size(cdl,2));   % Blank Display Labels

country_dict = containers.Map(countries, 1:n);
year_dict = containers.Map(years, 1:m);

% x is:
% 1: year number
% 2: country id
% 3: AIShame (treatment indicator)
% 4: cat_rat
% 5: ccpr_rat
% 6: democratic
% 7: log(gdppc)
% 8: log(pop)
% 9: Civilwar2
% 10: War
x = zeros(size(data,1), 10);
x(:,1) = arrayfun(@(x) year_dict(x), data.year);
x(:,2) = cellfun(@(x) country_dict(x), data.country);
% x(:,2) = 1;
% x(ismember(data.country, unique(data.country(data.PIRI==1))),2) = 2;
x(:,3) = data.AIShame;
x(:,4) = data.cat_rat;
x(:,5) = data.ccpr_rat;
x(:,6) = data.democratic;
x(:,7) = data.log_gdppc;
x(:,8) = data.log_pop;
x(:,9) = data.Civilwar2;
x(:,10) = data.War;
y = data.PIRI;