%% Return plot for ISI's

x = zeros(1,100);
y = zeros(1,100);

for i = 1:100
    x(i) = isi(i);
    y(i) = isi(i+1);
end

scatter(x,y)

matrix = cat(2, x', y');

indices = kmeans(matrix, 3);

gscatter(x,y,indices)
set(gca,'xscale','log')
set(gca,'yscale','log')