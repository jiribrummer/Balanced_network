%% perform kmeans analyses with different cluster sizes

% Create matrix for k-means clustering
matrix = cat(2, kmeansdata_cv', kmeansdata_syn');

xrange = ones(1,8);
for i = 1:8
    xrange(1,i) = i;
end

yrange = zeros(1,8);

% Returns vector with cluster number as index
for clusters = 1:8
    [indices, menas, sumdist] = kmeans(matrix, clusters, 'replicates', 20);
    yrange(clusters) = sum(sumdist);
end

plot(xrange, yrange, '-o');

%% K-means clustering for CV values and synchrony measures


[rx, cx] = size(xaxis);
[ry, cy] = size(yaxis);


colorplot_matrix = zeros(cy, cx);

% Create matrix for k-means clustering
matrix = cat(2, kmeansdata_cv', kmeansdata_syn');

% Returns vector with cluster number as index
indices = kmeans(matrix, 5, 'display', 'final', 'replicates', 20);

% Loop to make matrix for colorplot
index_value = 1;
 for gext_index = 1:cy
    for ginh_index = 1:cx
        colorplot_matrix(gext_index,ginh_index) = indices(index_value);
        index_value = index_value + 1;
    end
 end

pcolor_fleur(xaxis,yaxis,colorplot_matrix);

%% K-means clustering for CV values and synchrony measures
% !!WITH NORMALIZATION!!


[rx, cx] = size(xaxis);
[ry, cy] = size(yaxis);


colorplot_matrix = zeros(cy, cx);

% Create matrix for k-means clustering

norm_kmeans_cv = kmeansdata_cv/(sum(kmeansdata_cv));
norm_kmeans_syn = kmeansdata_syn/(sum(kmeansdata_syn));

matrix = cat(2, norm_kmeans_cv', norm_kmeans_syn');
%matrix = norm_kmeans_syn';

% Returns vector with cluster number as index
indices = kmeans(matrix, 5, 'display', 'final', 'replicates', 20);

% Loop to make matrix for colorplot
index_value = 1;
 for gext_index = 1:cy
    for ginh_index = 1:cx
        colorplot_matrix(gext_index,ginh_index) = indices(index_value);
        index_value = index_value + 1;
    end
 end

colormap gray;
pcolor_fleur(xaxis,yaxis,colorplot_matrix);


%% perform kmeans analyses with different cluster sizes
% !!WITH NORMALIZATION!!

% Create matrix for k-means clustering

norm_kmeans_cv = kmeansdata_cv/(sum(kmeansdata_cv));
norm_kmeans_syn = kmeansdata_syn/(sum(kmeansdata_syn));

matrix = cat(2, norm_kmeans_cv', norm_kmeans_syn');
%matrix = norm_kmeans_syn';

xrange = ones(1,8);
for i = 1:8
    xrange(1,i) = i;
end

yrange = zeros(1,8);

% Returns vector with cluster number as index
for clusters = 1:8
    [indices, menas, sumdist] = kmeans(matrix, clusters, 'replicates', 20);
    yrange(clusters) = sum(sumdist);
end

subplot(1,2,1)
plot(xrange, yrange, '-o');

% K-means clustering for CV values and synchrony measures
% !!WITH NORMALIZATION!!


[rx, cx] = size(xaxis);
[ry, cy] = size(yaxis);


colorplot_matrix = zeros(cy, cx);

% Create matrix for k-means clustering

norm_kmeans_cv = kmeansdata_cv/(sum(kmeansdata_cv));
norm_kmeans_syn = kmeansdata_syn/(sum(kmeansdata_syn));

matrix = cat(2, norm_kmeans_cv', norm_kmeans_syn');
%matrix = norm_kmeans_syn';

% Returns vector with cluster number as index
indices = kmeans(matrix, 5, 'display', 'final', 'replicates', 20);

% Loop to make matrix for colorplot
index_value = 1;
 for gext_index = 1:cy
    for ginh_index = 1:cx
        colorplot_matrix(gext_index,ginh_index) = indices(index_value);
        index_value = index_value + 1;
    end
 end

subplot(1,2,2)
pcolor_fleur(xaxis,yaxis,colorplot_matrix);
xlabel('ginh (nS)')
ylabel('gext (nS)')