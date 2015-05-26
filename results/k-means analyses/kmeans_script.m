%% perform kmeans analyses with different cluster sizes

% Create matrix for k-means clustering
matrix = cat(2, cv_data_kmeans', freq_data_kmeans');

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
matrix = cat(2, cv_data_kmeans', freq_data_kmeans');

% Returns vector with cluster number as index
indices = kmeans(matrix, 4, 'display', 'final', 'replicates', 20);

% Loop to make matrix for colorplot
index_value = 1;
 for gext_index = 1:cy
    for ginh_index = 1:cx
        colorplot_matrix(gext_index,ginh_index) = indices(index_value);
        index_value = index_value + 1;
    end
 end

pcolor_fleur(xaxis,yaxis,colorplot_matrix);

