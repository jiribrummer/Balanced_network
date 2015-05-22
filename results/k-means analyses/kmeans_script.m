%% K-means clustering for CV values and synchrony measures


[rx, cx] = size(xaxis);
[ry, cy] = size(yaxis);


colorplot_matrix = zeros(cy, cx);

% Create matrix for k-means clustering
matrix = cat(2, cv_data_kmeans', freq_data_kmeans');

% Returns vector with cluster number as index
indices = kmeans(matrix, 4);

% Loop to make matrix for colorplot
index_value = 1;
 for gext_index = 1:cy
    for ginh_index = 1:cx
        colorplot_matrix(gext_index,ginh_index) = indices(index_value);
        index_value = index_value + 1;
    end
 end

pcolor_fleur(xaxis,yaxis,colorplot_matrix)