%% K-means clustering for CV values and synchrony measures


[rx, cx] = size(xvalues);
[ry, cy] = size(yvalues);


colorplot_matrix = zeros(cy-1, cx-1);

% Create matrix for k-means clustering
matrix = cat(2, cv_values', synchrony_values');

% Returns vector with cluster number as index
indices = kmeans(matrix, 4);

% Loop to make matrix for colorplot
index_value = 1;
 for gext_index = 1:cy-1
    for ginh_index = 1:cx-1
        colorplot_matrix(gext_index,ginh_index) = indices(index_value);
        index_value = index_value + 1;
    end
 end

pcolor_fleur(xvalues(1:end-1),yvalues(1:end-1),colorplot_matrix)