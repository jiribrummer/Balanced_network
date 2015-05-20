%% K-means clustering for CV values and synchrony measures




gext_min = 2;
gext_max = 4;

ginh_min = 2;
ginh_max = 4;

colorplot_matrix = zeros(gext_max-gext_min, ginh_max-ginh_min);

% Create matrix for k-means clustering
matrix = cat(2, cv_values, synchrony_values);

% Returns vector with cluster number as index
indices = kmeans(matrix, 4);

% Loop to make matrix for colorplot
index_value = 1;
disp(index_value)
 for gext = 1:gext_max-gext_min
    for ginh = 1:ginh_max-ginh_min
        colorplot_matrix(gext,ginh) = indices(index_value);
        index_value = index_value + 1;
    end
 end

pcolor(colorplot_matrix)