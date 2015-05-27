%% Create CV colorplot 

pcolor_fleur([1 2], [2 3], [ 4 5; 6 7]);

%pcolor_fleur(xaxis,yaxis,zvalues);


%% create surface plot

surf(xaxis,yaxis, syn_values, cv_values);
xlabel('ginh')
ylabel('gext')
zlabel('synchrony')

