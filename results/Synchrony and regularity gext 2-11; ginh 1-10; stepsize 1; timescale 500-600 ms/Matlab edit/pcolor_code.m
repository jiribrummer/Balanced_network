%% Create CV colorplot 


pcolor_fleur(xaxis,yaxis,zvalues_syn);


%% create surface plot

surf(xaxis, yaxis, zvalues_syn, zvalues_cv);
xlabel('ginh')
ylabel('gext')
zlabel('synchrony')

