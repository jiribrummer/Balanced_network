%% Create CV colorplot 


pcolor_fleur(xvalues2(1,1:end),yvalues(1,1:5),zvalues)

%% create CV surface plot

surf(xaxis,yaxis, freq_data, cv_data)
xlabel('ginh')
ylabel('gext')
zlabel('synchrony')

