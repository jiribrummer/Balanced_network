%% Create CV colorplot 

pcolor(xvalues(1:9),yvalues(1:9),zvalues)

%% create CV surface plot

surf(xvalues(1:9),yvalues(1:9),zvalues, synchrony)
xlabel('ginh')
ylabel('gext')
zlabel('regularity')

