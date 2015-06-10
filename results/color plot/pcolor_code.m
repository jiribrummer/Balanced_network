%% Create CV colorplot 

subplot(1,2,1)
pcolor_fleur(xaxis,yaxis,zvalues_cv);
xlabel('ginh (nS)')
ylabel('gext (nS)')
subplot(1,2,2)
pcolor_fleur(xaxis,yaxis,zvalues_syn);
xlabel('ginh (nS)')
ylabel('gext (nS)')

%% create CV surface plot

surf(xaxis,yaxis, zvalues_syn, zvalues_cv)
xlabel('ginh (nS)')
ylabel('gext (nS)')
zlabel('synchrony')

