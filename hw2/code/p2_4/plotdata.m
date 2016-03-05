
function [] = plotdata(D)

I0 = find(D.y==0); 
I1 = find(D.y==1); 

h=plot(D.X(I0,1),D.X(I0,2),'ro'); hold on;
set(h,'MarkerSize',5);

h=plot(D.X(I1,1),D.X(I1,2),'bx'); hold off;
set(h,'MarkerSize',5);
