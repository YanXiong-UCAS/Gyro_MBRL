function u=NPV_control(x,xr,mu,Kp,rho)
L=10; tau=1/L;
q2=rho(1); 
K=zeros(2,4);
% x:q4,q1d,q2d,q4d
% rho:q2,q1d,q2d,q4d
for j=0:L-1
    s=j*tau;
    gs=xr*(1-s)+x*s;
    rh=[q2; gs(2:4)];
    Ks=lookup_table(Kp,rh);
    K=K+Ks;
end
K=K/L;
u=mu+K*(x-xr);
end

