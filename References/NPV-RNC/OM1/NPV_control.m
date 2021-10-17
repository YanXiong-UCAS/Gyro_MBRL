function u=NPV_control(x,xr,mu,Kp,rho)
L=5; tau=1/L;
q2=rho(1); q3=rho(2);
K=zeros(3,5);
% x:q2,q3,q1d,q2d,q3d
% rho:q2,q3,q1d,q2d,q3d
for j=0:L-1
    s=j*tau;
    gs=xr*(1-s)+x*s;
    rh=[q2; q3; gs(3:5)];
    Ks=lookup_table(Kp,rh);
    K=K+Ks;
end
K=K/L;
u=mu+K*(x-xr);
end

