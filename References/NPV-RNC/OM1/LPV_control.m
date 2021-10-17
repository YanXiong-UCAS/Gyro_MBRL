function u=LPV_control(x,xr,mu,Kp,rho)
K=lookup_table(Kp,rho);
u=mu+K*(x-xr);
end

