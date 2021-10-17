%% ode45
function x1=ode45_m(mdl,tspan,x0)
ts=tspan(2)-tspan(1);
k1=ts*mdl(0,x0);
k2=ts*mdl(ts/2,x0+k1/2);
k3=ts*mdl(ts/2,x0+k2/2);
k4=ts*mdl(ts,x0+k3);
x1=x0+(k1+2*k2+2*k3+k4)/6;
end