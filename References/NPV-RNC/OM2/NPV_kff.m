function Mu=NPV_kff(Xg,Xr,Ur,Qdd,p,Fv,Km)
% qdr=[Xr(3:5); 0]; qddr=[Qdd; 0]; q=Xg(1:4); 
% M=inertia(q,p);
% C=coriolis(q,qdr,p);
% Ic=(M*qddr+(C+Fv)*qdr)./Km;
% Mu=Ic(1:3);
Mu=Ur;
end