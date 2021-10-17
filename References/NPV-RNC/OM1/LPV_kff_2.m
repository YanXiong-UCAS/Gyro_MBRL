function Mu=LPV_kff_2(Xg,Xr,Ur,Qdd,p,Fv,Km)
qdr=[Xr(3:5); 0]; qddr=[Qdd; 0]; q=Xg(1:4); qd=Xg(5:8);
M=inertia(q,p);
C=coriolis(q,qd,p);
Ic=(M*qddr+(C+Fv)*qdr)./Km;
Mu=Ic(1:3);
end