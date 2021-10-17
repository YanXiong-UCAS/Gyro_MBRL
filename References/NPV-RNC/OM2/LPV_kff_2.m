function Mu=LPV_kff_2(Xg,Xr,Ur,p,Fv,Km)
qdr=[Xr(2:3); 0; Xr(4)]; q=Xg(1:4); qd=Xg(5:8);
C=coriolis(q,qd,p);
Ic=((C+Fv)*qdr)./Km;
Mu=Ic(1:2);
end