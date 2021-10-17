%% global LPV model
function [A,B]=LPV_model(q,qd,p,xs,us)
M=inertia(q,p);
C=coriolis(q,qd,p);
Fv=blkdiag(p.fv1,p.fv2,p.fv3,p.fv4);
Km=blkdiag(p.Km1,p.Km2,p.Km3,p.Km4);
Af=[zeros(4,4),eye(4);
    zeros(4,4),-M\(C+Fv)]; 
Bf=[zeros(4,4); M\Km];
A=Af(xs,xs); B=Bf(xs,us);
end