function dx=gyro(t,x,u,p,LS)
q=x(1:4); qd=x(5:8);
M=inertia(q,p);
C=coriolis(q,qd,p);
Fv=blkdiag(p.fv1,p.fv2,p.fv3,p.fv4);
Km=blkdiag(p.Km1,p.Km2,p.Km3,p.Km4);
T=Km*u-(C+Fv)*qd;
%% LS -- lock state: [D,B,R,S] 
for k=1:4
    if LS(k)
        M(k,:)=0; M(:,k)=0; M(k,k)=1; T(k)=0;
    end
end
qdd=M\T;
dx=[qd; qdd];
end