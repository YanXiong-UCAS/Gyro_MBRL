clear;
load('p.mat');
ts=0.001; L=4000; t=ts*(0:L); 
ic=zeros(4,L+1); 
q1d=0*t; q1dd=0*t;
q2=0*t; q2d=0*t; q2dd=0*t;
q3=0*t; q3d=0*t; q3dd=0*t;
r=0.8; f=0.8; w=2*pi*f; A=10; 
Fv=blkdiag(p.fv1,p.fv2,p.fv3,p.fv4);
Km=[p.Km1; p.Km2; p.Km3; p.Km4];

for k=1:L+1
    s=sin(w*t(k)); c=cos(w*t(k));
    q1d(k)=45+A*s; q1dd(k)=A*w*c; 
    q2(k)=r*s; q2d(k)=r*w*c; q2dd(k)=-r*w^2*s;
    q3(k)=r*c; q3d(k)=-r*w*s;q3dd(k)=-r*w^2*c;
    q=[0;q2(k);q3(k);0]; qd=[q1d(k);q2d(k);q3d(k);0]; qdd=[q1dd(k);q2dd(k);q3dd(k);0];
    M=inertia(q,p);
    C=coriolis(q,qd,p);
    ic(:,k)=(M*qdd+(C+Fv)*qd)./Km;
end
xr=[q2; q3; q1d; q2d; q3d];
Qdd=[q1dd; q2dd; q3dd];
ur=ic(1:3,:);
save('reference.mat','t','ts','L','Fv','Km','p','xr','ur','Qdd');