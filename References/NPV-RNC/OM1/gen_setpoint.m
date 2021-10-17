clear;
load('p.mat');
ts=0.002; L=10000; t=ts*(0:L); 
ic=zeros(4,L+1); 
q1d=0*t; 
q2=0*t; q2d=0*t; 
q3=0*t; q3d=0*t; 
Fv=blkdiag(p.fv1,p.fv2,p.fv3,p.fv4);
Km=[p.Km1; p.Km2; p.Km3; p.Km4];
q2r=[0.9, -0.8, 0.8, -0.9];
q3r=[-0.9, 0.8, -0.8, 0.9];
q1dr=[55, 40, 50, 35];
for k=1:L+1
    j=floor(t(k)/5)+1;
    if j > 4
        j=4;
    end
    q2(k)=q2r(j); q3(k)=q3r(j); q1d(k)=q1dr(j);
    q=[0;q2(k);q3(k);0]; qd=[q1d(k);q2d(k);q3d(k);0]; 
    C=coriolis(q,qd,p);
    ic(:,k)=(C+Fv)*qd./Km;
end
xr=[q2; q3; q1d; q2d; q3d];
Qdd=zeros(3,L+1);
ur=ic(1:3,:);
save('setpoint.mat','t','ts','L','Fv','Km','p','xr','ur','Qdd');