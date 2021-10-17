clear;
load('p.mat');
ts=0.005; L=8000; t=ts*(0:L); 
ic=zeros(4,L+1); 
q1d=0*t; 
q2d=0*t; 
q4=0*t; q4d=0*t; 
Fv=blkdiag(p.fv1,p.fv2,p.fv3,p.fv4);
Km=[p.Km1; p.Km2; p.Km3; p.Km4];
% q4r=[-0.9, 0.9, -0.9, 0.9]*pi/2.5;
q4r=[-0.9, 0.9, -0.9, 0.9]*pi/1;
q1dr=[55, 35, 55, 35];
for k=1:L+1
    j=floor(t(k)/10)+1;
    if j > 4
        j=4;
    end
    q4(k)=q4r(j); q1d(k)=q1dr(j);
    q=[0;0;0;q4(k)];
%     q=[0;-0.5*q4(k);0;q4(k)]; 
    qd=[q1d(k);q2d(k); 0; q4d(k)]; 
    C=coriolis(q,qd,p);
    ic(:,k)=(C+Fv)*qd./Km;
end
xr=[q4; q1d; q2d; q4d];
ur=ic(1:2,:);
save('setpoint.mat','t','ts','L','Fv','Km','p','xr','ur');
