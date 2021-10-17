%% control synthesis
clear; close all

load('p.mat');

% locked state: q3,q3d
% selected variables:
%   x: q4,q1d,q2d,q4d
%   u: u1,u2
% rho: q2,q1d,q2d,q4d
%   d: friction on q1d
%   z: [Cx; Du]
xs=[4,5,6,8]; us=[1,2];
nx=length(xs); nu=length(us);
nd=1; nz=nx+nu;
Bd=[zeros(1,nd); eye(nd); zeros(2,nd)];
Cx=[blkdiag(1,0.1,5,1); zeros(nu,nx)];
Du=[zeros(nx,nu); blkdiag(0.2,0.2)];
Dd=zeros(nz,nd);
Id=eye(nd); Iz=eye(nz); Ih=eye(nx+nd+nz);

Kp.Cx=Cx; Kp.Du=Du;
%% gridding method
q2m=-pi/3; q2M=pi/3; q2N=5;
q2=linspace(q2m,q2M,q2N);
q1dm=30; q1dM=60; q1dN=3;
q1d=linspace(q1dm,q1dM,q1dN);
q2dm=-1; q2dM=1; q2dN=3;
q2d=linspace(q2dm,q2dM,q2dN);
q4dm=-1; q4dM=1; q4dN=3;
q4d=linspace(q4dm,q4dM,q4dN);

Kp.dim=[q2N; q1dN; q2dN; q4dN];
Kp.step=[q2(2)-q2(1); q1d(2)-q1d(1); q2d(2)-q2d(1); q4d(2)-q4d(1)];
Kp.range=[q2m,q2M;
          q1dm,q1dM;
          q2dm,q2dM;
          q4dm,q4dM];
      
% Variables
W=sdpvar(nx);
Yp=sdpvar(nu,nx,q2N,q1dN,q2dN,q4dN);

sdpvar alpha;
lambda=10;
eps=1e-4;
LMI = [];

for q2i=1:q2N
    for q1di=1:q1dN
        for q2di=1:q2dN
            for q4di=1:q4dN
                q=zeros(4,1); qd=zeros(4,1);
                q(2)=q2(q2i); 
                qd(1)=q1d(q1di); qd(2)=q2d(q2di); qd(4)=q4d(q4di);
                Y=Yp(:,:,q2i,q1di,q2di,q4di); 
                [A,B]=LPV_model(q,qd,p,xs,us);
                Wc=A*W+W*A'+B*Y+Y'*B';
                P=[Wc,Bd,(Cx*W+Du*Y)';
                   Bd',-alpha*Id, Dd';
                   Cx*W+Du*Y,Dd,-alpha*Iz];
                LMI = [LMI; P<= -eps*Ih; Wc+2*lambda*W>=0];
            end
        end
    end
end

LMI = [LMI;W>=eps*eye(nx)];

optimize(LMI,alpha,sdpsettings('verbose',1,'solver','sdpt3'))

W = value(W); Wi=inv(W); Yp = value(Yp);

Kp.alpha=value(alpha);
Kp.M=Wi;
Kp.alpha

Kp.K=zeros(nu,nx,q2N,q1dN,q2dN,q4dN);
for q2i=1:q2N
    for q1di=1:q1dN
        for q2di=1:q2dN
            for q4di=1:q4dN
                Kp.K(:,:,q2i,q1di,q2di,q4di)=Yp(:,:,q2i,q1di,q2di,q4di)*Wi; 
            end
        end
    end
end

save('LPV_Kp_perf.mat','Kp');