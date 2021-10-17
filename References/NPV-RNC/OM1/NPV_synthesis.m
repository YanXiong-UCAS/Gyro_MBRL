%% control synthesis
clear; close all

load('p.mat');

% locked state: q4,q4d
% selected variables:
%   x: q2,q3,q1d,q2d,q3d
%   u: u1,u2,u3
xs=[2,3,5,6,7]; us=[1,2,3];
nx=length(xs); nu=length(us);

%% gridding method
q2m=-pi/3; q2M=pi/3; q2N=5;
q2=linspace(q2m,q2M,q2N);
q3m=-pi/3; q3M=pi/3; q3N=5;
q3=linspace(q3m,q3M,q3N);
q1dm=30; q1dM=60; q1dN=3;
q1d=linspace(q1dm,q1dM,q1dN);
q2dm=-1; q2dM=1; q2dN=3;
q2d=linspace(q2dm,q2dM,q2dN);
q3dm=-1; q3dM=1; q3dN=3;
q3d=linspace(q3dm,q3dM,q3dN);

Kp.dim=[q2N; q3N; q1dN; q2dN; q3dN];
Kp.step=[q2(2)-q2(1); q3(2)-q3(1); q1d(2)-q1d(1); q2d(2)-q2d(1); q3d(2)-q3d(1)];
Kp.range=[q2m,q2M;
          q3m,q3M;
          q1dm,q1dM;
          q2dm,q2dM;
          q3dm,q3dM];
      
% Variables
W=sdpvar(nx);
Yp=sdpvar(nu,nx,q2N,q3N,q1dN,q2dN,q3dN);

lambda=0.5; LMI = [];

for q2i=1:q2N
    for q3i=1:q3N
        for q1di=1:q1dN
            for q2di=1:q2dN
                for q3di=1:q3dN
                    q=zeros(4,1); qd=zeros(4,1);
                    q(2)=q2(q2i); q(3)=q3(q3i); 
                    qd(1)=q1d(q1di); qd(2)=q2d(q2di); qd(3)=q3d(q3di);
                    Y=Yp(:,:,q2i,q3i,q1di,q2di,q3di); 
                    [A,B]=NPV_model(q,qd,p,xs,us);
                    LMI = [LMI;A*W+W*A'+B*Y+Y'*B'+2*lambda*W <= 0];
                end
            end
        end
    end
end

LMI = [LMI;W>=1e-3*eye(nx)];

optimize(LMI,[],sdpsettings('verbose',1,'solver','sdpt3'))

W = value(W); Wi=inv(W); Yp = value(Yp);

Kp.K=zeros(nu,nx,q2N,q3N,q1dN,q2dN,q3dN);
for q2i=1:q2N
    for q3i=1:q3N
        for q1di=1:q1dN
            for q2di=1:q2dN
                for q3di=1:q3dN
                    Kp.K(:,:,q2i,q3i,q1di,q2di,q3di)=Yp(:,:,q2i,q3i,q1di,q2di,q3di)*Wi; 
                end
            end
        end
    end
end

save('NPV_Kp.mat','Kp');