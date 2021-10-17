clear; close all
load('reference.mat');
load('NPV_Kp_perf.mat');

% x: q2,q3,q1d,q2d,q3d
% u: i1,i2,i3
xs=[2,3,5,6,7]; us=[1,2,3]; rhos=[2,3,5,6,7];
nx=length(xs); nu=length(us); 
x=zeros(nx,L+1); u=zeros(nu,L+1); mu=zeros(nu,L+1); zs=zeros(1,L+1); zsi=0;

% gyroscope initial state
Xg=zeros(8,1); Xg(2)=-0.8; Xg(3)=-0.5; Xg(5)=55;
% lock state: [D,B,R,S]
LS=[0,0,0,1];
ps=p; 
% ps.fv1=3*ps.fv1; ps.fv2=3*ps.fv2; ps.fv3=3*ps.fv3;
for k=1:L+1
    x(:,k)=Xg(xs);
    mu(:,k)=NPV_kff(Xg,xr(:,k),ur(:,k),Qdd(:,k),p,Fv,Km);
    u(:,k)=NPV_control(x(:,k),xr(:,k),mu(:,k),Kp,Xg(rhos));
    z=Kp.Cx*(x(:,k)-xr(:,k))+Kp.Du*(u(:,k)-ur(:,k));
    zs(k)=z'*z;
    zsi=zsi+zs(k)*ts;
    Ug=[u(:,k); 0];
    Xg=ode45_m(@(t,x)gyro(t,x,Ug,ps,LS),[0 ts],Xg);
end

DS=5;
NPV_reference.t=downsample(t,DS);
NPV_reference.x=downsample(x',DS)';
NPV_reference.xr=downsample(xr',DS)';
NPV_reference.zs=downsample(zs,DS);
NPV_reference.zsi=zsi;
save('NPV_reference_perf.mat','NPV_reference');

%% Plotting
figure;
subplot(311);hold on;
plot(t,x(1,:));
plot(t,xr(1,:),'k--');
grid on;
xlabel('Time [s]','interpreter','latex');
ylabel('$q_2$ [rad]','interpreter','latex');

subplot(312);hold on;
plot(t,x(2,:));
plot(t,xr(2,:),'k--');
grid on;
xlabel('Time [s]','interpreter','latex');
ylabel('$q_3$ [rad]','interpreter','latex');

subplot(313);hold on;
plot(t,x(3,:));
plot(t,xr(3,:),'k--');
grid on;
xlabel('Time ($t$) [s]','interpreter','latex');
ylabel('$\dot{q}_1$ [rad/s]','interpreter','latex');