clear; close all
load('setpoint.mat');
load('LPV_Kp_perf.mat');

% x: q4,q1d,q2d,q4d
% u: i1,i2
xs=[4,5,6,8]; us=[1,2]; rhos=[2,5,6,8];
nx=length(xs); nu=length(us); 
x=zeros(nx,L+1); u=zeros(nu,L+1); mu=zeros(nu,L+1); zs=zeros(1,L+1); zsi=0;
q2=zeros(1,L+1);

% gyroscope initial state
Xg=zeros(8,1); Xg(5)=45; Xg(2)=0;
% lock state: [D,B,R,S]
LS=[0,0,1,0]; 
ps=p; 
% ps.fv1=3*ps.fv1; ps.fv2=3*ps.fv2;

for k=1:L+1
    x(:,k)=Xg(xs); q2(k)=Xg(2);
    mu(:,k)=LPV_kff_1(Xg,xr(:,k),ur(:,k),p,Fv,Km);
    u(:,k)=LPV_control(x(:,k),xr(:,k),mu(:,k),Kp,Xg(rhos));
    z=Kp.Cx*(x(:,k)-xr(:,k))+Kp.Du*(u(:,k)-ur(:,k));
    zs(k)=z'*z;
    zsi=zsi+zs(k)*ts;
    Ug=[u(:,k); zeros(2,1)];
    Xg=ode45_m(@(t,x)gyro(t,x,Ug,ps,LS),[0 ts],Xg);
end

DS=2;
LPV_setpoint_1.t=downsample(t,DS);
LPV_setpoint_1.x=downsample(x',DS)';
LPV_setpoint_1.xr=downsample(xr',DS)';
LPV_setpoint_1.q2=downsample(q2,DS);
LPV_setpoint_1.zs=downsample(zs,DS);
LPV_setpoint_1.zsi=zsi;
save('LPV_setpoint_1_perf.mat','LPV_setpoint_1');

%% Plotting
figure;
subplot(211);hold on;
plot(t,x(1,:));
plot(t,xr(1,:),'k--');
grid on;
xlabel('Time [s]','interpreter','latex');
ylabel('$q_4$ [rad]','interpreter','latex');

subplot(212);hold on;
plot(t,x(2,:));
plot(t,xr(2,:),'k--');
grid on;
xlabel('Time ($t$) [s]','interpreter','latex');
ylabel('$\dot{q}_1$ [rad/s]','interpreter','latex');

figure(2);
plot(t,q2);