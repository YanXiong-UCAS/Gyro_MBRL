clear; close all
load('reference.mat');
load('LPV_Kp.mat');

% x: q2,q3,q1d,q2d,q3d
% u: i1,i2,i3
xs=[2,3,5,6,7]; us=[1,2,3]; rhos=[2,3,5,6,7];
nx=length(xs); nu=length(us); 
x=zeros(nx,L+1); u=zeros(nu,L+1); mu=zeros(nu,L+1);

% gyroscope initial state
Xg=zeros(8,1); Xg(2)=-0.8; Xg(3)=-0.5; Xg(5)=55; 
% lock state: [D,B,R,S]
LS=[0,0,0,1];

for k=1:L+1
    x(:,k)=Xg(xs);
    mu(:,k)=LPV_kff_1(Xg,xr(:,k),ur(:,k),Qdd(:,k),p,Fv,Km);
    u(:,k)=LPV_control(x(:,k),xr(:,k),mu(:,k),Kp,Xg(rhos));
    Ug=[u(:,k); 0];
    Xg=ode45_m(@(t,x)gyro(t,x,Ug,p,LS),[0 ts],Xg);
end

DS=2;
LPV_reference_1.t=downsample(t,DS);
LPV_reference_1.x=downsample(x',DS)';
LPV_reference_1.xr=downsample(xr',DS)';
save('LPV_reference_1.mat','LPV_reference_1');

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