% clear; close all
load('LPV_setpoint_1.mat');
load('LPV_setpoint_2.mat');
load('NPV_setpoint.mat');

lw=1.5; fz=12; lfz=16;
figure(1);
subplot(311);hold on; k=1;
plot(LPV_setpoint_1.t,LPV_setpoint_1.xr(k,:),'k--');
plot(LPV_setpoint_1.t,LPV_setpoint_1.x(k,:),'r','linewidth',lw);
plot(LPV_setpoint_2.t,LPV_setpoint_2.x(k,:),'b','linewidth',lw);
plot(NPV_setpoint.t,NPV_setpoint.x(k,:),'color',[0,0.3,0],'linewidth',lw);
legend('Ref','LPV1','LPV2','NPV');
grid on;
set(gca,'fontsize',fz);
% xlabel('Time [s]','interpreter','latex','fontsize',lfz);
ylabel('$q_2$ [rad]','interpreter','latex','fontsize',lfz);

subplot(312);hold on; k=2;
plot(LPV_setpoint_1.t,LPV_setpoint_1.xr(k,:),'k--');
plot(LPV_setpoint_1.t,LPV_setpoint_1.x(k,:),'r','linewidth',lw);
plot(LPV_setpoint_2.t,LPV_setpoint_2.x(k,:),'b','linewidth',lw);
plot(NPV_setpoint.t,NPV_setpoint.x(k,:),'color',[0,0.3,0],'linewidth',lw);
% legend('Ref','LPV1','LPV2','NPV');
grid on;
set(gca,'fontsize',fz);
% xlabel('Time [s]','interpreter','latex','fontsize',lfz);
ylabel('$q_3$ [rad]','interpreter','latex','fontsize',lfz);

subplot(313);hold on; k=3;
plot(LPV_setpoint_1.t,LPV_setpoint_1.xr(k,:),'k--');
plot(LPV_setpoint_1.t,LPV_setpoint_1.x(k,:),'r','linewidth',lw);
plot(LPV_setpoint_2.t,LPV_setpoint_2.x(k,:),'b','linewidth',lw);
plot(NPV_setpoint.t,NPV_setpoint.x(k,:),'color',[0,0.3,0],'linewidth',lw);
% legend('Ref','LPV1','LPV2','NPV');
grid on;
set(gca,'fontsize',fz);
xlabel('Time [s]','interpreter','latex','fontsize',lfz);
ylabel('$\dot{q}_1$ [rad/s]','interpreter','latex','fontsize',lfz);
