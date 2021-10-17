clear; close all
load('LPV_setpoint_1_perf.mat');
load('LPV_setpoint_2_perf.mat');
load('NPV_setpoint_perf.mat');

lw=1.5; fz=12; lfz=16;
figure;
subplot(311);hold on; k=2;
plot(LPV_setpoint_1.t,LPV_setpoint_1.xr(k,:),'k--');
plot(LPV_setpoint_1.t,LPV_setpoint_1.x(k,:),'r','linewidth',lw);
plot(LPV_setpoint_2.t,LPV_setpoint_2.x(k,:),'b','linewidth',lw);
plot(NPV_setpoint.t,NPV_setpoint.x(k,:),'color',[0,0.3,0],'linewidth',lw);
legend('Ref','LPV1','LPV2','NPV');
grid on;
set(gca,'fontsize',fz);
ylabel('$\dot{q}_1$ [rad/s]','interpreter','latex','fontsize',lfz);

subplot(312);hold on; 
plot(LPV_setpoint_1.t,LPV_setpoint_1.q2,'r','linewidth',lw);
plot(LPV_setpoint_2.t,LPV_setpoint_2.q2,'b','linewidth',lw);
plot(NPV_setpoint.t,NPV_setpoint.q2,'color',[0,0.3,0],'linewidth',lw);
% ylim([-0.8,0.6]);
grid on;
set(gca,'fontsize',fz);
% xlabel('Time [s]','interpreter','latex','fontsize',lfz);
ylabel('$q_2$ [rad]','interpreter','latex','fontsize',lfz);

subplot(313);hold on; k=1;
plot(LPV_setpoint_1.t,LPV_setpoint_1.xr(k,:),'k--');
plot(LPV_setpoint_1.t,LPV_setpoint_1.x(k,:),'r','linewidth',lw);
plot(LPV_setpoint_2.t,LPV_setpoint_2.x(k,:),'b','linewidth',lw);
plot(NPV_setpoint.t,NPV_setpoint.x(k,:),'color',[0,0.3,0],'linewidth',lw);
grid on;
set(gca,'fontsize',fz);
xlabel('Time [s]','interpreter','latex','fontsize',lfz);
ylabel('$q_4$ [rad]','interpreter','latex','fontsize',lfz);



