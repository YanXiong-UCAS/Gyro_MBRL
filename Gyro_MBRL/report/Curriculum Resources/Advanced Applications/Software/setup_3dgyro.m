% SETUP_LAB_3DOF_GYRO
%
% 3 DOF Gyroscope (3DGYRO) Control Lab: 
% Design of a LQR+I position controller
% 
% SETUP_LAB_3DOF_GYRO sets the model parameters and set the controller
% parameters for the Quanser 3DOF Gyroscope system.
%
% Copyright (C) 2013 Quanser Consulting Inc.
% Quanser Consulting Inc.
%
clear all
%
% ############### USER-DEFINED 2DOF HELI CONFIGURATION ###############
%CONTROL TYPE CAN BE SET TO 'AUTO' OR 'MANUAL'. 
CONTROL_TYPE = 'AUTO';
%AMPAQ MAXIMUM OUTPUT VOLTAGE (V)
V_MAX = 28;
%DAQ MAXIMUM & MINIMUM VOLTAGE (V)
DAC_LIMIT_MAX = 10;
DAC_LIMIT_MIN = -DAC_LIMIT_MAX;
%MAXIMUM & MINIMUM ALLOWABLE CURRENT  (A)
CURRENT_LIMIT_MAX = 3;
CURRENT_LIMIT_MIN = -CURRENT_LIMIT_MAX;
%MAXIMUM ALLOWABLE SIGMOID VELOCITY (rad/s)
SIGMOID_VEL_MAX = 60;
%MAXIMUM ALLOWABLE SIGMOID ACCELERATION (rad/s^2)
SIGMOID_ACCEL_MAX = 60;
%REFERENCE RPM FOR THE DISK (RPM
REFERENCE_RPM = 750;
%
% ###############MODEL PARAMETERS###############
%
s=tf('s');
%MOMENT OF INERTIA AROUND Y-AXIS
%Jy=.0007+.0019;
Jy = 0.0028 + 0.0012;
%MOMENT OF INERTIA AROUND Z-AXIS
%Jz=.0089+.0205+.0019+.0029;
Jz = 0.0175 + 0.002 + 0.0028;

%h=.0052*2000*2*pi/60;
h = 0.0056 * 750 * 2 * pi / 60;
%OPEN-LOOP GYROSCOPE TRANSFER FUNCTION
G=h/(Jy*Jz*s^3+h^2*s);

Kt = 0.115;
%
% Set the model parameters of the 3DOF GYRO.
%This section sets the A,B,C and D matrices for the gyroscope model. It also

A = [0 0 -h/Jy 0; 0 0 1 0; h/Jz 0 0 0; 0 1 0 0];
B = [1/Jy; 0; 0; 0];
C = [0 1 0 0];
D = 0;

C_s = 1.23*(1-0.16*s)/(1+0.038*s);

if strcmp(CONTROL_TYPE, 'AUTO') 
Q = diag([2, 16, 0.01, 0.0001]);
R = 5;
K = lqr(A,B,Q,R);

%LQR CONTROL DESIGN FOR DISK ONLY
%This section sets the transfer function for a tuned LEAD controller as well as the
%state-feedback gain for a tuned LQR controller.
A_w = 0;
B_w = 1/Jy;

Q_w = 0.5;
R_w = 1;
K1 = lqr(A_w, B_w, Q_w, R_w);

elseif strcmp(CONTROL_TYPE, 'MANUAL')
    disp( [ 'K = [' 0 ' V/rad  '  0 ' V/rad  ' 0 ' V.s/rad  '  0 ' V.s/rad]'] )
    disp( ' ' )
    disp( 'STATUS: manual mode' ) 
    disp( 'The model parameters of your Gyro system have been set.' )
    disp( 'You can now design your state-feedback position controller.' )
    disp( ' ' )
end


    
    





