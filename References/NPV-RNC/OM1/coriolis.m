function C = coriolis(q,qd,p)

s2 = sin(q(2)); c2 = cos(q(2));
s3 = sin(q(3)); c3 = cos(q(3));

a3 = p.Id - p.Jc - p.Jd + p.Kc;
a4 = p.Ic + p.Id;
a5 = p.Ib + p.Ic - p.Kb - p.Kc;

Qd = blkdiag(qd',qd',qd',qd');

Lambda1 = 0.5*[0,0,0,0;0,0,-p.Jd*s2,p.Jd*c2*c3;0,0,0,-p.Jd*s2*s3;0,0,0,0]; 
Lambda1 = Lambda1+Lambda1';
Lambda2 = 0.5*[0,0,p.Jd*s2,-p.Jd*c2*c3;0,0,0,0;0,0,-a3*s2*c2,a3*(c2^2*c3-s2^2*c3)-a4*c3;0,0,0,.5*a3*c2*c3^2*s2];
Lambda2 = Lambda2+Lambda2';
Lambda3 = 0.5*[0,-p.Jd*s2,0,p.Jd*s2*s3;0,0,2*a3*s2*c2,a4*c3+a3*(c3*s2^2-c2^2*c3);0,0,0,0;0,0,0,-0.5*(a5+a3*s2^2)*c3*s3];
Lambda3 = Lambda3+Lambda3';
Lambda4 = 0.5*[0,p.Jd*c2*c3,-p.Jd*s2*s3,0;0,0,a3*(c3*s2^2-c2^2*c3)-a4*c3,-a3*c2*c3^2*s2;0,0,.5*a3*c2*s2*s3,(a5+a3*s2^2)*c3*s3;0,0,0,0];
Lambda4 = Lambda4+Lambda4';

C = Qd*[Lambda1;Lambda2;Lambda3;Lambda4];

end